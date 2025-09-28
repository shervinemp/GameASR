from contextlib import contextmanager
import os
import sys
from typing import Optional
from dotenv import load_dotenv

from .asr.model import ASRProviders
from .tts.model import TTSProviders
from .rag.model import SimpleRAG
from .hotkey_dispatcher import HotkeyDispatcher
from .llm import Session, LLMProviders
from .llm.tools import Tool
from .rag import RAG
from .rag.knowledge import KnowledgeGraph
from .bridge.llm_server import LLMServer, LLMService

from .common.base import stream_splitter
from .common.utils import setup_logging, get_logger
from .common.config import config


class Pipeline:
    """
    Orchestrates the integration between ASR, LLM, and TTS components.
    """

    def __init__(
        self,
        session: Optional[Session] = None,
        rag: Optional[RAG] = None,
        server_endpoint: str | None = None,
        push_to_talk: str | None = None,
    ):
        """
        Initialize the voice control pipeline.
        """
        self.logger = get_logger(__name__)

        self._running = False

        asr_cls = getattr(ASRProviders, config.get("asr.provider"))
        self.asr = asr_cls()

        tts_cls = getattr(TTSProviders, config.get("tts.provider"))
        self.tts = tts_cls()

        llm_cls = getattr(LLMProviders, config.get("llm.provider"))
        self.session = session or Session(llm=llm_cls())

        self.rag = rag

        if server_endpoint:
            auth_token = config.get("llm_server.auth_token")
            self.llm_server = LLMServer(
                LLMService(self.session),
                endpoint=server_endpoint,
                auth_token=auth_token,
            )

        self.push_to_talk = push_to_talk

    @property
    def rag(self) -> RAG:
        return self.__rag

    @rag.setter
    def rag(self, value: RAG):
        if value:
            name = "retrieve"
            rag_tool = Tool.from_callable(name, value)
            self.session.conversation._tools.update({name: rag_tool})
        self.__rag = value

    @property
    def push_to_talk(self):
        return self.__p2t

    @push_to_talk.setter
    def push_to_talk(self, value: str | None):
        self.__p2t = value

        if dispatcher := getattr(self, "_hk_dispatch", None):
            dispatcher.stop()
            del self._hk_dispatch

        if value is None:
            self.asr.enable()
            self.logger.info("Push-to-talk disabled")
        else:

            @contextmanager
            def p2t_action():
                """Action to enable/disable ASR on hotkey press/release."""
                self.asr.enable()
                self.logger.info("Push-to-talk ACTIVE. Listening...")
                yield
                self.asr.disable_w_passthrough()
                self.logger.info("Push-to-talk RELEASED. ASR muted.")

            self.asr.disable_w_passthrough()
            dispatcher = HotkeyDispatcher()
            dispatcher.register(value, p2t_action)
            self._hk_dispatch = dispatcher

            if self._running:
                dispatcher.start()

            self.logger.info(f"Push-to-talk enabled with hotkey '{value}'")

    def _callback(self, transcription: str):
        """
        Process transcribed text through LLM and then send to TTS.
        """
        if not transcription:
            return

        interrupt = True
        out = self.session(transcription)
        for sentence in stream_splitter(out, min_len=8):
            if s := sentence.strip():
                self.tts(s, interrupt=interrupt)
                interrupt = False

    def run(self):
        """Start the voice control pipeline."""
        self.asr.start()
        self.tts.start()
        if server := getattr(self, "llm_server", None):
            server.start()
        if dispatcher := getattr(self, "_hk_dispatch", None):
            dispatcher.start()
        self._running = True
        try:
            for transcript in self.asr:
                self.logger.debug(f"{transcript=}")
                try:
                    self._callback(transcript)
                except Exception as e:
                    self.logger.error(f"Error in callback: {e}", exc_info=True)
        finally:
            self._running = False
            if server := getattr(self, "llm_server", None):
                server.stop()
            if dispatcher := getattr(self, "_hk_dispatch", None):
                dispatcher.stop()
            self.asr.stop()
            self.tts.stop()


def main():
    """
    Main function to run the pipeline.
    """
    setup_logging(log_level="DEBUG")
    logger = get_logger(__name__)

    load_dotenv()

    neo4j_config = config.get("database.neo4j")
    if not neo4j_config:
        raise ValueError("Neo4j configuration not found in config file.")

    uri = neo4j_config.uri
    user = neo4j_config.user
    password_env_var = neo4j_config.password_env

    if not password_env_var:
        raise ValueError(
            "Neo4j password environment variable not specified in config."
        )

    password = os.getenv(password_env_var)

    if not all([uri, user, password]):
        raise ValueError(
            f"Neo4j credentials not fully configured. Check your config file and the '{password_env_var}' environment variable."
        )

    try:
        graph = KnowledgeGraph(uri, user, password)
        pipe = Pipeline(push_to_talk="<ctrl_r>+<shift_r>")
        llm = pipe.session.llm
        rag = SimpleRAG(llm=llm, graph=graph, web_search=True)
        pipe.rag = rag
        logger.info("Starting voice control pipeline...")
        pipe.run()
    except Exception as e:
        logger.error(f"Error in main(): {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
