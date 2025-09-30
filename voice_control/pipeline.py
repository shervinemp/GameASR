from contextlib import contextmanager
import os
import sys
from typing import Optional
from dotenv import load_dotenv

from .llm.conversation import Conversation

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
        press_to_reset: str | None = None,
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

        llm_provider = config.get("llm.provider")
        llm_cls = getattr(LLMProviders, llm_provider)
        # Environment variables are resolved by the Config class
        llm_settings = config.get("llm.providers").get(llm_provider.lower(), {})
        self.session = session or Session(llm=llm_cls(**llm_settings))

        self.rag = rag

        if server_endpoint:
            auth_token = config.get("llm_server.auth_token")
            self.llm_server = LLMServer(
                LLMService(self.session),
                endpoint=server_endpoint,
                auth_token=auth_token,
            )

        self.push_to_talk = push_to_talk
        self.press_to_reset = press_to_reset

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
        self._hotkey_dispatcher.start()
        if server := getattr(self, "llm_server", None):
            server.start()
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

    def _configure_session(self):
        if cb := getattr(self, "_rag", None):
            name = "retrieve"
            rag_tool = Tool.from_callable(name, cb)
            self.session.conversation._tools.update({name: rag_tool})

    @property
    def rag(self) -> RAG:
        return self._rag

    @rag.setter
    def rag(self, value: RAG):
        self._rag = value
        self._configure_session()

    @property
    def push_to_talk(self):
        return self.__push_to_talk

    @push_to_talk.setter
    def push_to_talk(self, value: str | None):
        dispatcher = self._hotkey_dispatcher
        if k_ := getattr(self, "__push_to_talk", None):
            dispatcher.unregister(k_)

        self.__push_to_talk = value
        if value is None:
            self.asr.enable()
            self.logger.info("Push-to-talk disabled")
        else:

            @contextmanager
            def cb():
                self.asr.enable()
                self.logger.info("Push-to-talk ACTIVE. Listening...")
                yield
                self.asr.disable_w_passthrough()
                self.logger.info("Push-to-talk RELEASED. ASR muted.")

            self.asr.disable_w_passthrough()
            dispatcher.register(value, cb)
            self.logger.info(f"Push-to-talk enabled with hotkey '{value}'")

    @property
    def press_to_reset(self):
        return self.__press_to_reset

    @press_to_reset.setter
    def press_to_reset(self, value: str | None):
        dispatcher = self._hotkey_dispatcher
        if k_ := getattr(self, "__press_to_reset", None):
            dispatcher.unregister(k_)

        self.__press_to_reset = value
        if value:

            def cb():
                self.logger.info("Conversation reset.")
                self.session = Session(
                    llm=self.session.llm, conversation=Conversation()
                )
                self._configure_session()

            dispatcher.register(value, cb)
            self.logger.info(f"Press-to-reset enabled with hotkey '{value}'")

    @property
    def _hotkey_dispatcher(self) -> HotkeyDispatcher:
        attr_name = "__hotkey_dispatcher"
        if dispatcher := getattr(self, attr_name, None):
            return dispatcher
        else:
            dispatcher = HotkeyDispatcher()
            setattr(self, attr_name, dispatcher)
            return dispatcher


def main():
    """
    Main function to run the pipeline.
    """
    setup_logging(log_level="DEBUG")
    logger = get_logger(__name__)

    load_dotenv()

    neo4j_config = config.get("database.neo4j")
    if not all([neo4j_config.uri, neo4j_config.user, neo4j_config.password]):
        raise ValueError(
            "Neo4j credentials not fully configured. Check your config file and environment variables."
        )

    try:
        graph = KnowledgeGraph(
            neo4j_config.uri, neo4j_config.user, neo4j_config.password
        )
        graph.verify_connectivity()
    except Exception as e:
        graph = None
        logger.warning(
            f"Skipping knowledge graph initialization due to error: {e}"
        )

    try:
        pipe = Pipeline(
            push_to_talk="<ctrl_r>+<shift_r>",
            press_to_reset="<ctrl_l>+<ctrl_r>",
        )
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
