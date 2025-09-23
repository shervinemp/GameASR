from contextlib import contextmanager
import os
import sys
from typing import Optional
from dotenv import load_dotenv
from voice_control.rag.model import SimpleRAG
from .hotkey_dispatcher import HotkeyDispatcher
from .asr import default_class as default_asr
from .llm import Session, default_class as default_llm
from .llm.tools import Tool
from .tts import TTS
from .rag import RAG
from .rag.knowledge import KnowledgeGraph
from .bridge.rpc_server import LLMService, RpcServer

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
        rpc_server: str | None = None,
        p2t_key: str | None = None,
    ):
        """
        Initialize the voice control pipeline.
        """
        self.logger = get_logger(__name__)

        self.asr = default_asr()
        self.session = session or Session()
        if rag is not None:
            name = "retrieve"
            rag_tool = Tool.from_callable(name, rag)
            self.session.conversation._tools.update({name: rag_tool})
        self.tts = TTS()
        self.rpc_server = (
            RpcServer(LLMService(self.session), endpoint=rpc_server)
            if rpc_server
            else None
        )

        if p2t_key:

            @contextmanager
            def p2t_action():
                """Action to enable/disable ASR on hotkey press/release."""
                self.asr.enable()
                self.logger.info("Push-to-talk ACTIVE. Listening...")
                yield
                self.asr.disable_w_passthrough()
                self.logger.info("Push-to-talk RELEASED. ASR muted.")

            self.asr.disable_w_passthrough()
            hotkey_dispatcher = HotkeyDispatcher()
            hotkey_dispatcher.register(p2t_key, p2t_action)
            self.hotkey_dispatcher = hotkey_dispatcher
            self.logger.info(f"Push-to-talk enabled with hotkey '{p2t_key}'")

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
        if self.hotkey_dispatcher:
            self.hotkey_dispatcher.start()

        self.asr.start()
        self.tts.start()
        if self.rpc_server:
            self.rpc_server.start()
        try:
            for transcript in self.asr:
                self.logger.debug(f"{transcript=}")
                try:
                    self._callback(transcript)
                except Exception as e:
                    self.logger.error(f"Error in callback: {e}", exc_info=True)
        finally:
            if self.rpc_server:
                self.rpc_server.stop()
            if self.hotkey_dispatcher.hotkeys:
                self.hotkey_dispatcher.stop()
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

    uri = neo4j_config.get("uri")
    user = neo4j_config.get("user")
    password_env_var = neo4j_config.get("password_env")

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
        llm = default_llm()
        rag = SimpleRAG(llm=llm, graph=graph, web_search=True)
        session = Session(llm=llm)
        pipe = Pipeline(session=session, rag=rag, p2t_key="<ctrl_r>+<shift_r>")
        logger.info("Starting voice control pipeline...")
        pipe.run()
    except Exception as e:
        logger.error(f"Error in main(): {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
