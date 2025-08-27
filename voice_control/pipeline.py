import os
import sys
from typing import Optional

from .asr import get_model_class
from .llm import Session
from .llm.tools import Tool
from .tts import TTS
from .rag import RAG
from .rag.knowledge_base import KnowledgeGraph
from .bridge.rpc_server import LLMService, RpcServer

from .common.base import stream_splitter
from .common.utils import setup_logging, get_logger
from .common.config import config


class Pipeline:
    """
    Orchestrates the integration between ASR, LLM, and TTS components.

    This class initializes all three components and sets up a pipeline where:
    - ASR captures audio and transcribes it to text
    - The transcribed text is passed to LLM for processing
    - The LLM response is sent to TTS for audio output
    """

    def __init__(
        self,
        session: Optional[Session] = None,
        rag: Optional[RAG] = None,
        rpc_server: str | None = None,
    ):
        """
        Initialize the voice control pipeline with ASR, LLM, and TTS components,
        and dynamically set up tool execution based on the game API spec.

        Args:
            session: An optional session object for the LLM.
            rpc_server: An optional RPC server endpoint for the LLM service.
        """
        self.logger = get_logger(__name__)

        asr_provider = config.get("asr.provider", "parakeetv2")
        AsrModel = get_model_class(asr_provider)
        self.asr = AsrModel()
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

    def _callback(self, transcription: str):
        """
        Process transcribed text through LLM, handle tool calls, and then send to TTS.
        This is the main callback for ASR output.
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
            self.asr.stop()
            self.tts.stop()


def main():
    """
    Main function to run the pipeline.
    """
    setup_logging(log_level="DEBUG")
    logger = get_logger(__name__)

    # Load Neo4j credentials from the central config
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
        rag = RAG(graph)
        pipe = Pipeline(rag=rag)
        logger.info("Starting voice control pipeline...")
        pipe.run()
    except Exception as e:
        logger.error(f"Error in main(): {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
