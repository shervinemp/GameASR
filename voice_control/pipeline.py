import sys
from typing import Optional

from dotenv import dotenv_values

from .asr import ParakeetV2
from .llm import Session
from .llm.tools import Tool
from .tts import TTS
from .rag import RAG
from .rag.graph import KnowledgeGraph
from .bridge.rpc_server import LLMService, RpcServer

from .common.base import stream_splitter
from .common.utils import setup_logging, get_logger


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

        self.asr = ParakeetV2()
        self.session = session or Session()
        if rag is not None:
            rag_tool = Tool.from_callable("retrieve", rag)
            self.session.conversation._tools.update({"rag": rag_tool})
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
                self._callback(transcript)
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

    env = dotenv_values(".env")
    NEO4J_URI = env.get("NEO4J_URI")
    NEO4J_USER = env.get("NEO4J_USER")
    NEO4J_PASSWORD = env.get("NEO4J_PASSWORD")

    if not all([NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD]):
        raise ValueError("Neo4j credentials not found in .env file.")

    try:
        graph = KnowledgeGraph(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        rag = RAG(graph)
        pipe = Pipeline(rag=rag)
        logger.info("Starting voice control pipeline...")
        pipe.run()
    except Exception as e:
        logger.error(f"Error in main(): {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
