import asyncio
from contextlib import contextmanager
import re
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

from .common.utils import setup_logging, get_logger
from .common.config import config


class Pipeline:
    """
    Orchestrates the integration between ASR, LLM, and TTS components
    using an asynchronous event loop.
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
        else:
            self.llm_server = None

        self.__push_to_talk = None
        self.__press_to_reset = None
        # Hotkey dispatcher setup
        self._hk_dispatch = HotkeyDispatcher()

        # Initialize properties
        if push_to_talk:
            self.push_to_talk = push_to_talk
        if press_to_reset:
            self.press_to_reset = press_to_reset

        # Async state
        self.input_queue = asyncio.Queue()
        self.current_interaction_task: Optional[asyncio.Task] = None

    async def run_async(self):
        """Async entry point."""
        self.asr.start()
        self.tts.start()
        self._hk_dispatch.start()
        if self.llm_server:
            self.llm_server.start()

        self._running = True
        loop = asyncio.get_running_loop()

        # Start ASR monitoring task
        asr_task = asyncio.create_task(self.monitor_asr(loop))

        # Start Processing Loop
        process_task = asyncio.create_task(self.process_queue(loop))

        self.logger.info("Pipeline started (Async).")

        try:
            # Keep alive until stopped
            while self._running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            self._running = False
            asr_task.cancel()
            process_task.cancel()
            if self.current_interaction_task:
                self.current_interaction_task.cancel()

            if self.llm_server:
                self.llm_server.stop()
            self._hk_dispatch.stop()
            self.asr.stop()
            self.tts.stop()

    async def monitor_asr(self, loop):
        """Runs the blocking ASR iterator in a thread."""
        def asr_producer():
            for transcript in self.asr:
                if transcript:
                    asyncio.run_coroutine_threadsafe(
                        self.input_queue.put(transcript), loop
                    )

        await loop.run_in_executor(None, asr_producer)

    async def process_queue(self, loop):
        """Consumes ASR transcripts and spawns processing tasks."""
        while True:
            text = await self.input_queue.get()
            self.logger.debug(f"Received input: {text}")

            # Interruption Logic: Cancel current processing if active
            if self.current_interaction_task and not self.current_interaction_task.done():
                self.logger.info("Interrupting current interaction.")
                self.current_interaction_task.cancel()
                # Stop current TTS audio immediately
                # self.tts calls AudioPlayer. We assume calling with interrupt=True works,
                # but here we just want to stop output.
                # Since we are starting a NEW interaction, the new TTS calls will handle interrupt if passed.
                # Or we can explicitly stop.
                # Given current TTS implementation, calling self.tts with interrupt=True stops previous.
                # We can do that in the new task.

            self.current_interaction_task = asyncio.create_task(
                self.handle_interaction(text, loop)
            )

    async def handle_interaction(self, text: str, loop):
        """Handles LLM generation and TTS streaming."""
        queue = asyncio.Queue()
        sentinel = object()
        interrupt_audio = True

        def llm_producer():
            try:
                # Synchronous LLM call
                out = self.session(text)
                for chunk in out:
                    asyncio.run_coroutine_threadsafe(queue.put(chunk), loop)
            except Exception as e:
                self.logger.error(f"LLM Error: {e}", exc_info=True)
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(sentinel), loop)

        # Start LLM in background thread
        loop.run_in_executor(None, llm_producer)

        # Async Stream Splitter Logic
        sentences = re.compile(r"[^.][.!?]\s+")
        buffer = ""
        min_len = 8

        try:
            while True:
                chunk = await queue.get()
                if chunk is sentinel:
                    break

                buffer += chunk
                # Check for sentences
                if len(buffer) >= min_len:
                    # Implementation matching original stream_splitter logic but async compatible
                    while True:
                         match = sentences.search(buffer)
                         if not match:
                             break

                         end = match.end()
                         sentence = buffer[:end]
                         buffer = buffer[end:]

                         if s := sentence.strip():
                             await self.speak(s, loop, interrupt=interrupt_audio)
                             interrupt_audio = False # Only first chunk interrupts

            if s := buffer.strip():
                await self.speak(s, loop, interrupt=interrupt_audio)

        except asyncio.CancelledError:
            self.logger.debug("Interaction cancelled.")
            # Verify TTS stops?
            # We can optionally call self.tts(..., interrupt=True) with empty string to stop?
            pass

    async def speak(self, text: str, loop, interrupt: bool = False):
        await loop.run_in_executor(None, self.tts, text, "af_heart", "en-us", 1.0, interrupt)

    def run(self):
        """Start the voice control pipeline."""
        try:
            asyncio.run(self.run_async())
        except KeyboardInterrupt:
            pass

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
        dispatcher = self._hk_dispatch
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
        dispatcher = self._hk_dispatch
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
        return self._hk_dispatch


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
