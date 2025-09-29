from functools import partial
import asyncio
from queue import Queue
import threading
from typing import Any, Dict, Generator, Optional, Tuple

from ..common.utils import get_logger

from .model import LLM
from .conversation import Conversation


class Session:

    def __init__(
        self,
        llm: LLM,
        conversation: Optional[Conversation] = None,
    ):
        self.logger = get_logger(__name__)
        self.llm = llm
        self.conversation = conversation or Conversation()
        self.tool_caller = ToolCaller()

        self._session_state = dict()
        self._lock = threading.Lock()

        self.tool_caller.start()

    def __call__(
        self, query: str | None = None, **kwargs
    ) -> Generator[str, None, None]:
        with self._lock:
            if query:
                self.conversation.add_user_message(query)
                self.logger.info(f"{query=}")

            yield from self._generate_response()

            tool_responses = self.tool_caller.gather()
            if tool_responses:
                for k, v in tool_responses.items():
                    self.conversation.add_tool_message(f"{k}: {v}")

                self.conversation.add_tool_message(
                    "Now, generate an answer based only on the returned responses."
                )
                yield from self._generate_response(tool_choice="none")

    def _generate_response(self, **kwargs) -> Generator[str, None, None]:
        response = ""
        for chunk in self.llm(
            self.conversation, session_state=self._session_state, **kwargs
        ):
            if isinstance(chunk, dict):
                try:
                    tool_name = chunk.get("name", None) or chunk["function"]
                    tool_args = chunk["arguments"]
                except Exception as e:
                    self.logger.warning(
                        f"Error parsing tool response: {repr(e)}"
                    )
                    continue
                tool = self.conversation.tools[tool_name]
                self.tool_caller(tool, **tool_args)
            else:
                response += chunk
                yield chunk

        if response:
            self.conversation.add_assistant_message(response)
            self.logger.info(f"{response=}")


class ToolCaller:

    def __init__(self):
        self._loop: asyncio.AbstractEventLoop = None
        self._loop_thread: threading.Thread = None
        self._loop_ready_event = threading.Event()
        self._futures: Queue[Tuple[str, asyncio.Future]] = None
        self.logger = get_logger(__name__)

    def __call__(self, tool, **tool_args):
        sync_tool_callable = partial(tool.__call__, **tool_args)

        future = asyncio.run_coroutine_threadsafe(
            asyncio.to_thread(sync_tool_callable), self._loop
        )
        self._futures.put((tool.name, future))

    def gather(self) -> Dict[str, Any]:
        responses = {}
        while not self._futures.empty():
            tool_name, future = self._futures.get()
            try:
                responses[tool_name] = future.result()
            except Exception as e:
                responses[tool_name] = f"Tool Error: {e}"
                self.logger.error(
                    f"Error calling tool {tool_name}: {e}", exc_info=True
                )

        return responses

    def start(self):
        """Starts the background asyncio event loop for tool execution."""
        if self._loop_thread and self._loop_thread.is_alive():
            return

        def run_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            self._loop_ready_event.set()
            loop.run_forever()
            loop.close()
            self._loop = None
            self._loop_ready_event.clear()

        self._futures = Queue()
        self._loop_thread = threading.Thread(target=run_loop, daemon=True)
        self._loop_thread.start()

        self._loop_ready_event.wait(timeout=5.0)
        if not self._loop_ready_event.is_set():
            raise RuntimeError(
                "Failed to start background asyncio loop within timeout."
            )

    def stop(self):
        """Stops the background asyncio event loop gracefully."""
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._loop_thread.join(timeout=5)
            if self._loop_thread.is_alive():
                self.logger.warning(
                    "Background loop thread did not stop gracefully."
                )
            self._loop_thread = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False

    def __del__(self):
        self.stop()
