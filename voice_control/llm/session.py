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
        llm: Optional[LLM] = None,
        conversation: Optional[Conversation] = None,
    ):
        self.llm = llm or LLM()
        self.conversation = conversation or Conversation()
        self.tool_caller = ToolCaller()
        self.tool_caller.start()

    def __call__(self, query: str, cutoff_idx: int = 0) -> Generator[str, None, None]:
        self.conversation.add_user_message(query)
        messages = self.conversation.messages[cutoff_idx:]

        response = ""
        for chunk in self.llm(messages):
            if isinstance(chunk, dict):
                tool_name = chunk["name"]
                tool_args = chunk["args"]
                tool = self.conversation.tools[tool_name]
                self.tool_caller(tool, **tool_args)
            else:
                response += chunk
                yield chunk

        tool_responses = self.tool_caller.gather()

        tool_response = ""
        if tool_responses:
            tool_response = "\n".join(
                f'<response name="{k}">{v}</response>'
                for k, v in tool_responses.items()
            )

        if response:
            self.conversation.add_assistant_message(response)

        if tool_response:
            self.conversation.add_assistant_message(tool_response)

            response = ""
            messages = self.conversation.messages[
                cutoff_idx - 2 if cutoff_idx < 0 else cutoff_idx :
            ]
            for chunk in self.llm(messages, force_toolcall=False):
                response += chunk
                yield chunk

            if response:
                self.conversation.add_assistant_message(response)


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
        while self._futures:
            tool_name, future = self._futures.get()
            try:
                responses[tool_name] = future.result()
            except Exception as e:
                responses[tool_name] = f"Tool Error: {e}"
                self.logger.error(f"Error calling tool {tool_name}: {e}")

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
            self._loop_thread.join(timeout=5.0)
            if self._loop_thread.is_alive():
                self.logger.warning("Background loop thread did not stop gracefully.")
            self._loop_thread = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False

    def __del__(self):
        self.stop()
