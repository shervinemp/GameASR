from functools import partial
import asyncio
import inspect
import json
import os
from queue import Queue
import threading
from typing import Any, Dict, Generator, Optional, Tuple

from ..common.utils import get_logger
from ..exceptions import LLMError, ToolError

from .model import LLM
from .conversation import Conversation
from .tools import ToolCall


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

    def reset(self, conversation: Optional[Conversation] = None):
        """Reset conversation and provider state without replacing the session."""
        with self._lock:
            self.conversation = conversation or Conversation()
            self._session_state.clear()

    def close(self):
        """Release the background tool execution loop."""
        self.tool_caller.stop()

    def save(self, path: str, *, save_kv_cache: bool = False):
        """Save session to a directory."""
        os.makedirs(path, exist_ok=True)
        self.conversation.save(os.path.join(path, "conversation.json"))
        state = {}
        with self._lock:
            for k, v in self._session_state.items():
                if k == "model_state" and not save_kv_cache:
                    continue
                if isinstance(v, (bytes, bytearray)):
                    continue
                state[k] = v
        with open(os.path.join(path, "state.json"), "w") as f:
            json.dump(state, f, indent=2)

        if save_kv_cache:
            state_data = self._session_state.get("model_state")
            if isinstance(state_data, (bytes, bytearray)):
                with open(os.path.join(path, "kv_cache.bin"), "wb") as f:
                    f.write(state_data)

    @classmethod
    def load(cls, path: str, llm: LLM) -> "Session":
        """Load session from a directory with a fresh LLM instance."""
        conv_path = os.path.join(path, "conversation.json")
        state_path = os.path.join(path, "state.json")
        kv_path = os.path.join(path, "kv_cache.bin")

        conversation = Conversation.load(conv_path) if os.path.exists(conv_path) else Conversation()
        session = cls(llm=llm, conversation=conversation)

        if os.path.exists(state_path):
            with open(state_path) as f:
                state = json.load(f)
                session._session_state.update(state)

        if os.path.exists(kv_path):
            with open(kv_path, "rb") as f:
                session._session_state["model_state"] = f.read()

        return session

    def complete_once(
        self,
        query: str,
        *,
        system: str | None = None,
        **kwargs,
    ) -> str:
        """Run an isolated completion without mutating conversation history."""
        if not isinstance(query, str) or not query.strip():
            raise LLMError("One-shot queries must be non-empty strings.")

        conversation = Conversation()
        if system:
            conversation.set_system_message(system)
        conversation.add_user_message(query)
        state = {}
        response_chunks = []

        # ASVS 15.4.1 / 15.4.3: the shared provider and its model state are
        # accessed under the Session-owned lock, just like normal completions.
        with self._lock:
            for chunk in self.llm(
                conversation,
                session_state=state,
                tool_choice="none",
                **kwargs,
            ):
                if isinstance(chunk, ToolCall):
                    self.logger.warning(
                        "Ignored an unexpected tool call during one-shot inference."
                    )
                    continue
                response_chunks.append(chunk)
        return "".join(response_chunks).strip()

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
                has_error = False
                for k, v in tool_responses.items():
                    if isinstance(v, str) and v.startswith("Tool Error:"):
                        has_error = True
                    self.conversation.add_tool_message(f"{k}: {v}")

                if has_error:
                    self.conversation.add_tool_message(
                        "One or more tools failed to execute. Please inform the user of the error and suggest an alternative action."
                    )
                else:
                    self.conversation.add_tool_message(
                        "Now, generate an answer based only on the returned responses."
                    )
                yield from self._generate_response(tool_choice="none")

    def _generate_response(self, _retry_count: int = 0, **kwargs) -> Generator[str, None, None]:
        if _retry_count > 2:
            self.logger.error("Too many tool parse retries. Aborting.")
            yield "I encountered a persistent error processing tool calls."
            return

        response_chunks = []
        for chunk in self.llm(
            self.conversation, session_state=self._session_state, **kwargs
        ):
            if isinstance(chunk, ToolCall):
                try:
                    tool_name = chunk.name
                    tool_args = chunk.arguments
                except Exception as e:
                    self.logger.warning(f"Error parsing tool response: {repr(e)}")
                    self.conversation.add_tool_message(
                        "Tool Error: Could not parse tool call. Please try again with valid JSON."
                    )
                    yield from self._generate_response(_retry_count=_retry_count + 1, tool_choice="none")
                    return
                tool = self.conversation.tools.get(tool_name)
                if tool is None:
                    self.logger.warning(f"Tool '{tool_name}' not found in conversation tools.")
                    self.conversation.add_tool_message(
                        f"Tool Error: '{tool_name}' is not a recognized tool. Available: {list(self.conversation.tools.keys())}."
                    )
                    yield from self._generate_response(_retry_count=_retry_count + 1, tool_choice="none")
                    return
                self.tool_caller(tool, **tool_args)
            else:
                response_chunks.append(chunk)
                yield chunk

        final_response = "".join(response_chunks)
        if final_response:
            self.conversation.add_assistant_message(final_response)
            self.logger.info(f"response={final_response}")


class ToolCaller:

    def __init__(self):
        self._loop: asyncio.AbstractEventLoop = None
        self._loop_thread: threading.Thread = None
        self._loop_ready_event = threading.Event()
        self._futures: Queue[Tuple[str, asyncio.Future]] = None
        self.logger = get_logger(__name__)

    def __call__(self, tool, **tool_args):
        tool_callable = partial(tool.__call__, **tool_args)

        if asyncio.iscoroutinefunction(tool.callback) or inspect.iscoroutinefunction(getattr(tool_callable, "func", None)):
            future = asyncio.run_coroutine_threadsafe(
                tool_callable(), self._loop
            )
        else:
            future = asyncio.run_coroutine_threadsafe(
                asyncio.to_thread(tool_callable), self._loop
            )
        self._futures.put((tool.name, future))

    def gather(self) -> Dict[str, Any]:
        responses = {}
        while not self._futures.empty():
            tool_name, future = self._futures.get()
            try:
                responses[tool_name] = future.result(timeout=10.0)
            except TimeoutError:
                future.cancel()
                responses[tool_name] = f"Tool Error: {tool_name} timed out after 10s"
                self.logger.error(f"Tool {tool_name} timed out")
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
            loop.call_soon_threadsafe(self._loop_ready_event.set)
            loop.run_forever()
            loop.close()
            self._loop = None
            self._loop_ready_event.clear()

        self._futures = Queue()
        self._loop_thread = threading.Thread(target=run_loop, daemon=True)
        self._loop_thread.start()

        self._loop_ready_event.wait(timeout=5.0)
        if not self._loop_ready_event.is_set():
            raise LLMError(
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
        try:
            self.stop()
        except Exception:
            # Destructors may run during interpreter teardown.
            pass
