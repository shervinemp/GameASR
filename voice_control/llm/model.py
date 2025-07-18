"""
Core module for LLM functionality.

This module provides the main classes and functions for working with language models.
"""

from collections import defaultdict
import json
import os
from typing import Any, Dict, Generator

from llama_cpp import (
    CreateChatCompletionResponse,
    CreateChatCompletionStreamResponse,
    Iterator,
    Llama,
    Union,
)

from .conversation import Conversation

from ..common.utils import get_logger


class LLM:

    def __init__(self):
        self.logger = get_logger(__name__)
        model_filename = "Nemotron-Mini-4B-Instruct-Q4_K_M.gguf"
        model_path = os.path.join("models", "llm", model_filename)
        self.max_tokens = 4096
        self.stream_processor = self._parse

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Please run the download script first."
            )

        self.logger.info("Loading model...")
        self.model = Llama(
            model_path=model_path,
            n_ctx=4096,  # Context window size
            n_gpu_layers=-1,  # Offload all layers to GPU
            verbose=False,
        )
        self.logger.info("Model loaded successfully.")

    def __call__(
        self,
        conversation: Conversation,
        tool_choice: str | Dict[str, Any] = "auto",
    ) -> Generator[str | dict, None, None]:
        """
        Generate text from the language model using a streaming approach.

        This version yields tool calls as soon as they are considered complete,
        allowing for earlier execution. A tool call is considered complete when
        the model begins generating text content or the stream ends.
        """

        stream = self.model.create_chat_completion(
            messages=conversation.messages,
            tools=[t.to_dict() for t in conversation.tools.values()],
            tool_choice=tool_choice,
            max_tokens=self.max_tokens,
            stream=True,
        )

        yield from self.stream_processor(stream)

    def _parse(
        self,
        stream: Union[
            CreateChatCompletionResponse, Iterator[CreateChatCompletionStreamResponse]
        ],
    ) -> Generator[str | Dict[str, Any], None, None]:
        beg_marker, end_marker = (" &lt;toolcall&gt;", " &lt;/toolcall&gt;")
        beg_len, end_len = len(beg_marker), len(end_marker)

        buffer = ""
        is_call = False
        for chunk in stream:
            delta = chunk["choices"][0]["delta"]

            if content := delta.get("content"):
                buffer += content
            else:
                continue

            if not is_call:
                is_call = buffer.startswith(beg_marker[: len(buffer)])
                if not is_call:
                    yield buffer
                    buffer = ""

            if is_call and buffer.endswith(end_marker):
                tool_call = buffer[beg_len:-end_len]
                try:
                    tool_call = json.loads(tool_call)
                    yield tool_call
                except (json.JSONDecodeError, KeyError) as e:
                    self.logger.error(
                        f"Failed to parse tool call: {tool_call}. Error: {e}"
                    )

                buffer = ""
                is_call = False
        else:
            yield buffer

    def _parse2(
        self,
        stream: Union[
            CreateChatCompletionResponse, Iterator[CreateChatCompletionStreamResponse]
        ],
    ) -> Generator[str | Dict[str, Any], None, None]:

        tool_calls = defaultdict(lambda: defaultdict(str))

        def resolve_and_yield_tools(buffer):
            """Helper to process the buffer and yield complete tool calls."""
            for idx in sorted(buffer.keys()):
                call_data = buffer[idx]
                try:
                    yield {
                        "id": call_data["id"],
                        "name": call_data["name"],
                        "arguments": json.loads(call_data["arguments"]),
                    }
                except (json.JSONDecodeError, KeyError) as e:
                    self.logger.error(
                        f"Failed to parse tool call: {call_data}. Error: {e}"
                    )
            buffer.clear()

        for chunk in stream:
            delta = chunk["choices"][0]["delta"]

            if content := delta.get("content"):
                if tool_calls:
                    yield from resolve_and_yield_tools(tool_calls)
                yield content
                continue

            if streamed_tool_chunks := delta.get("tool_calls"):
                for tc_chunk in streamed_tool_chunks:
                    idx = tc_chunk["index"]

                    if new_id := tc_chunk.get("id"):
                        tool_calls[idx]["id"] = new_id

                    if func := tc_chunk.get("function"):
                        if name := func.get("name"):
                            tool_calls[idx]["name"] += name
                        if args := func.get("arguments"):
                            tool_calls[idx]["arguments"] += args

        if tool_calls:
            yield from resolve_and_yield_tools(tool_calls)
