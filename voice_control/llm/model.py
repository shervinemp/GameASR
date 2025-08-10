import json
import os
from typing import Any, Dict, Generator, Union

from llama_cpp import (
    CreateChatCompletionResponse,
    CreateChatCompletionStreamResponse,
    Iterator,
    Llama,
)

from .conversation import Conversation

from ..common.utils import download_hf_file, get_logger


class LLM:
    hf_repo: str = "bartowski/Nemotron-Mini-4B-Instruct-GGUF"
    filename: str = "Nemotron-Mini-4B-Instruct-Q5_K_M.gguf"
    # hf_repo: str = "Qwen/Qwen3-4B-GGUF"
    # filename: str = "Qwen3-4B-Q5_K_M.gguf"
    local_dir: str = os.path.join("model_files", "llm")

    def __init__(self):
        self.logger = get_logger(__name__)
        model_path = os.path.join(self.local_dir, self.filename)
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

    @classmethod
    def download(cls):
        os.makedirs(cls.local_dir, exist_ok=True)
        download_hf_file(
            repo_id=cls.hf_repo,
            filename=cls.filename,
            directory=cls.local_dir,
        )

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

        # TODO: Skip `think` tags as well

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
