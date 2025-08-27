import json
import os
from typing import Any, Dict, Generator, Union

import ollama
from llama_cpp import (
    CreateChatCompletionResponse,
    CreateChatCompletionStreamResponse,
    Iterator,
    Llama,
)

from .conversation import Conversation

from ..common.config import config
from ..common.utils import download_hf_file, get_logger


class NemotronLLM:
    hf_repo: str = "bartowski/Nemotron-Mini-4B-Instruct-GGUF"
    filename: str = "Nemotron-Mini-4B-Instruct-Q5_K_M.gguf"
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
            n_ctx=4096,
            n_gpu_layers=-1,
            flash_attn=True,
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
        tool_beg, tool_end = (" $lt;toolcall$gt;", " $lt;/toolcall$gt;")
        tool_beg_l, tool_end_l = len(tool_beg), len(tool_end)
        think_beg, think_end = (" $lt;think$gt;", " $lt;/think$gt;")

        buffer = ""
        is_call = False
        is_thought = False
        for chunk in stream:
            delta = chunk["choices"][0]["delta"]

            if content := delta.get("content"):
                buffer += content
                b_ = buffer.strip()
            else:
                continue

            if not is_thought:
                is_thought = b_.startswith(think_beg[: len(b_)])

            if is_thought:
                if b_.endswith(think_end):
                    buffer = ""
                    is_thought = False
                continue

            if not is_call:
                is_call = b_.startswith(tool_beg[: len(b_)])
                if not is_call:
                    yield buffer
                    buffer = ""

            if is_call and b_.endswith(tool_end):
                tool_call = b_[tool_beg_l:-tool_end_l]
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


class QwenLLM:
    hf_repo: str = "Qwen/Qwen3-4B-GGUF"
    filename: str = "Qwen3-4B-Q5_K_M.gguf"
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
            n_ctx=8192,
            n_gpu_layers=-1,
            flash_attn=True,
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
        tool_beg, tool_end = (
            "<tool_call>",
            "</tool_call>",
        )
        tool_beg_l, tool_end_l = len(tool_beg), len(tool_end)
        think_beg, think_end = ("<think>", "</think>")

        buffer = ""
        is_call = False
        is_thought = False
        for chunk in stream:
            delta = chunk["choices"][0]["delta"]

            if content := delta.get("content"):
                buffer += content
                b_ = buffer.strip()
                if not b_:
                    continue
            else:
                continue

            if not is_thought:
                is_thought = b_.startswith(think_beg[: len(b_)])

            if is_thought:
                if b_.endswith(think_end):
                    buffer = ""
                    is_thought = False
                continue

            if not is_call:
                is_call = b_.startswith(tool_beg[: len(b_)])
                if not is_call:
                    yield buffer
                    buffer = ""

            if is_call and b_.endswith(tool_end):
                tool_call = b_[tool_beg_l:-tool_end_l]
                try:
                    tool_call = json.loads(tool_call)
                    if "name" in tool_call:
                        tool_call["function"] = tool_call["name"]
                        del tool_call["name"]
                    yield tool_call
                except (json.JSONDecodeError, KeyError) as e:
                    self.logger.error(
                        f"Failed to parse tool call: {tool_call}. Error: {e}"
                    )

                buffer = ""
                is_call = False
        else:
            yield buffer


class OllamaLLM:
    def __init__(self):
        self.logger = get_logger(__name__)
        self.client = ollama.Client(
            host=config.get("llm.providers.ollama.base_url")
        )
        self.model = config.get("llm.models.default")
        self.stream_processor = self._parse

    def __call__(
        self,
        conversation: Conversation,
        tool_choice: str | Dict[str, Any] = "auto",
    ) -> Generator[str | dict, None, None]:
        stream = self.client.chat(
            model=self.model,
            messages=conversation.messages,
            stream=True,
        )
        yield from self.stream_processor(stream)

    def _parse(self, stream) -> Generator[str | Dict[str, Any], None, None]:
        for chunk in stream:
            if "content" in chunk["message"]:
                yield chunk["message"]["content"]


# ----------------------------------------------------------------------

llm_providers = {
    "nemotron": NemotronLLM,
    "qwen": QwenLLM,
    "ollama": OllamaLLM,
}

provider = config.get("llm.default_provider", "nemotron")
LLM = llm_providers.get(provider)

if not LLM:
    raise ValueError(f"Invalid LLM provider specified in config: {provider}")

# ----------------------------------------------------------------------
