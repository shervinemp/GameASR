from abc import ABC, abstractmethod
import json
import os
from typing import Any, Dict, Generator, Iterator

from .conversation import Conversation

from ..common.config import config
from ..common.utils import download_hf_file, get_logger


class LLM(ABC):
    def __call__(
        self, conversation: Conversation, *args, **kwargs
    ) -> Generator[str | dict, None, None]:
        yield from self._parse(self._infer(conversation, *args, **kwargs))

    @abstractmethod
    def _infer(
        self, conversation: Conversation, *args, **kwargs
    ) -> Generator[str, None, None]: ...

    def _parse(
        self,
        stream: Iterator[str],
    ) -> Generator[str | Dict[str, Any], None, None]:
        tag_pairs = [
            ("<", ">"),
            ("&lt;", "&gt;"),
        ]
        tool_beg, tool_end = ("toolcall", "/toolcall")
        think_beg, think_end = ("think", "/think")

        buffer = ""
        tag_body = ""
        bound_pair = None
        is_tag = False
        is_call = False
        is_thought = False

        for content in stream:
            buffer += content
            b_ = buffer.strip()

            bound_pair = next(
                filter(
                    lambda p: b_ and b_.startswith(p[0][: len(b_)]), tag_pairs
                ),
                None,
            )

            is_tag = is_tag or bound_pair  # TODO: FIX
            if is_tag:
                if b_.endswith(bound_pair[1]):
                    inner = b_[len(bound_pair[0]) : -len(bound_pair[1])]
                    buffer = ""
                    is_tag = False
                    if inner == think_beg:
                        is_thought = True
                    elif inner == tool_beg:
                        is_call = True
                    elif inner == think_end:
                        is_thought = False
                    elif inner == tool_end:
                        try:
                            tool_call = json.loads(tag_body)
                            tag_body = ""
                            yield tool_call
                        except (json.JSONDecodeError, KeyError) as e:
                            self.logger.error(
                                f"Failed to parse tool call: {tag_body}. Error: {e}"
                            )

                        buffer = ""
                        is_call = False
            else:
                if is_call or is_thought:
                    tag_body += buffer
                    buffer = ""
                    continue
                yield buffer
                buffer = ""
        if b_:
            yield buffer


class GGUFLLM(LLM):
    hf_repo: str = ""
    filename: str = ""
    local_dir: str = os.path.join("model_files", "llm")
    n_ctx: int = 8192
    max_tokens: int = 8192

    def __init__(self):
        from llama_cpp import Llama

        self.logger = get_logger(self.__class__.__name__)
        model_path = os.path.join(self.local_dir, self.filename)
        self.stream_processor = self._parse

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Please run the download script first."
            )

        self.logger.info("Loading model...")
        self.model = Llama(
            model_path=model_path,
            n_ctx=self.n_ctx,
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

    def _infer(
        self,
        conversation: Conversation,
        tool_choice: str | Dict[str, Any] = "auto",
    ) -> Generator[str, None, None]:

        stream = self.model.create_chat_completion(
            messages=conversation.messages,
            tools=[t.to_dict() for t in conversation.tools.values()],
            tool_choice=tool_choice,
            max_tokens=self.max_tokens,
            stream=True,
        )

        for chunk in stream:
            if (k := "content") in (delta := chunk["choices"][0]["delta"]):
                yield delta[k]


class NemotronMini(GGUFLLM):
    hf_repo: str = "bartowski/Nemotron-Mini-4B-Instruct-GGUF"
    filename: str = "Nemotron-Mini-4B-Instruct-Q5_K_M.gguf"


class Qwen3(GGUFLLM):
    hf_repo: str = "Qwen/Qwen3-4B-GGUF"
    filename: str = "Qwen3-4B-Q5_K_M.gguf"


class OllamaModel(LLM):
    def __init__(
        self,
        host: str = config.get("llm.providers.ollama.base_url"),
        model: str = config.get("llm.models.default"),
    ):
        from ollama import Client

        self.logger = get_logger(__name__)
        self.client = Client(host=host)
        self.model = model

    def _infer(
        self, conversation: Conversation
    ) -> Generator[str | Dict[str, Any], None, None]:

        stream = self.client.chat(
            model=self.model,
            messages=conversation.messages,
            stream=True,
        )

        for chunk in stream:
            if "content" in chunk["message"]:
                yield chunk["message"]["content"]


# ----------------------------------------------------------------------

llm_providers = {
    "nemotron": NemotronMini,
    "qwen3": Qwen3,
    "ollama": OllamaModel,
}

provider = config.get("llm.default_provider", "nemotron")
llm_class = llm_providers.get(provider)

if not llm_class:
    raise ValueError(f"Invalid LLM provider specified in config: {provider}")

LLM = llm_class

# ----------------------------------------------------------------------
