from abc import ABC, abstractmethod
from itertools import chain
import json
import os
from threading import Lock
from typing import Any, Dict, Generator, Iterator

from .conversation import Conversation
from .context import ContextManager  # Added import

from ..common.utils import download_hf_file, get_logger


class LLM(ABC):
    def __init__(self):
        # Initialize ContextManager in the base class or let subclasses handle it.
        pass

    def __call__(
        self,
        conversation: Conversation,
        session_state: dict | None = None,
        **kwargs,
    ) -> Generator[str | dict, None, None]:
        session_state = session_state or dict()

        # Manage context before inference
        context_manager = ContextManager()
        context_manager.manage_context(conversation, self)

        try:
            yield from self._parse(
                self._infer(
                    conversation=conversation,
                    session_state=session_state,
                    **kwargs,
                )
            )
        except Exception as e:
            # Check for specific timeouts or generic errors
            self.logger.error(
                f"Error during LLM inference: {e}", exc_info=True
            )
            yield "Sorry, I encountered an error or timed out while processing your request."

    @abstractmethod
    def _infer(
        self, conversation: Conversation, *, session_state: dict, **kwargs
    ) -> Generator[str, None, None]: ...

    def count_tokens(self, text: str) -> int:
        """Counts the number of tokens in a string.
        Default implementation is an approximation (4 characters per token).
        """
        return len(text) // 4

    def _parse(
        self,
        stream: Iterator[str],
        flush: bool = False,
    ) -> Generator[str | Dict[str, Any], None, None]:
        # Updated to support Gemma 4 E2B channel tags alongside standard think tags
        tag_boundaries = [
            ("<", ">"),
            ("&lt;", "&gt;"),
            ("<|", "|>"),  # Added for Gemma 4
        ]
        tool_tags = ("toolcall", "tool_call")
        think_tags = ("think", "channel>thought\n", "channel")

        buffer = ""
        tag_body = ""
        bounds = None
        is_call = False
        is_thought = False

        for content in chain.from_iterable(stream):
            buffer += content
            if content.isspace():
                continue

            b_ = buffer.strip()

            bounds = bounds or next(
                filter(
                    lambda p: b_ and b_.startswith(p[0][: len(b_)]),
                    tag_boundaries,
                ),
                None,
            )

            if bounds:
                if b_.endswith(bounds[1]):
                    tag = b_[len(bounds[0]) : -len(bounds[1])]
                    # Normalize tag for Gemma 4 e.g. <channel|> -> channel
                    if tag.startswith("/"):
                        tag = tag[1:]
                        if tag.endswith("|"):
                            tag = tag[:-1]

                        if is_thought and any(t in tag for t in think_tags):
                            is_thought = False
                        elif is_call and any(t in tag for t in tool_tags):
                            is_call = False
                            tb_ = tag_body.strip()
                            try:
                                yield json.loads(tb_)
                            except (json.JSONDecodeError, KeyError) as e:
                                self.logger.error(
                                    f"Failed to parse tool call: {tb_}. Error: {e}"
                                )
                        tag_body = ""
                    else:
                        if tag.endswith("|"):
                            tag = tag[:-1]
                        if any(t in tag for t in think_tags):
                            is_thought = True
                        elif any(t in tag for t in tool_tags):
                            is_call = True
                    buffer = ""
                    bounds = None
                continue

            if buffer and not bounds:
                # Catch Gemma's specific start token: <|channel>thought\n
                if "<|channel>thought" in buffer:
                    is_thought = True
                    buffer = buffer.replace("<|channel>thought", "").strip()

                if is_call or is_thought:
                    tag_body += buffer
                else:
                    yield buffer
                buffer = ""

        if buffer and flush and not is_thought:
            yield buffer


class GGUFLLM(LLM):
    hf_repo: str = ""
    filename: str = ""
    local_dir: str = os.path.join("model_files", "llm")
    n_ctx: int = 512
    max_tokens: int = 128

    def __init__(self):
        super().__init__()  # Call base init if it exists
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

        self._last_state = None
        self._lock = Lock()

    @classmethod
    def download(cls):
        os.makedirs(cls.local_dir, exist_ok=True)
        download_hf_file(
            repo_id=cls.hf_repo,
            filename=cls.filename,
            directory=cls.local_dir,
        )

    def count_tokens(self, text: str) -> int:
        return len(self.model.tokenize(text.encode("utf-8")))

    def _infer(
        self,
        conversation: Conversation,
        *,
        session_state: dict,
        tool_choice: str | Dict[str, Any] = "auto",
        **kwargs,
    ) -> Generator[str, None, None]:
        with self._lock:
            if self._last_state and id(self._last_state) != id(session_state):
                k_ = "model_state"
                self._last_state[k_] = self.model.save_state()
                self.model.reset()
                if s_ := session_state.get(k_):
                    self.model.load_state(s_)

            self._last_state = session_state

            stream = self.model.create_chat_completion(
                messages=conversation.messages,
                tools=[t.to_dict() for t in conversation.tools.values()],
                tool_choice=tool_choice,
                max_tokens=self.max_tokens,
                stream=True,
                **kwargs,
            )

            for chunk in stream:
                if (k := "content") in (delta := chunk["choices"][0]["delta"]):
                    yield delta[k]


class Ollama(LLM):
    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
    ):
        super().__init__()
        from ollama import Client

        self.logger = get_logger(__name__)

        self.model = model
        self.client = Client(host=host)

    # Using default approximation for count_tokens

    def _infer(
        self, conversation: Conversation, *, session_state: dict
    ) -> Generator[str | Dict[str, Any], None, None]:

        stream = self.client.chat(
            model=self.model,
            messages=conversation.messages,
            stream=True,
            tools=[t.to_dict() for t in conversation.tools.values()],
        )

        for chunk in stream:
            if "content" in chunk["message"]:
                yield chunk["message"]["content"]


class NemotronMini(GGUFLLM):
    hf_repo: str = "bartowski/Nemotron-Mini-4B-Instruct-GGUF"
    filename: str = "Nemotron-Mini-4B-Instruct-Q5_K_M.gguf"
    n_ctx: int = 4096
    max_tokens: int = 1024


class Qwen3(GGUFLLM):
    hf_repo: str = "Qwen/Qwen3-4B-GGUF"
    filename: str = "Qwen3-4B-Q5_K_M.gguf"
    n_ctx: int = 40960
    max_tokens: int = 8192


class Gemma4E2B(GGUFLLM):
    hf_repo: str = "unsloth/gemma-4-E2B-it-GGUF"
    filename: str = "gemma-4-E2B-it-Q4_K_M.gguf"
    n_ctx: int = 131072
    max_tokens: int = 8192


class ChatGPT(LLM):
    def __init__(self, model: str = "gpt-4o", api_key: str = None):
        super().__init__()
        from openai import OpenAI

        self.logger = get_logger(self.__class__.__name__)
        self.model = model

        if not api_key:
            raise ValueError("An API key for OpenAI is required.")

        self.client = OpenAI(api_key=api_key)

    # Using default approximation for count_tokens

    def _infer(
        self,
        conversation: Conversation,
        *,
        session_state: dict,
        **kwargs,
    ) -> Generator[str, None, None]:
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=conversation.messages,
            stream=True,
            **kwargs,
        )
        for chunk in stream:
            if content := chunk.choices[0].delta.content:
                yield content


class Gemini(LLM):
    def __init__(self, model: str = "gemini-1.5-flash", api_key: str = None):
        super().__init__()
        import google.generativeai as genai

        self.logger = get_logger(self.__class__.__name__)
        self.model = model

        if not api_key:
            raise ValueError("An API key for Google Gemini is required.")

        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(self.model)

    # Using default approximation for count_tokens

    def _infer(
        self,
        conversation: Conversation,
        *,
        session_state: dict,
        **kwargs,
    ) -> Generator[str, None, None]:
        response = self.client.generate_content(
            conversation.messages, stream=True, **kwargs
        )
        for chunk in response:
            if content := chunk.text:
                yield content


# ----------------------------------------------------------------------


class LLMProviders:
    NemotronMini: type = NemotronMini
    Qwen3: type = Qwen3
    Ollama: type = Ollama
    ChatGPT: type = ChatGPT
    Gemini: type = Gemini
    Gemma4E2B: type = Gemma4E2B


# ----------------------------------------------------------------------
