from abc import ABC, abstractmethod
import ipaddress
import json
import os
from threading import Lock
from typing import Any, Dict, Generator
from urllib.parse import urlparse

from .conversation import Conversation
from .context import DropOldestStrategy
from .tools import ToolCall
from .decoders import StreamDecoder, NativeDecoder, LegacyXMLDecoder, GemmaE2BDecoder

from ..common.utils import get_logger
from ..exceptions import LLMError, ProviderError


class LLM(ABC):
    decoder: StreamDecoder = NativeDecoder()

    def __init__(self):
        # Initialize ContextManager in the base class or let subclasses handle it.
        pass

    def create_context_strategy(self, max_turns: int = 20):
        """Return a context strategy for trimming conversation history."""
        return DropOldestStrategy(max_turns)

    def __call__(
        self,
        conversation: Conversation,
        session_state: dict | None = None,
        **kwargs,
    ) -> Generator[str | ToolCall, None, None]:
        session_state = session_state or dict()

        try:
            raw_stream = self._infer(
                conversation=conversation,
                session_state=session_state,
                **kwargs,
            )
            yield from self.decoder(raw_stream)
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
        """Return a deterministic offline estimate for remote providers."""
        # Some tiktoken distributions download encoding data on first use.
        # Remote providers only need a conservative context-pruning estimate;
        # direct GGUF providers override this with their exact local tokenizer.
        return max(1, (len(text) + 3) // 4)


class GGUFLLM(LLM):
    local_dir: str = os.path.join("model_files", "llm")
    n_ctx: int = 512
    max_tokens: int = 128
    type_k: str | None = None  # KV cache quantization for K (e.g. "q4_0", "q8_0", "f16")
    type_v: str | None = None  # KV cache quantization for V

    def __init__(self):
        super().__init__()  # Call base init if it exists
        from llama_cpp import Llama

        self.logger = get_logger(self.__class__.__name__)
        from ..common.model_manager import ensure_downloaded

        paths = ensure_downloaded(self.__class__.__name__, local_dir=self.local_dir)
        model_path = paths["model"]

        self.logger.info("Loading model...")
        n_gpu_layers = -1
        flash_attn = True

        # String-to-int mapping for KV cache quantization types
        _GGML_CACHE_TYPES = {"f16": 1, "q4_0": 2, "q8_0": 8}

        for attempt in range(3):
            try:
                kwargs = dict(
                    model_path=model_path,
                    n_ctx=self.n_ctx,
                    n_gpu_layers=n_gpu_layers,
                    flash_attn=flash_attn,
                    verbose=False,
                )
                if self.type_k:
                    kwargs["type_k"] = _GGML_CACHE_TYPES.get(self.type_k, self.type_k)
                if self.type_v:
                    kwargs["type_v"] = _GGML_CACHE_TYPES.get(self.type_v, self.type_v)
                self.model = Llama(**kwargs)
                break
            except Exception as e:
                self.logger.warning(
                    f"Model load attempt {attempt + 1} failed (gpu={n_gpu_layers}, flash_attn={flash_attn}): {e}"
                )
                if attempt == 0 and n_gpu_layers == -1:
                    n_gpu_layers = 0
                    self.logger.info("Retrying with CPU only (n_gpu_layers=0)...")
                elif attempt == 1 and flash_attn:
                    flash_attn = False
                    self.logger.info("Retrying with flash_attn=False...")
                else:
                    raise
        self.logger.info("Model loaded successfully.")

        self._last_state = None
        self._lock = Lock()

    def create_context_strategy(self, max_turns: int = 20):
        return DropOldestStrategy(max_turns)

    @classmethod
    def download(cls):
        from ..common.model_manager import ensure_downloaded
        ensure_downloaded(cls.__name__, local_dir=cls.local_dir)

    def count_tokens(self, text: str) -> int:
        return len(self.model.tokenize(text.encode("utf-8")))

    def _infer(
        self,
        conversation: Conversation,
        *,
        session_state: dict,
        tool_choice: str | Dict[str, Any] = "auto",
        **kwargs,
    ) -> Generator[str | ToolCall, None, None]:
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
                delta = chunk["choices"][0]["delta"]

                # Handle standard text content
                if "content" in delta and delta["content"]:
                    yield delta["content"]

                # Handle native tool calls from llama.cpp
                elif "tool_calls" in delta:
                    for tool_call in delta["tool_calls"]:
                        if tool_call.get("type") == "function":
                            func_info = tool_call["function"]
                            try:
                                # Parse the JSON string arguments into a dictionary
                                args_dict = json.loads(func_info.get("arguments", "{}"))
                            except json.JSONDecodeError:
                                args_dict = {}

                            # Yield the ToolCall format expected by Session._generate_response
                            yield ToolCall(
                                name=func_info["name"],
                                arguments=args_dict
                            )
                            return # Halt stream!


class LiteLLMProvider(LLM):
    """OpenAI-format adapter for provider-backed models supported by LiteLLM."""

    supported_providers = frozenset({"openai", "gemini", "ollama"})
    _provider_key_env = {
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
    }
    _allowed_generation_params = frozenset(
        {
            "frequency_penalty",
            "presence_penalty",
            "seed",
            "stop",
            "temperature",
            "top_p",
        }
    )

    def __init__(
        self,
        model: str,
        *,
        provider: str,
        api_key: str | None = None,
        api_base: str | None = None,
        timeout: float = 60.0,
        n_ctx: int = 16_384,
        max_tokens: int = 1_024,
        completion_fn=None,
    ):
        super().__init__()
        self.logger = get_logger(self.__class__.__name__)

        if provider not in self.supported_providers:
            raise ProviderError(
                f"Unsupported LiteLLM provider: {provider!r}."
            )
        if not isinstance(model, str) or not 1 <= len(model.strip()) <= 256:
            raise ProviderError("Model must be a non-empty string up to 256 characters.")
        if "/" in model and not model.startswith(f"{provider}/"):
            raise ProviderError("The model prefix must match its configured provider.")
        if not isinstance(timeout, (int, float)) or not 1 <= timeout <= 300:
            raise ProviderError("Provider timeout must be between 1 and 300 seconds.")
        if not isinstance(n_ctx, int) or not 512 <= n_ctx <= 2_000_000:
            raise ProviderError("n_ctx must be between 512 and 2000000.")
        if not isinstance(max_tokens, int) or not 1 <= max_tokens < n_ctx:
            raise ProviderError("max_tokens must be positive and smaller than n_ctx.")
        if api_base is not None:
            self._validate_api_base(api_base)

        key_env = self._provider_key_env.get(provider)
        api_key = api_key or (os.environ.get(key_env) if key_env else None)
        if key_env and not api_key:
            # Local servers (llama.cpp, Ollama) don't need API keys
            if api_base and self._is_loopback(api_base):
                api_key = "sk-no-key-required"
            else:
                raise ProviderError(f"{key_env} is required for the {provider} provider.")
        if api_key is not None and not isinstance(api_key, str):
            raise ProviderError("Provider API keys must be strings.")
        if isinstance(api_key, str) and not api_key.strip():
            raise ProviderError("Provider API keys must not be blank.")

        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.timeout = float(timeout)
        self.n_ctx = n_ctx
        self.max_tokens = max_tokens
        self._openai_client = None

        # For local OpenAI-compatible servers (llama.cpp, Ollama), use the
        # OpenAI SDK directly — LiteLLM's CustomStreamWrapper has issues with
        # synchronous iteration on Windows.
        if provider == "openai" and api_base and self._is_loopback(api_base):
            from openai import OpenAI
            self._openai_client = OpenAI(
                base_url=api_base.rstrip("/") + "/v1" if "/v1" not in api_base else api_base,
                api_key=api_key or "sk-no-key-required",
            )
        else:
            if completion_fn is None:
                os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
                from litellm import completion
                completion_fn = completion
            if not callable(completion_fn):
                raise ProviderError("completion_fn must be callable.")
            self._completion = completion_fn

    @staticmethod
    def _is_loopback(api_base: str) -> bool:
        parsed = urlparse(api_base)
        hostname = parsed.hostname.lower() if parsed.hostname else ""
        if hostname == "localhost":
            return True
        try:
            return ipaddress.ip_address(hostname).is_loopback
        except ValueError:
            return False

    @staticmethod
    def _validate_api_base(api_base: str) -> None:
        if not isinstance(api_base, str) or len(api_base) > 2_048:
            raise ProviderError("api_base must be a URL up to 2048 characters.")
        parsed = urlparse(api_base)
        if not parsed.hostname or parsed.username or parsed.password:
            raise ProviderError("api_base must be an absolute URL without credentials.")
        if parsed.scheme == "https":
            return
        if parsed.scheme != "http":
            raise ProviderError("api_base must use HTTPS, or HTTP for loopback only.")

        # ASVS 12.3.1 / 12.3.2: custom remote provider endpoints must use
        # certificate-validated HTTPS; plaintext is limited to local runtimes.
        if not LiteLLMProvider._is_loopback(api_base):
            raise ProviderError("Plaintext provider endpoints are allowed on loopback only.")

    @staticmethod
    def _field(value, name: str, default=None):
        if isinstance(value, dict):
            return value.get(name, default)
        return getattr(value, name, default)

    def _infer(
        self,
        conversation: Conversation,
        *,
        session_state: dict,
        tool_choice: str | Dict[str, Any] = "auto",
        **kwargs,
    ) -> Generator[str | ToolCall, None, None]:
        if self._openai_client:
            yield from self._infer_openai(conversation, tool_choice=tool_choice)
            return

        unexpected = set(kwargs) - self._allowed_generation_params
        if unexpected:
            raise LLMError(
                "Unsupported generation parameters: "
                + ", ".join(sorted(unexpected))
            )
        if not isinstance(tool_choice, (str, dict)):
            raise LLMError("tool_choice must be a string or object.")
        if isinstance(tool_choice, str) and tool_choice not in {
            "auto",
            "none",
            "required",
        }:
            raise LLMError("Unsupported tool_choice value.")

        request = {
            "model": self.model,
            "messages": conversation.messages,
            "stream": True,
            "timeout": self.timeout,
            "num_retries": 0,
            "max_tokens": self.max_tokens,
            **kwargs,
        }
        if self.api_key:
            request["api_key"] = self.api_key
        if self.api_base:
            request["api_base"] = self.api_base

        tools = [tool.to_dict() for tool in conversation.tools.values()]
        if tools and tool_choice != "none":
            request["tools"] = tools
            request["tool_choice"] = tool_choice

        stream = self._completion(**request)
        tool_call_buffer = {}
        for chunk in stream:
            choices = self._field(chunk, "choices", ()) or ()
            if not choices:
                continue
            delta = self._field(choices[0], "delta")
            if delta is None:
                continue

            content = self._field(delta, "content")
            if isinstance(content, str) and content:
                yield content

            for tool_call in self._field(delta, "tool_calls", ()) or ():
                index = self._field(tool_call, "index", 0)
                if not isinstance(index, int) or not 0 <= index <= 128:
                    raise LLMError("Provider returned an invalid tool call index.")
                function = self._field(tool_call, "function", {}) or {}
                entry = tool_call_buffer.setdefault(
                    index, {"name": "", "arguments": "", "object_args": None}
                )
                name = self._field(function, "name")
                arguments = self._field(function, "arguments")
                if isinstance(name, str):
                    entry["name"] += name
                    if len(entry["name"]) > 64:
                        raise LLMError("Tool name exceeds 64 characters.")
                if isinstance(arguments, dict):
                    entry["object_args"] = arguments
                elif isinstance(arguments, str):
                    entry["arguments"] += arguments
                    # ASVS 2.2.1: bound model-generated serialized tool input.
                    if len(entry["arguments"]) > 65_536:
                        raise LLMError("Tool arguments exceed 65536 characters.")

        for entry in tool_call_buffer.values():
            name = entry["name"]
            if not name:
                continue
            try:
                arguments = entry["object_args"]
                if arguments is None:
                    arguments = (
                        json.loads(entry["arguments"])
                        if entry["arguments"]
                        else {}
                    )
                # ASVS 1.5.2 / 15.3.5: accept only a JSON object as tool input.
                if not isinstance(arguments, dict):
                    raise LLMError("Tool arguments must be an object.")
                yield ToolCall(name=name, arguments=arguments)
            except (json.JSONDecodeError, ValueError):
                self.logger.warning(
                    "Provider returned malformed arguments for tool %s.", name
                )
                yield ToolCall(
                    name="_parse_error",
                    arguments={"tool_name": name},
                )

    def _infer_openai(
        self,
        conversation: Conversation,
        *,
        tool_choice: str | Dict[str, Any] = "auto",
    ) -> Generator[str | ToolCall, None, None]:
        kwargs = {
            "model": self.model,
            "messages": conversation.messages,
            "stream": True,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
        }
        tools = [tool.to_dict() for tool in conversation.tools.values()]
        if tools and tool_choice != "none":
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice

        stream = self._openai_client.chat.completions.create(**kwargs)
        tool_call_buffer = {}
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta is None:
                continue

            if delta.content:
                yield delta.content

            for tc in (getattr(delta, "tool_calls", None) or ()):
                index = getattr(tc, "index", 0) if hasattr(tc, "index") else 0
                function = getattr(tc, "function", {}) or {}
                entry = tool_call_buffer.setdefault(
                    index, {"name": "", "arguments": "", "object_args": None}
                )
                name = getattr(function, "name", None)
                arguments = getattr(function, "arguments", None)
                if isinstance(name, str):
                    entry["name"] += name
                    if len(entry["name"]) > 64:
                        raise LLMError("Tool name exceeds 64 characters.")
                if isinstance(arguments, dict):
                    entry["object_args"] = arguments
                elif isinstance(arguments, str):
                    entry["arguments"] += arguments
                    if len(entry["arguments"]) > 65_536:
                        raise LLMError("Tool arguments exceed 65536 characters.")

        for entry in tool_call_buffer.values():
            name = entry["name"]
            if not name:
                continue
            try:
                arguments = entry["object_args"] or (
                    json.loads(entry["arguments"]) if entry["arguments"] else {}
                )
                if not isinstance(arguments, dict):
                    raise LLMError("Tool arguments must be an object.")
                yield ToolCall(name=name, arguments=arguments)
            except (json.JSONDecodeError, ValueError):
                self.logger.warning(
                    "Provider returned malformed arguments for tool %s.", name
                )
                yield ToolCall(
                    name="_parse_error",
                    arguments={"tool_name": name},
                )


class Qwen3(GGUFLLM):
    n_ctx: int = 40960
    max_tokens: int = 8192
    decoder = LegacyXMLDecoder()


class Gemma4E2B(GGUFLLM):
    n_ctx: int = 131072
    max_tokens: int = 8192
    decoder = GemmaE2BDecoder()
    type_k: str = "q4_0"
    type_v: str = "q4_0"


class Gemma4_12B(GGUFLLM):
    n_ctx: int = 8192
    max_tokens: int = 2048
    type_k: str = "q4_0"
    type_v: str = "q4_0"
    decoder = LegacyXMLDecoder()


# ----------------------------------------------------------------------


class LLMProviders:
    _providers = {
        "Qwen3": Qwen3,
        "Gemma4E2B": Gemma4E2B,
        "Gemma4_12B": Gemma4_12B,
        "LiteLLM": LiteLLMProvider,
    }

    _settings_aliases = {
        "LiteLLM": "litellm",
    }

    @classmethod
    def get(cls, name: str):
        if not isinstance(name, str) or not name:
            raise ProviderError("LLM provider name must be a non-empty string.")
        provider = cls._providers.get(name)
        if provider is None:
            raise ProviderError(f"Unknown LLM provider: {name!r}.")
        return provider

    @classmethod
    def create(cls, name: str, settings: dict | None = None):
        provider = cls.get(name)
        settings = settings or {}
        if not isinstance(settings, dict):
            raise ProviderError("LLM provider settings must be an object.")
        settings_key = cls._settings_aliases.get(name, name.lower())
        provider_settings = settings.get(settings_key, {})
        if not isinstance(provider_settings, dict):
            raise ProviderError("Selected LLM provider settings must be an object.")
        # ASVS 2.2.1 / 15.3.3: only the selected provider's documented
        # configuration object is passed into its constructor.
        return provider(**provider_settings)


# ----------------------------------------------------------------------
