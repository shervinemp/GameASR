from abc import ABC, abstractmethod
import ipaddress
import json
import os
from threading import Lock
from typing import Any, Dict, Generator
from urllib.parse import urlparse

from .conversation import Conversation
from .context import ContextManager  # Added import
from .tools import ToolCall
from .decoders import StreamDecoder, NativeDecoder, LegacyXMLDecoder, GemmaE2BDecoder

from ..common.utils import download_hf_file, get_logger, verify_file_sha256


class LLM(ABC):
    decoder: StreamDecoder = NativeDecoder()

    def __init__(self):
        # Initialize ContextManager in the base class or let subclasses handle it.
        pass

    def __call__(
        self,
        conversation: Conversation,
        session_state: dict | None = None,
        **kwargs,
    ) -> Generator[str | ToolCall, None, None]:
        session_state = session_state or dict()

        # Manage context before inference
        context_manager = ContextManager()
        context_manager.manage_context(conversation, self)

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
    hf_repo: str = ""
    filename: str = ""
    revision: str = ""
    sha256: str = ""
    local_dir: str = os.path.join("model_files", "llm")
    n_ctx: int = 512
    max_tokens: int = 128
    type_k: str | None = None  # KV cache quantization for K (e.g. "q4_0", "q8_0", "f16")
    type_v: str | None = None  # KV cache quantization for V

    def __init__(self):
        super().__init__()  # Call base init if it exists
        from llama_cpp import Llama

        self.logger = get_logger(self.__class__.__name__)
        model_path = os.path.join(self.local_dir, self.filename)

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Please run the download script first."
            )

        # ASVS 15.2.4: verify model integrity before native code parses it.
        verify_file_sha256(model_path, self.sha256)

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

    @classmethod
    def download(cls):
        os.makedirs(cls.local_dir, exist_ok=True)
        download_hf_file(
            repo_id=cls.hf_repo,
            filename=cls.filename,
            directory=cls.local_dir,
            revision=cls.revision,
            expected_sha256=cls.sha256,
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
            raise ValueError(
                f"Unsupported LiteLLM provider: {provider!r}."
            )
        if not isinstance(model, str) or not 1 <= len(model.strip()) <= 256:
            raise ValueError("Model must be a non-empty string up to 256 characters.")
        if "/" in model and not model.startswith(f"{provider}/"):
            raise ValueError("The model prefix must match its configured provider.")
        if not isinstance(timeout, (int, float)) or not 1 <= timeout <= 300:
            raise ValueError("Provider timeout must be between 1 and 300 seconds.")
        if not isinstance(n_ctx, int) or not 512 <= n_ctx <= 2_000_000:
            raise ValueError("n_ctx must be between 512 and 2000000.")
        if not isinstance(max_tokens, int) or not 1 <= max_tokens < n_ctx:
            raise ValueError("max_tokens must be positive and smaller than n_ctx.")
        if api_base is not None:
            self._validate_api_base(api_base)

        key_env = self._provider_key_env.get(provider)
        api_key = api_key or (os.environ.get(key_env) if key_env else None)
        if key_env and not api_key:
            raise ValueError(f"{key_env} is required for the {provider} provider.")
        if api_key is not None and not isinstance(api_key, str):
            raise TypeError("Provider API keys must be strings.")
        if isinstance(api_key, str) and not api_key.strip():
            raise ValueError("Provider API keys must not be blank.")

        if completion_fn is None:
            # Keep local/Ollama startup offline by default. LiteLLM otherwise
            # refreshes its model-cost map from GitHub during import. An
            # explicit environment value still lets operators opt back in.
            os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
            from litellm import completion

            completion_fn = completion
        if not callable(completion_fn):
            raise TypeError("completion_fn must be callable.")

        self.provider = provider
        self.model = model if "/" in model else f"{provider}/{model}"
        self.api_key = api_key
        self.api_base = api_base
        self.timeout = float(timeout)
        self.n_ctx = n_ctx
        self.max_tokens = max_tokens
        self._completion = completion_fn

    @staticmethod
    def _validate_api_base(api_base: str) -> None:
        if not isinstance(api_base, str) or len(api_base) > 2_048:
            raise ValueError("api_base must be a URL up to 2048 characters.")
        parsed = urlparse(api_base)
        if not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("api_base must be an absolute URL without credentials.")
        if parsed.scheme == "https":
            return
        if parsed.scheme != "http":
            raise ValueError("api_base must use HTTPS, or HTTP for loopback only.")

        hostname = parsed.hostname.lower()
        is_loopback = hostname == "localhost"
        if not is_loopback:
            try:
                is_loopback = ipaddress.ip_address(hostname).is_loopback
            except ValueError:
                is_loopback = False
        # ASVS 12.3.1 / 12.3.2: custom remote provider endpoints must use
        # certificate-validated HTTPS; plaintext is limited to local runtimes.
        if not is_loopback:
            raise ValueError("Plaintext provider endpoints are allowed on loopback only.")

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
        unexpected = set(kwargs) - self._allowed_generation_params
        if unexpected:
            raise TypeError(
                "Unsupported generation parameters: "
                + ", ".join(sorted(unexpected))
            )
        if not isinstance(tool_choice, (str, dict)):
            raise TypeError("tool_choice must be a string or object.")
        if isinstance(tool_choice, str) and tool_choice not in {
            "auto",
            "none",
            "required",
        }:
            raise ValueError("Unsupported tool_choice value.")

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
                    raise ValueError("Provider returned an invalid tool call index.")
                function = self._field(tool_call, "function", {}) or {}
                entry = tool_call_buffer.setdefault(
                    index, {"name": "", "arguments": "", "object_args": None}
                )
                name = self._field(function, "name")
                arguments = self._field(function, "arguments")
                if isinstance(name, str):
                    entry["name"] += name
                    if len(entry["name"]) > 64:
                        raise ValueError("Tool name exceeds 64 characters.")
                if isinstance(arguments, dict):
                    entry["object_args"] = arguments
                elif isinstance(arguments, str):
                    entry["arguments"] += arguments
                    # ASVS 2.2.1: bound model-generated serialized tool input.
                    if len(entry["arguments"]) > 65_536:
                        raise ValueError("Tool arguments exceed 65536 characters.")

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
                    raise ValueError("Tool arguments must be an object.")
                yield ToolCall(name=name, arguments=arguments)
            except (json.JSONDecodeError, ValueError):
                self.logger.warning(
                    "Provider returned malformed arguments for tool %s.", name
                )
                yield ToolCall(
                    name="_parse_error",
                    arguments={"tool_name": name},
                )


class Ollama(LiteLLMProvider):
    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        **kwargs,
    ):
        super().__init__(
            model=model,
            provider="ollama",
            api_base=host,
            **kwargs,
        )


class NemotronMini(GGUFLLM):
    hf_repo: str = "bartowski/Nemotron-Mini-4B-Instruct-GGUF"
    filename: str = "Nemotron-Mini-4B-Instruct-Q5_K_M.gguf"
    revision: str = "63e28adee7c1228bdcdda1c6f10f8b5e4d17c42c"
    sha256: str = "6230395444f14a10e4675e4d625183ae3c094567ddfe757991c0909f0f21c2d8"
    n_ctx: int = 4096
    max_tokens: int = 1024
    decoder = LegacyXMLDecoder()


class Qwen3(GGUFLLM):
    hf_repo: str = "Qwen/Qwen3-4B-GGUF"
    filename: str = "Qwen3-4B-Q5_K_M.gguf"
    revision: str = "a9a60d009fa7ff9606305047c2bf77ac25dbec49"
    sha256: str = "aca596860e8cb40af6539e3f2ea40df305f42515deac56d49c08d39a02e6533f"
    n_ctx: int = 40960
    max_tokens: int = 8192
    decoder = LegacyXMLDecoder()


class Gemma4E2B(GGUFLLM):
    hf_repo: str = "unsloth/gemma-4-E2B-it-GGUF"
    filename: str = "gemma-4-E2B-it-UD-Q4_K_XL.gguf"
    revision: str = "90f9618340396838ee7ff5b0ba2da27da62953d3"
    sha256: str = "b8906b8c5e05e57b657646bbc657bd35814a269b2c20f0a2579047fafa1a67dd"
    n_ctx: int = 131072
    max_tokens: int = 8192
    decoder = GemmaE2BDecoder()
    type_k: str = "q4_0"
    type_v: str = "q4_0"


class Gemma4_12B(GGUFLLM):
    hf_repo: str = "unsloth/gemma-4-12b-it-GGUF"
    filename: str = "gemma-4-12b-it-UD-IQ3_XXS.gguf"
    revision: str = "3249fa54d5efa384afc552cc6700ad091efd5c39"
    sha256: str = "1eb42ae04731500a614acaf658c707c07af5a320822eabc50040852442fa6a4c"
    n_ctx: int = 8192
    max_tokens: int = 2048
    type_k: str = "q4_0"
    type_v: str = "q4_0"
    decoder = LegacyXMLDecoder()


class ChatGPT(LiteLLMProvider):
    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        api_base: str | None = None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            provider="openai",
            api_key=api_key,
            api_base=api_base,
            **kwargs,
        )


class Gemini(LiteLLMProvider):
    def __init__(
        self,
        model: str = "gemini-1.5-flash",
        api_key: str | None = None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            provider="gemini",
            api_key=api_key,
            **kwargs,
        )


# ----------------------------------------------------------------------


class LLMProviders:
    NemotronMini: type = NemotronMini
    Qwen3: type = Qwen3
    Ollama: type = Ollama
    ChatGPT: type = ChatGPT
    Gemini: type = Gemini
    Gemma4E2B: type = Gemma4E2B
    Gemma4_12B: type = Gemma4_12B
    LiteLLM: type = LiteLLMProvider

    # Configuration-friendly aliases retained for documented lower-case names.
    openai: type = ChatGPT
    gemini: type = Gemini
    ollama: type = Ollama

    _providers = {
        "NemotronMini": NemotronMini,
        "Qwen3": Qwen3,
        "Ollama": Ollama,
        "ChatGPT": ChatGPT,
        "Gemini": Gemini,
        "Gemma4E2B": Gemma4E2B,
        "Gemma4_12B": Gemma4_12B,
        "LiteLLM": LiteLLMProvider,
        "openai": ChatGPT,
        "gemini": Gemini,
        "ollama": Ollama,
    }

    _settings_aliases = {
        "ChatGPT": "openai",
        "Gemini": "gemini",
        "LiteLLM": "litellm",
        "Ollama": "ollama",
    }

    @classmethod
    def get(cls, name: str):
        if not isinstance(name, str) or not name:
            raise ValueError("LLM provider name must be a non-empty string.")
        provider = cls._providers.get(name)
        if provider is None:
            raise ValueError(f"Unknown LLM provider: {name!r}.")
        return provider

    @classmethod
    def create(cls, name: str, settings: dict | None = None):
        provider = cls.get(name)
        settings = settings or {}
        if not isinstance(settings, dict):
            raise TypeError("LLM provider settings must be an object.")
        settings_key = cls._settings_aliases.get(name, name.lower())
        provider_settings = settings.get(settings_key, {})
        if not isinstance(provider_settings, dict):
            raise TypeError("Selected LLM provider settings must be an object.")
        # ASVS 2.2.1 / 15.3.3: only the selected provider's documented
        # configuration object is passed into its constructor.
        return provider(**provider_settings)


# ----------------------------------------------------------------------
