from abc import ABC, abstractmethod
import json
import os
import re
from threading import Lock
from typing import Any, Dict, Generator, Iterator

from .conversation import Conversation
from .context import ContextManager  # Added import
from .tools import ToolCall

from ..common.utils import download_hf_file, get_logger


class StreamDecoder(ABC):
    """Strategy interface for interpreting and intercepting LLM streams."""
    @abstractmethod
    def __call__(self, stream: Iterator[str | dict | ToolCall]) -> Generator[str | dict | ToolCall, None, None]: ...


class NativeDecoder(StreamDecoder):
    """Yields stream exactly as it arrives (used for ChatGPT, Gemini, and native tools)."""
    def __call__(self, stream: Iterator[str | dict | ToolCall]) -> Generator[str | dict | ToolCall, None, None]:
        yield from stream


class LegacyXMLDecoder(StreamDecoder):
    """Legacy parser for standard <toolcall>...</toolcall> used by Qwen/Nemotron."""
    def __call__(self, stream: Iterator[str | dict | ToolCall]) -> Generator[str | dict | ToolCall, None, None]:
        buffer, in_tool = "", False
        for chunk in stream:
            if isinstance(chunk, (dict, ToolCall)):
                yield chunk; return
            buffer += chunk

            while buffer:
                if in_tool:
                    if "</toolcall>" in buffer:
                        try:
                            # Isolate the tool body
                            tool_body = buffer.split("</toolcall>")[0]
                            tool_dict = json.loads(tool_body.strip())
                            yield ToolCall(
                                name=tool_dict.get("name") or tool_dict.get("function"),
                                arguments=tool_dict.get("arguments", {})
                            )
                        except Exception: pass
                        return # Halt stream to execute tool
                    else:
                        break # Wait for more chunks
                else:
                    if "<toolcall>" in buffer:
                        pre = buffer.split("<toolcall>")[0]
                        if pre: yield pre
                        in_tool = True
                        buffer = buffer.split("<toolcall>", 1)[1]
                        continue # Re-evaluate buffer
                    else:
                        safe_idx = buffer.rfind("<")
                        if safe_idx != -1 and len(buffer) - safe_idx < 15:
                            if safe_idx > 0: yield buffer[:safe_idx]; buffer = buffer[safe_idx:]
                            break # Wait for more chunks
                        else:
                            yield buffer; buffer = ""
                            break

        if buffer and not in_tool: yield buffer


class GemmaE2BDecoder(StreamDecoder):
    """Clean, chunk-based sliding window parser for Gemma 4's unique custom syntax."""
    def __call__(self, stream: Iterator[str | dict | ToolCall]) -> Generator[str | dict | ToolCall, None, None]:
        buffer = ""
        in_tool, in_thought = False, False

        for chunk in stream:
            if isinstance(chunk, (dict, ToolCall)):
                yield chunk
                return

            buffer += chunk

            while buffer:
                # 1. Handle Tool Execution
                if in_tool:
                    if "<tool_call|>" in buffer:
                        body = buffer.split("<tool_call|>")[0].replace("<|tool_call>", "").strip()
                        if body.startswith("call:"):
                            match = re.match(r"call:([a-zA-Z0-9_]+)(.*)", body, re.DOTALL)
                            if match:
                                name, args = match.group(1).strip(), match.group(2).strip()
                                args = args.replace('<|"|>', '"')
                                args = re.sub(r'([{,]\s*)([a-zA-Z0-9_]+)(\s*:)', r'\1"\2"\3', args)
                                try:
                                    yield ToolCall(name=name, arguments=json.loads(args))
                                except json.JSONDecodeError: pass
                        return  # CRITICAL: Halt stream to allow RAG pipeline to trigger
                    break # Wait for more chunks

                # 2. Filter Thoughts
                if in_thought:
                    if "<channel|>" in buffer:
                        in_thought = False
                        buffer = buffer.split("<channel|>", 1)[1]
                        continue # Re-evaluate buffer
                    break # Wait for more chunks

                # 3. Detect Openers
                if "<|tool_call>" in buffer:
                    in_tool = True
                    pre = buffer.split("<|tool_call>")[0]
                    if pre: yield pre
                    buffer = buffer.split("<|tool_call>", 1)[1]
                    continue # Re-evaluate buffer

                if "<|channel>thought" in buffer:
                    in_thought = True
                    pre = buffer.split("<|channel>thought")[0]
                    if pre: yield pre
                    buffer = buffer.split("<|channel>thought", 1)[1]
                    continue # Re-evaluate buffer

                # 4. Safe Yield (Wait for partial tags to resolve)
                safe_idx = max(buffer.rfind("<"), buffer.rfind("&"))
                if safe_idx != -1 and (len(buffer) - safe_idx) < 15:
                    if safe_idx > 0:
                        yield buffer[:safe_idx]
                        buffer = buffer[safe_idx:]
                    break # Wait for more chunks
                else:
                    if buffer: yield buffer
                    buffer = ""
                    break # Finished processing this chunk

        if buffer and not in_tool and not in_thought:
            yield buffer


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
        """Counts the number of tokens in a string.
        Default implementation is an approximation (4 characters per token).
        """
        return len(text) // 4


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
    ) -> Generator[str | ToolCall, None, None]:

        stream = self.client.chat(
            model=self.model,
            messages=conversation.messages,
            stream=True,
            tools=[t.to_dict() for t in conversation.tools.values()],
        )

        for chunk in stream:
            message = chunk.get("message", {})
            if "content" in message and message["content"]:
                yield message["content"]

            if "tool_calls" in message:
                for tool_call in message["tool_calls"]:
                    func_info = tool_call.get("function", {})
                    yield ToolCall(
                        name=func_info.get("name"),
                        arguments=func_info.get("arguments", {})
                    )


class NemotronMini(GGUFLLM):
    hf_repo: str = "bartowski/Nemotron-Mini-4B-Instruct-GGUF"
    filename: str = "Nemotron-Mini-4B-Instruct-Q5_K_M.gguf"
    n_ctx: int = 4096
    max_tokens: int = 1024
    decoder = LegacyXMLDecoder()


class Qwen3(GGUFLLM):
    hf_repo: str = "Qwen/Qwen3-4B-GGUF"
    filename: str = "Qwen3-4B-Q5_K_M.gguf"
    n_ctx: int = 40960
    max_tokens: int = 8192
    decoder = LegacyXMLDecoder()


class Gemma4E2B(GGUFLLM):
    hf_repo: str = "unsloth/gemma-4-E2B-it-GGUF"
    filename: str = "gemma-4-E2B-it-Q4_K_M.gguf"
    n_ctx: int = 131072
    max_tokens: int = 8192
    decoder = GemmaE2BDecoder()


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
