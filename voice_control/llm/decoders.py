from abc import ABC, abstractmethod
import json
import re
from typing import Generator, Iterator

from .tools import ToolCall


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
                        if len(buffer) > 500:
                            yield buffer
                            buffer = ""
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
