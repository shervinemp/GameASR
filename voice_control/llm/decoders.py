from abc import ABC, abstractmethod
import json
import re
from typing import Generator, Iterator

from ..common.utils import get_logger
from .tools import ToolCall

_log = get_logger(__name__)


class StreamDecoder(ABC):
    """Strategy interface for interpreting and intercepting LLM streams."""
    @abstractmethod
    def __call__(self, stream: Iterator[str | dict | ToolCall]) -> Generator[str | dict | ToolCall, None, None]: ...


class NativeDecoder(StreamDecoder):
    """Yields stream exactly as it arrives (used for OpenAI, Gemini, and native tools)."""
    def __call__(self, stream: Iterator[str | dict | ToolCall]) -> Generator[str | dict | ToolCall, None, None]:
        yield from stream


class LegacyXMLDecoder(StreamDecoder):
    """Legacy parser for standard <toolcall>...</toolcall> used by Qwen/Nemotron."""
    def __call__(self, stream: Iterator[str | dict | ToolCall]) -> Generator[str | dict | ToolCall, None, None]:
        buffer, in_tool = "", False
        for chunk in stream:
            if isinstance(chunk, (dict, ToolCall)):
                yield chunk
                return
            buffer += chunk

            while buffer:
                if in_tool:
                    if "</toolcall>" in buffer:
                        try:
                            tool_body = buffer.split("</toolcall>")[0]
                            tool_dict = json.loads(tool_body.strip())
                            yield ToolCall(
                                name=tool_dict.get("name") or tool_dict.get("function"),
                                arguments=tool_dict.get("arguments", {})
                            )
                        except Exception:
                            _log.warning(
                                "Failed to parse toolcall: %s", tool_body[:200]
                            )
                            yield ToolCall(name="_parse_error", arguments={"raw": tool_body})
                        return  # Halt stream to execute tool
                    else:
                        if len(buffer) > 10_000:
                            _log.warning(
                                "Toolcall buffer exceeded 10K chars, discarding"
                            )
                            yield buffer
                            buffer = ""
                        break  # Wait for more chunks
                else:
                    if "<toolcall>" in buffer:
                        pre = buffer.split("<toolcall>")[0]
                        if pre:
                            yield pre
                        in_tool = True
                        buffer = buffer.split("<toolcall>", 1)[1]
                        continue  # Re-evaluate buffer
                    else:
                        match = re.search(r'<t(?:o(?:o(?:l(?:c(?:a(?:l(?:l)?)?)?)?)?)?)?)?$', buffer)
                        if match:
                            safe_idx = match.start()
                            if safe_idx > 0:
                                yield buffer[:safe_idx]
                                buffer = buffer[safe_idx:]
                            break  # Wait for more chunks to complete the tag
                        else:
                            yield buffer
                            buffer = ""
                            break

        if buffer and not in_tool:
            yield buffer


class GemmaE2BDecoder(StreamDecoder):
    """Sliding-window parser for Gemma 4's tool call format.

    Gemma 4 uses ``<|tool_call|>`` for both opening *and* closing:
    ``<|tool_call|>call:func(args)<|tool_call|>`` — the first occurrence
    opens tool mode, the second closes it and yields a ``ToolCall``.
    """

    _TAG = "<|tool_call|>"

    def __call__(self, stream: Iterator[str | dict | ToolCall]) -> Generator[str | dict | ToolCall, None, None]:
        buffer = ""
        in_tool = False
        in_thought = False

        for chunk in stream:
            if isinstance(chunk, (dict, ToolCall)):
                yield chunk
                return

            buffer += chunk

            while buffer:
                # 1. Tool mode — looking for the *closing* <|tool_call|>
                if in_tool:
                    idx = buffer.find(self._TAG)
                    if idx != -1:
                        body = buffer[:idx].strip()
                        if body.startswith("call:"):
                            match = re.match(r"call:([a-zA-Z0-9_]+)(.*)", body, re.DOTALL)
                            if match:
                                name = match.group(1).strip()
                                args = match.group(2).strip()
                                args = args.replace('<|"|>', '"')
                                args = re.sub(r'([{,]\s*)([a-zA-Z0-9_]+)(\s*:)', r'\1"\2"\3', args)
                                try:
                                    yield ToolCall(name=name, arguments=json.loads(args))
                                except json.JSONDecodeError:
                                    _log.warning("Failed to parse gemma toolcall args: %s", args[:200])
                                    yield ToolCall(name="_parse_error", arguments={"raw": args})
                        # CRITICAL: halt so the pipeline dispatches the tool
                        return
                    break  # Wait for more chunks

                # 2. Filter Thoughts
                if in_thought:
                    if "<channel|>" in buffer:
                        in_thought = False
                        buffer = buffer.split("<channel|>", 1)[1]
                        continue
                    break

                # 3. Detect openers — first <|tool_call|> = open tool mode
                idx = buffer.find(self._TAG)
                if idx != -1:
                    in_tool = True
                    pre = buffer[:idx]
                    if pre:
                        yield pre
                    buffer = buffer[idx + len(self._TAG):]
                    continue

                if "<|channel>thought" in buffer:
                    in_thought = True
                    pre = buffer.split("<|channel>thought")[0]
                    if pre:
                        yield pre
                    buffer = buffer.split("<|channel>thought", 1)[1]
                    continue

                # 4. Safe Yield — hold back partial tags
                safe_idx = max(buffer.rfind("<"), buffer.rfind("&"))
                if safe_idx != -1 and (len(buffer) - safe_idx) < 15:
                    if safe_idx > 0:
                        yield buffer[:safe_idx]
                        buffer = buffer[safe_idx:]
                    break
                else:
                    if buffer:
                        yield buffer
                    buffer = ""
                    break

        # Flush — yield leftover text even when a tool / thought was unclosed
        if buffer:
            if in_tool:
                _log.warning("Gemma tool call was opened but never closed — yielding as text")
            yield buffer
