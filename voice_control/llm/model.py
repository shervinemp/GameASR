""" "
Core module for LLM functionality.

This module provides the main classes and functions for working with language models.
"""

import json
import os
from typing import Generator
import onnxruntime_genai as og

from .conversation import Message

from ..common.utils import get_logger


class LLM:

    def __init__(self):
        self.logger = get_logger(__name__)

        model_path = os.path.join("models", "llm")
        self.config = og.Config(model_path)
        self.model = og.Model(self.config)
        self.tokenizer = og.Tokenizer(self.model)
        self.max_tokens = 4096
        self._toolcall_markers = ("<toolcall>", "</toolcall>")

    def __call__(
        self,
        messages: list[dict],
        force_toolcall=True,
    ) -> Generator[str | dict, None, None]:
        """
        Generate and parse text from the language model.

        Args:
            messages (list[dict]): A list of messages to provide to the language model

        Yields:
            str | dict: The generated text or a tool call
        """
        yield from self._parse(self._generate(messages, force_toolcall=force_toolcall))

    def _generate(
        self,
        messages: list[dict],
        force_toolcall=True,
    ) -> Generator[str, None, None]:
        """
        Generate and parse text from the language model.

        Args:
            messages (list[dict]): A list of messages to provide to the language model

        Yields:
            str : The generated text
        """
        params = og.GeneratorParams(self.model)
        generator = og.Generator(self.model, params)

        if force_toolcall:
            messages = messages + [
                Message(role=Message.Role.system, content=self._toolcall_markers[0])
            ]

        tokenizer_input = self.tokenizer.apply_chat_template(
            messages=json.dumps(messages),
            add_generation_prompt=(not force_toolcall),
        )
        self.logger.debug(f"LLM input:\n{tokenizer_input}")

        input_tokens = self.tokenizer.encode(tokenizer_input)
        generator.append_tokens(input_tokens)

        try:
            tokenizer_stream = self.tokenizer.create_stream()
            for i in range(self.max_tokens):
                if generator.is_done():
                    break
                generator.generate_next_token()
                token = generator.get_next_tokens()[0]
                token_str = tokenizer_stream.decode(token)
                if len(token_str) == 0:
                    break
                yield token_str
        except Exception as e:
            self.logger.error(f"Error generating tokens: {e}")

    def _parse(self, response: Generator[str]) -> Generator[str | dict, None, None]:
        """
        Parse the response from the language model.

        Args:
            response (Generator[str]): A generator of strings from the language model

        Yields:
            str | dict: The parsed text or a tool call
        """
        try:
            beg_marker, end_marker = self._toolcall_markers
            beg_len, end_len = len(beg_marker), len(end_marker)

            buffer = ""
            is_call = False
            for chunk in response:
                buffer += chunk

                if not is_call:
                    is_call = buffer.startswith(beg_marker[: len(buffer)])
                    if not is_call:
                        yield buffer
                        buffer = ""

                if is_call and buffer.endswith(end_marker):
                    tool_text = buffer[beg_len:-end_len].replace("'", '"')
                    try:
                        tool_call = json.loads(tool_text)
                        yield tool_call
                    except json.JSONDecodeError as e:
                        self.logger.error(f"Invalid JSON in tool call: {e}")

                    buffer = ""
                    is_call = False
            else:
                yield buffer

        except Exception as e:
            self.logger.error(f"Error parsing response: {e}")
