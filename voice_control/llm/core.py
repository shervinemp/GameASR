""" "
Core module for LLM functionality.

This module provides the main classes and functions for working with language models.
"""

import json
import onnxruntime_genai as og

from ..common.utils import get_logger


class LLMCore:
    """
    Core class for interacting with language models.
    """

    def __init__(self, tool_use: bool = True):
        """
        Initialize the LLM core.
        """
        self.logger = get_logger(__name__)

        model_path = "models\\llm"
        self.config = og.Config(model_path)
        self.model = og.Model(self.config)
        self.tokenizer = og.Tokenizer(self.model)
        self.tool_use = tool_use
        self.tools = []
        self.contexts = []
        self.system_prompt = (
            "You are a helpful assistant."
            "You can answer questions, provide information, and assist with various tasks."
            "If you don't know the answer, you can say 'I don't know'."
        )

    def generate_response(self, messages: list[dict]) -> str:
        """
        Generate a response from the language model.

        Args:
            messages (list[dict]): The input messages to generate a response for.

        Returns:
            str: The generated response.
        """
        # Create a generator with the specified parameters
        params = og.GeneratorParams(self.model)
        generator = og.Generator(self.model, params)

        # Apply chat template to the prompt
        tokenizer_input = self.tokenizer.apply_chat_template(
            messages=json.dumps(messages),
            add_generation_prompt=False,
        )
        self.logger.debug(f"LLM input:\n{tokenizer_input}")

        input_tokens = self.tokenizer.encode(tokenizer_input)
        generator.append_tokens(input_tokens)

        # Generate the response
        output = ""
        try:
            max_tokens = 100  # Limit to prevent infinite loop
            tokenizer_stream = self.tokenizer.create_stream()
            for i in range(max_tokens):
                if generator.is_done():
                    break
                generator.generate_next_token()
                token = generator.get_next_tokens()[0]
                token_str = tokenizer_stream.decode(token)
                output += token_str
                if len(token_str) == 0:
                    break
        except Exception as e:
            self.logger.error(f"Error generating tokens: {e}")

        self.logger.debug(f"LLM ouput:\n{output}")

        return output

    def create_messages(
        self,
        query: str,
        tool_only: bool = False,
        no_tool: bool = False,
    ) -> list[dict]:
        """
        Generate a list of messages for the LLM prompt.

        This function combines system, user, and tool messages into a structured format
        that can be used in LLM prompts.

        Args:
            query (str): The user query to format.
            tool_only (bool): If True, the response will only include tool calls.
            no_tool (bool): If True, tools will not be included in the system prompt.

        Returns:
            list[dict]: A list of dictionaries representing the messages.
        """

        tools_string = ""
        if self.tool_use and not no_tool:
            tools_string = "</tool>\n<tool>".join(str(tool) for tool in self.tools)
        contexts_string = "\n".join(
            map(lambda x: f"<context> {x} </context>", self.contexts)
        )

        messages = [
            {
                "role": "system",
                "content": self.system_prompt
                + "\n"
                + tools_string
                + "\n"
                + contexts_string,
            },
            {"role": "user", "content": query},
            {"role": "assistant", "content": "<toolcall>" if tool_only else ""},
        ]

        return messages

    def parse_response(self, response: str) -> tuple[str, list[dict]]:
        """
        Parse the LLM response into a tuple of text and tool calls.

        Args:
            response (str): The raw response from the language model.

        Returns:
            tuple: A tuple containing:
                - str: Textual response
                - list: List of dictionaries with tool call information
        """
        try:
            text = ""
            tool_calls = []

            beg_marker = "<toolcall>"
            end_marker = "</toolcall>"
            beg_len = len(beg_marker)

            # Split response into parts based on tool call markers
            parts = response.split(end_marker)
            for part in parts[:-1]:
                marker_idx = part.find(beg_marker)
                start_idx = (marker_idx + beg_len) if marker_idx != -1 else 0

                text += part[:marker_idx].strip() + "\n"
                tool_call_json = part[start_idx:].strip()
                tool_call_json = tool_call_json.replace("'", '"').replace("\n", "")

                try:
                    tool_calls.append(json.loads(tool_call_json))
                except json.JSONDecodeError as e:
                    self.logger.error(f"Invalid JSON in tool call: {e}")

            text += parts[-1].strip() + "\n"

            return text, tuple(tool_calls)
        except Exception as e:
            self.logger.error(f"Error parsing response: {e}")
            return "", []
