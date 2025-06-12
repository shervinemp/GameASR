""" "
Core module for LLM functionality.

This module provides the main classes and functions for working with language models.
"""

import onnxruntime_genai as og

from ..common.logging_utils import get_logger

# Get a logger for this module
logger = get_logger(__name__)


class LLMCore:
    """
    Core class for interacting with language models.
    """

    def __init__(self):
        """
        Initialize the LLM core.
        """
        model_path = "models\\llm"
        self.config = og.Config(model_path)
        self.model = og.Model(self.config)
        self.tokenizer = og.Tokenizer(self.model)
        self.system_prompt = (
            "You are a helpful assistant. "
            "You can answer questions, provide information, and assist with various tasks. "
            "If you don't know the answer, you can say 'I don't know'."
        )

    def _convert_messages_to_string(self, messages):
        """
        Convert a list of message dictionaries to a JSON-formatted string.

        Args:
            messages (list): List of dictionaries with 'role' and 'content'.

        Returns:
            str: JSON-formatted string representation of the messages.
        """
        import json

        return json.dumps(messages)

    def generate_response(self, prompt):
        """
        Generate a response from the language model.

        Args:
            prompt (str): The input text to generate a response for.

        Returns:
            str: The generated response.
        """
        # Create a generator with the specified parameters
        params = og.GeneratorParams(self.model)
        generator = og.Generator(self.model, params)

        # Apply chat template to the prompt
        tokenizer_input_system_prompt = self.tokenizer.apply_chat_template(
            messages=self._convert_messages_to_string(
                [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ]
            ),
            add_generation_prompt=True,
        )

        input_tokens = self.tokenizer.encode(tokenizer_input_system_prompt)
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

            return output
        except Exception as e:
            logger.error(f"Error generating tokens: {e}")

        return ""


def create_tool_call_template(tools_list):
    """
    Create a tool call template for use in prompts.

    Args:
        tools_list (list): List of tool definitions to include in the template.

    Returns:
        str: Template string with tool definitions.
    """
    template = "<tool> "
    for tool in tools_list:
        template += f'{{"name": "{tool["name"]}", "description": "{tool["description"]}", "parameters": {str(tool["parameters"])}}} </tool>'
    return template
