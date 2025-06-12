"""
Core module for LLM functionality.

This module provides the main classes and functions for working with language models.
"""

import onnxruntime_genai as og


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
                [{"role": "system", "content": prompt}]
            ),
            add_generation_prompt=True,
        )

        input_tokens = self.tokenizer.encode(tokenizer_input_system_prompt)
        generator.append_tokens(input_tokens)

        # Generate the response
        output = ""
        while not generator.is_done():
            try:
                next_tokens = generator.get_next_tokens()
                if next_tokens is not None and len(next_tokens) > 0:
                    for token in next_tokens:
                        decoded = self.tokenizer.decode(token)
                        output += decoded
            except Exception as e:
                print(f"Error generating or decoding tokens: {e}")
                break

        return output

    def generate_with_guidance(self, prompt, guidance_type, guidance_input):
        """
        Generate a response with structured guidance (e.g., JSON schema).

        Args:
            prompt (str): The input text to generate a response for.
            guidance_type (str): Type of guidance to use ('json_schema', 'lark_grammar').
            guidance_input (dict or str): Guidance input based on the type.

        Returns:
            str: The generated response following the guidance structure.
        """
        params = og.GeneratorParams(self.model)

        # Set the guidance
        params.set_guidance(guidance_type, guidance_input)

        generator = og.Generator(self.model, params)

        # Apply chat template to the prompt
        tokenizer_input_system_prompt = self.tokenizer.apply_chat_template(
            messages=self._convert_messages_to_string(
                [{"role": "system", "content": prompt}]
            ),
            add_generation_prompt=False,
        )

        input_tokens = self.tokenizer.encode(tokenizer_input_system_prompt)
        generator.append_tokens(input_tokens)

        # Generate the response with guidance
        output = ""
        while not generator.is_done():
            try:
                next_tokens = generator.get_next_tokens()
                if next_tokens is not None and len(next_tokens) > 0:
                    for token in next_tokens:
                        decoded = self.tokenizer.decode(token)
                        output += decoded
            except Exception as e:
                print(f"Error generating or decoding tokens: {e}")
                break

        return output


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
