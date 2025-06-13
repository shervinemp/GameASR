""" "
Core module for LLM functionality.

This module provides the main classes and functions for working with language models.
"""

from dataclasses import asdict, dataclass, field
import json
import onnxruntime_genai as og

from ..common.logging_utils import get_logger

# Get a logger for this module
logger = get_logger(__name__)


@dataclass
class Parameter:
    type: str
    description: str
    properties: dict | None = field(default_factory=dict)
    required: list | None = None
    enum: list | None = None


@dataclass
class Tool:
    name: str
    description: str
    parameters: list[Parameter] = field(default_factory=list)

    def __str__(self):
        tool_dict = asdict(self)
        tool_dict["parameters"] = list(map(asdict, self.parameters))
        return f"<tool> {tool_dict} </tool>"


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
        self.tools = []
        self.contexts = []
        self.system_prompt = (
            "You are a helpful assistant. "
            "You can answer questions, provide information, and assist with various tasks."
            "If you don't know the answer, you can say 'I don't know'."
        )

    def generate_response(self, prompt, tool_use: bool = True) -> str:
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
        messages = self._create_messages(prompt, tool_use)

        # Apply chat template to the prompt
        tokenizer_input_system_prompt = self.tokenizer.apply_chat_template(
            messages=json.dumps(messages),
            add_generation_prompt=False,
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

    def _create_messages(
        self,
        query: str,
        tool_use: bool = True,
    ) -> list[dict]:
        """
        Generate a list of messages for the LLM prompt.

        This function combines system, user, and tool messages into a structured format
        that can be used in LLM prompts.

        Args:
            query (str): The user query to format.
        """

        tools_string = ""
        if tool_use:
            tools_string = "\n".join(str(tool) for tool in self.tools)
        contexts_string = "\n".join(
            map(lambda x: f"<context> {x} </context>", self.contexts)
        )

        messages = (
            {
                "role": "system",
                "content": (
                    f"{self.system_prompt}\n" f"{tools_string}\n" f"{contexts_string}"
                ),
            },
            {"role": "user", "content": query},
            {"role": "assistant", "content": '<toolcall>{"' if tool_use else ""},
        )

        return messages
