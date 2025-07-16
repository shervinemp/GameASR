#!/usr/bin/env python3
"""
Main entry point for the LLM module.

This script tests the language model functionality.
"""

from . import Session, SystemPrompt, Tool
from ..common.utils import setup_logging, get_logger


def main():
    """
    Main function to test the LLM module.
    """
    setup_logging(log_level="DEBUG")
    logger = get_logger(__name__)

    tools = list(
        map(
            Tool.from_dict,
            [
                {
                    "name": "get_current_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                            },
                            "unit": {
                                "type": "string",
                                "description": "The temperature unit to use. Infer this from the users location.",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                    },
                },
                {
                    "name": "get_stock_price",
                    "description": "Get the current stock price",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "stock": {
                                "type": "string",
                                "description": "The stock symbol",
                            },
                        },
                    },
                },
            ],
        )
    )
    tools[0].callback = lambda **kwargs: "Rainy"
    tools[1].callback = lambda **kwargs: 100

    session = Session()
    sys_prompt = SystemPrompt(
        (
            "You are a helpful assistant."
            "You can answer questions, provide information, and assist with various tasks."
            "If you don't know the answer, you can say 'I don't know'."
        ),
        tools=tools,
    )
    session.conversation.add_system_message(str(sys_prompt))
    logger.info("Session initialized successfully.")

    prompt = (
        "What is the weather like in Beijing now and what's the stock price of NVDA?"
    )
    logger.info(f"Prompt: {prompt}")

    response = "".join(session(prompt))
    print(response)

    logger.info(f"Response: {response}")


if __name__ == "__main__":
    main()
