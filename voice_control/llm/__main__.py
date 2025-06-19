#!/usr/bin/env python3
"""
Main entry point for the LLM module.

This script tests the language model functionality.
"""

from .core import LLMCore
from ..common.utils import setup_logging, get_logger


def main():
    """
    Main function to test the LLM module.
    """
    # Configure logging for this session
    setup_logging(log_level="DEBUG")

    # Get a logger for this module
    logger = get_logger(__name__)

    # Initialize the LLM service
    llm_service = LLMCore()
    llm_service.tools = [
        {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": [
                {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                {
                    "type": "string",
                    "description": "The temperature unit to use. Infer this from the users location.",
                    "enum": ["celsius", "fahrenheit"],
                },
            ],
        },
        {
            "name": "get_stock_price",
            "description": "Get the current stock price",
            "parameters": [
                {
                    "type": "string",
                    "description": "The stock symbol",
                },
            ],
        },
    ]

    logger.info("LLM Service initialized successfully.")

    # Test generating a response
    prompt = "Tell me a joke"

    messages = llm_service.create_messages(prompt, no_tool=True)
    logger.info("No tool use forced.")

    response = llm_service.generate_response(messages)

    logger.info(f"Prompt: {prompt}")
    logger.info(f"Response: {response}")

    prompt = (
        "What is the weather like in Beijing now and what's the stock price of NVDA?"
    )

    messages = llm_service.create_messages(prompt, tool_only=True)
    logger.info("Tool use forced.")

    output = llm_service.generate_response(messages)
    response, tool_calls = llm_service.parse_response(output)

    logger.info(f"Prompt: {prompt}")
    logger.info(f"Response: {response}")
    logger.info(f"Tool calls: {tool_calls}")


if __name__ == "__main__":
    main()
