#!/usr/bin/env python3
"""
Main entry point for the LLM module.

This script tests the language model functionality.
"""

from .core import LLMCore, Tool, Parameter
from ..common.logging_utils import setup_logging, get_logger


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
        Tool(
            name="get_current_weather",
            description="Get the current weather in a given location",
            parameters=[
                Parameter(
                    type="string",
                    description="The city and state, e.g. San Francisco, CA",
                    properties={},
                    required=None,
                ),
                Parameter(
                    type="string",
                    description="The temperature unit to use. Infer this from the users location.",
                    enum=["celsius", "fahrenheit"],
                ),
            ],
        ),
        Tool(
            name="get_stock_price",
            description="Get the current stock price",
            parameters=[
                Parameter(
                    type="string",
                    description="The stock symbol",
                    properties={},
                    required=None,
                ),
            ],
        ),
    ]

    logger.info("LLM Service initialized successfully.")

    # Test generating a response
    prompt = "Tell me a joke"
    response = llm_service.generate_response(prompt, tool_use=False)

    logger.info("Tool use disabled.")
    logger.info(f"Prompt: {prompt}")
    logger.info(f"Response: {response}")

    prompt = (
        "What is the weather like in Beijing now and what's the stock price of NVDA?"
    )
    response = llm_service.generate_response(prompt, tool_use=True)

    logger.info("Tool use enabled.")
    logger.info(f"Prompt: {prompt}")
    logger.info(f"Response: {response}")


if __name__ == "__main__":
    main()
