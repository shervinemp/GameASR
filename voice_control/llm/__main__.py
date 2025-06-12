#!/usr/bin/env python3
"""
Main entry point for the LLM module.

This script tests the language model functionality.
"""

from .core import LLMCore
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

    logger.info("LLM Service initialized successfully.")

    # Test generating a response
    prompt = "Tell me a joke"
    response = llm_service.generate_response(prompt)

    logger.info(f"Prompt: {prompt}")
    logger.info(f"Response: {response}")


if __name__ == "__main__":
    main()
