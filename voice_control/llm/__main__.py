#!/usr/bin/env python3
"""
Main entry point for the LLM module.

This script tests the language model functionality.
"""

from .core import LLMCore


def main():
    """
    Main function to test the LLM module.
    """
    # Initialize the LLM service
    llm_service = LLMCore()

    print("LLM Service initialized successfully.")

    # Test generating a response
    prompt = "Tell me a joke"
    response = llm_service.generate_response(prompt)

    print(f"Prompt: {prompt}")
    print(f"Response: {response}")


if __name__ == "__main__":
    main()
