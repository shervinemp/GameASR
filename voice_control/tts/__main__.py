#!/usr/bin/env python3
"""
Main entry point for the Text-to-Speech (TTS) system.

This script demonstrates how to set up and run a continuous TTS loop that gets text from user input.
"""

import sys

# Import standard logging utilities
from ..common.utils import setup_logging, get_logger

# Import the core TTS component
from .core import TTSCore


def main():
    """
    Main function to run a continuous TTS loop.
    """
    # Configure logging for this session
    setup_logging(log_level="DEBUG")

    # Get a logger for this module
    logger = get_logger(__name__)

    try:
        # Create an instance of TTSCore
        tts_core = TTSCore()

        # Continuous TTS loop
        logger.info("Starting continuous TTS loop. Type 'exit' to quit.")
        while True:
            try:
                # Get text input from user
                user_input = input("Enter text to speak (or 'exit' to quit): ")

                if user_input.lower() == "exit":
                    logger.info("Exiting TTS loop...")
                    break

                # Convert text to speech and play it
                tts_core.speak(user_input)
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt detected. Exiting...")
                break
            except Exception as e:
                logger.error(f"Error in TTS processing: {e}")

    except Exception as e:
        logger.error(f"Error in main(): {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Run the main function
    main()
