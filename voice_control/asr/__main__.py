#!/usr/bin/env python3
"""
Main entry point for the Automatic Speech Recognition (ASR) system.

This script demonstrates how to set up and run the ASR pipeline.
"""

import sys

# Import standard logging utilities
from ..common.logging_utils import setup_logging, get_logger

# Import the core ASR component
from .core import ASRCore


def main():
    """
    Main function to run the ASR system.
    """
    # Configure logging for this session
    setup_logging(log_level="DEBUG")

    # Get a logger for this module
    logger = get_logger(__name__)

    try:
        # Create an instance of ASRCore
        asr_core = ASRCore()

        # Start processing audio from the microphone
        logger.info("Starting ASR pipeline...")
        asr_core.process_audio()
    except Exception as e:
        logger.error(f"Error in main(): {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Run the main function
    main()
