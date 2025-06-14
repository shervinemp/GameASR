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


def parse_asr_args():
    """Parses command-line arguments for the ASR system."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Real-time, continuous ASR from microphone using ONNX models and VAD."
    )

    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=0.3,
        help="Probability threshold for Voice Activity Detection (0.0-1.0). Higher values are less sensitive to noise.",
    )
    parser.add_argument(
        "--end-silence-duration",
        type=float,
        default=0.7,
        help="Duration of continuous silence (in seconds) to consider an utterance ended.",
    )
    parser.add_argument(
        "--pre-speech-duration",
        type=float,
        default=0.8,
        help="Duration of audio (in seconds) to include before detected speech starts, for context.",
    )
    parser.add_argument(
        "--queue-size",
        type=int,
        default=5,
        help="Maximum number of utterances to queue for ASR processing. Prevents memory overload.",
    )
    parser.add_argument(
        "--sound-device",
        type=int,
        help="Specific audio input device ID. Use `python -m sounddevice` to list devices.",
    )
    return parser.parse_args()


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
        asr_core = ASRCore(**parse_asr_args().__dict__)

        # Start processing audio from the microphone
        logger.info("Starting ASR pipeline...")
        asr_core.process_audio()
    except Exception as e:
        logger.error(f"Error in main(): {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Run the main function
    main()
