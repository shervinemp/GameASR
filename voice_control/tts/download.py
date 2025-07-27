#!/usr/bin/env python3
"""
Setup script for Text-to-Speech (TTS) system using Kokoro-ONNX.

This script handles installing dependencies and setting up the environment.
"""

import sys

from voice_control.tts.model import TTS
from ..common.utils import get_logger, setup_logging


def main():
    """
    Main function to set up the TTS environment.
    """
    setup_logging(stream=sys.stdout)
    logger = get_logger(__name__)

    try:
        TTS().download()
        logger.info("Model download completed successfully.")
    except Exception as e:
        logger.error(f"Failed to download model: {e}")


if __name__ == "__main__":
    main()
