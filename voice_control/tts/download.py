#!/usr/bin/env python3
"""
Setup script for Text-to-Speech (TTS) system using Kokoro-ONNX.

This script handles installing dependencies and setting up the environment.
"""

import os
from ..common.utils import download_file, get_logger


def main():
    """
    Main function to set up the TTS environment.
    """
    logger = get_logger(__name__)

    # Define the destination directory for TTS models
    models_dir = os.path.join("model_files", "tts")
    os.makedirs(models_dir, exist_ok=True)
    logger.info(f"Ensured directory exists: {models_dir}")

    # List of required model files and their download URLs
    required_files = [
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
    ]

    for url in required_files.items():
        filename = url.split("/")[-1]
        destination = os.path.join(models_dir, filename)
        if not os.path.exists(destination):
            # Download the file if it does not exist
            download_file(url, destination)
        else:
            logger.info(f"File {destination} already exists, skipping download.")

    logger.info("TTS setup completed successfully.")


if __name__ == "__main__":
    main()
