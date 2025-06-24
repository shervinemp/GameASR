#!/usr/bin/env python3
"""
Setup script for Text-to-Speech (TTS) system using Kokoro-ONNX.

This script handles installing dependencies and setting up the environment.
"""

import os
from ..common.utils import download_file


def main():
    """
    Main function to set up the TTS environment.
    """

    # Check if required files exist
    required_files = [
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
    ]

    for file_path in required_files:
        if not os.path.exists(file_path):
            # Download the file if it does not exist
            destination = os.path.join(
                os.getcwd(), "models", "tts", os.path.basename(file_path)
            )
            os.makedirs(os.path.split(destination[0]), exist_ok=True)
            download_file(file_path, destination)
        else:
            print(f"File {file_path} already exists, skipping download.")

    print("TTS setup completed successfully.")


if __name__ == "__main__":
    main()
