#!/usr/bin/env python3
"""
Setup script for LLM module.

This script handles installing dependencies and setting up the environment.
"""

import os
from ..common.utils import download_file


def setup_environment():
    """
    Set up the environment for LLM by creating necessary directories.
    """
    # Create the models directory if it doesn't exist
    models_dir = os.path.join("models", "llm")
    if not os.path.exists(models_dir):
        try:
            os.makedirs(models_dir)
            print(f"Created directory: {models_dir}")
        except Exception as e:
            print(f"Failed to create directory {models_dir}: {e}")


def download_all_files():
    """
    Download all required files for the LLM model.
    """
    base_url = "https://api.ngc.nvidia.com/v2/models/org/nvidia/nemotoron-mini-4b-instruct-onnx-int4-rtx/1.0/files?redirect=true&path="
    files_to_download = [
        "genai_config.json",
        "model.onnx",
        "model.onnx_data",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]

    for filename in files_to_download:
        url = f"{base_url}{filename}"
        destination = os.path.join("models", "llm", filename)
        if not os.path.exists(destination):
            download_file(url, destination)
        else:
            print(f"File {destination} already exists, skipping download.")


def main():
    """
    Main function to set up the LLM environment.
    """
    # Set up the environment (create directories)
    setup_environment()

    # Download all required files
    download_all_files()

    print("LLM setup completed successfully.")


if __name__ == "__main__":
    main()
