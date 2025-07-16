#!/usr/bin/env python3
"""
Setup script for LLM module.

This script handles installing dependencies and setting up the environment.
"""

import os
from ..common.utils import download_file, get_logger


def download_all_files(models_dir: str):
    """
    Download all required files for the LLM model.
    """
    logger = get_logger(__name__)
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
        destination = os.path.join(models_dir, filename)
        if not os.path.exists(destination):
            download_file(url, destination)
        else:
            logger.info(f"File {destination} already exists, skipping download.")


def main():
    """
    Main function to set up the LLM environment.
    """
    logger = get_logger(__name__)

    models_dir = os.path.join("models", "llm")
    os.makedirs(models_dir, exist_ok=True)
    logger.info(f"Ensured directory exists: {models_dir}")

    download_all_files(models_dir)

    logger.info("LLM setup completed successfully.")


if __name__ == "__main__":
    main()
