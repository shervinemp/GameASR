#!/usr/bin/env python3
"""
Setup script for LLM module.

This script handles downloading the model from the Hugging Face Hub.
"""

import os
import sys
from huggingface_hub import hf_hub_download
from ..common.utils import get_logger, setup_logging


def download_hf_file(repo_id: str, filename: str, save_directory: str):
    """
    Downloads a single file from the Hugging Face Hub.

    Args:
        repo_id (str): The Hugging Face repository identifier.
        filename (str): The specific file to download from the repo.
        save_directory (str): The local directory to save the model file.
    """
    logger = get_logger(__name__)
    logger.info(f"Preparing to download '{filename}' from '{repo_id}'...")

    if os.path.exists(os.path.join(save_directory, filename)):
        logger.info(f"File already exist in {save_directory}. Skipping download.")
        return

    logger.info("Starting download...")
    os.makedirs(save_directory, exist_ok=True)

    try:
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=save_directory,
            local_dir_use_symlinks=False,
        )

        print("Model and tokenizer files downloaded successfully.")
    except Exception as e:
        # We catch the exception, print it, and then re-raise it to stop the script
        print(f"An error occurred during download: {e}")
        raise


def main():
    """
    Main function to set up the LLM environment.
    """
    setup_logging(stream=sys.stdout)
    logger = get_logger(__name__)

    repo_id = "bartowski/Nemotron-Mini-4B-Instruct-GGUF"
    filename = "Nemotron-Mini-4B-Instruct-Q4_K_M.gguf"

    models_dir = os.path.join("models", "llm")

    logger.info(f"Ensuring directory exists: {models_dir}")
    os.makedirs(models_dir, exist_ok=True)

    try:
        download_hf_file(repo_id=repo_id, filename=filename, save_directory=models_dir)
        logger.info("Model download completed successfully.")
    except Exception as e:
        logger.error(f"Failed to download model: {e}")


if __name__ == "__main__":
    main()
