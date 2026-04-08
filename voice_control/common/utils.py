import json
import logging
import os
import re
from huggingface_hub import hf_hub_download
import requests
from typing import Dict, Any


def safe_json_loads(text: str, fallback: Any = None) -> Any:
    """Robustly parses JSON from LLM output, stripping Markdown/filler."""
    try:
        match = re.search(r'(\[.*\]|\{.*\})', text.strip(), re.DOTALL)
        clean_text = match.group(0) if match else text
        return json.loads(clean_text)
    except (json.JSONDecodeError, AttributeError):
        return fallback if fallback is not None else []


def setup_logging(log_level=logging.INFO, log_format=None, stream=None):
    """
    Configure the logging system with a standardized format and level.

    Args:
        log_level: The minimum severity level to log (default: INFO)
        log_format: A custom format string or None for default format
    """
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Ensure logging is only configured once to avoid duplicate handlers
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=log_level,
            format=log_format,
            datefmt="%Y-%m-%d %H:%M:%S",
            stream=stream,
            force=True
        )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the given name.

    Args:
        name: The name of the logger

    Returns:
        logging.Logger: A configured logger instance
    """
    return logging.getLogger(name)


def download_hf_file(repo_id: str, filename: str, directory: str):
    """
    Downloads a single file from the Hugging Face Hub.

    Args:
        repo_id (str): The Hugging Face repository identifier.
        filename (str): The specific file to download from the repo.
        directory (str): The local directory to save the model file.
    """
    logger = get_logger(__name__)
    logger.info(f"Preparing to download '{filename}' from '{repo_id}'...")

    if os.path.exists(os.path.join(directory, filename)):
        logger.info(f"File already exist in {directory}. Skipping download.")
        return

    logger.info("Starting download...")
    os.makedirs(directory, exist_ok=True)

    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=directory,
    )


def download_file(url: str, destination: str):
    """
    Download a file from a URL to a specified destination.

    Args:
        url: The URL of the file to download.
        destination: The local path to save the file.
    """
    logger = get_logger(__name__)  # Get logger inside function
    try:
        response = requests.get(url, stream=True, allow_redirects=True)
        response.raise_for_status()  # Raise an error for bad responses

        with open(destination, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        logger.info(f"Downloaded {url} to {destination}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download {url}: {e}", exc_info=True)
        raise
    except IOError as e:
        logger.error(
            f"Failed to write file to {destination}: {e}", exc_info=True
        )
        raise
    except Exception as e:
        logger.error(
            f"An unexpected error occurred during download of {url}: {e}",
            exc_info=True,
        )
        raise


def load_specs(specs_path: str) -> Dict[str, Any]:
    with open(specs_path, "r") as f:
        return json.load(f)
