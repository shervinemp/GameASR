# utils.py
import logging
import requests
from typing import Dict, Any, Tuple


def setup_logging(log_level=logging.INFO, log_format=None):
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
            level=log_level, format=log_format, datefmt="%Y-%m-%d %H:%M:%S"
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


def map_json_type_to_python(json_type: str) -> str:
    """Maps JSON schema types to Python type hints."""
    type_map = {
        "string": "str",
        "number": "float",
        "integer": "int",
        "boolean": "bool",
        "array": "list",
        "object": "dict",
    }
    return type_map.get(json_type, "Any")


def map_json_return_to_python(
    returns_spec: Dict[str, Any], method_name: str
) -> Tuple[str, str]:
    """
    Maps JSON schema return types to Python type hints and result handling logic
    for client stubs.
    Assumes Lua-facing methods will return a single string.
    """
    return_type = returns_spec["type"]

    if return_type == "string":
        return (
            "Tuple[str | None, str | None]",
            'if isinstance(result, str):\n            return result, None\n        logging.error(f"Unexpected response format for {method_name}: Expected string, got {{type(result).__name__}}.")\n        return None, "Unexpected response format: Expected string."',
        )
    elif return_type == "boolean":
        return (
            "Tuple[bool | None, str | None]",
            'if isinstance(result, bool):\n            return result, None\n        logging.error(f"Unexpected response format for {method_name}: Expected boolean, got {{type(result).__name__}}.")\n        return None, "Unexpected response format: Expected boolean."',
        )
    elif return_type == "number":
        return (
            "Tuple[float | None, str | None]",
            'if isinstance(result, (int, float)):\n            return float(result), None\n        logging.error(f"Unexpected response format for {method_name}: Expected number, got {{type(result).__name__}}.")\n        return None, "Unexpected response format: Expected number."',
        )
    elif return_type == "object":
        return (
            "Tuple[dict | None, str | None]",
            'if isinstance(result, dict):\n            return result, None\n        logging.error(f"Unexpected response format for {method_name}: Expected dict, got {{type(result).__name__}}.")\n        return None, "Unexpected response format: Expected dict."',
        )
    elif return_type == "array":
        return (
            "Tuple[list | None, str | None]",
            'if isinstance(result, list):\n            return result, None\n        logging.error(f"Unexpected response format for {method_name}: Expected list, got {{type(result).__name__}}.")\n        return None, "Unexpected response format: Expected list."',
        )
    else:
        return "Tuple[Any | None, str | None]", "return result, error"


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
        logger.error(f"Failed to write file to {destination}: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(
            f"An unexpected error occurred during download of {url}: {e}", exc_info=True
        )
        raise
