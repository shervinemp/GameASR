#!/usr/bin/env python3
"""
Setup script for LLM module.

This script handles downloading the model from the Hugging Face Hub.
"""

import sys

from . import LLMProviders

from ..common.utils import get_logger, setup_logging
from ..common.config import config


def main():
    """
    Main function to set up the LLM environment.
    """
    setup_logging(stream=sys.stdout)
    logger = get_logger(__name__)

    try:
        provider = config.get("llm.provider")
        getattr(LLMProviders, provider).download()
        logger.info("Model download completed successfully.")
    except Exception as e:
        logger.error(f"Failed to download model: {e}")


if __name__ == "__main__":
    main()
