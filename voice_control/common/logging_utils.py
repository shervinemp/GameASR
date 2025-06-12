#!/usr/bin/env python3
"""
Module for standardizing logging across all components.

This module provides helper functions and configurations for consistent logging.
"""

import logging


def setup_logging(log_level=logging.INFO, log_format=None):
    """
    Configure the logging system with a standardized format and level.

    Args:
        log_level: The minimum severity level to log (default: INFO)
        log_format: A custom format string or None for default format
    """
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logging.basicConfig(level=log_level, format=log_format, datefmt="%Y-%m-%d %H:%M:%S")


def get_logger(name):
    """
    Get a logger with the given name.

    Args:
        name: The name of the logger

    Returns:
        logging.Logger: A configured logger instance
    """
    return logging.getLogger(name)


# Example usage (can be removed in production)
if __name__ == "__main__":
    # Configure logging for this module
    setup_logging(log_level=logging.DEBUG)

    # Get a logger with a specific name
    logger = get_logger(__name__)

    # Test different log levels
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
