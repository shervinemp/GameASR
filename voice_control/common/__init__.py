"""
Common Utilities Package

This package contains shared utilities that can be used by various voice control modules:
- logging_utils: Standardized logging configuration
- base_component: Base classes for components with common interfaces
"""

from .logging_utils import get_logger, setup_logging
from .base_component import BaseComponent

__all__ = ["BaseComponent", "get_logger", "setup_logging"]
