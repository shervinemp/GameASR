"""
Configuration management for the Voice Control application.

This module provides a singleton `Config` class that handles loading
default and user-provided configurations from YAML files.
"""

import os
import yaml
from typing import Any


class Config:
    """
    A singleton class to manage application configuration.

    It loads a default configuration from a YAML file shipped with the package,
    and then overrides it with a user-provided configuration file if one exists
    in the project root.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Config, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self, default_config_path=None, user_config_path=None):
        # The __init__ will be called every time Config() is invoked,
        # but we use a flag to ensure the loading logic runs only once.
        if hasattr(self, "_initialized") and self._initialized:
            return

        if default_config_path is None:
            # Look for the default config relative to this file's location
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            default_config_path = os.path.join(base_dir, "config.defaults.yaml")

        if user_config_path is None:
            # Look for user config in the current working directory
            user_config_path = os.path.join(os.getcwd(), "config.yaml")

        self.config = self._load_config(default_config_path)

        if os.path.exists(user_config_path):
            user_config = self._load_config(user_config_path)
            self._deep_merge(self.config, user_config)

        self._initialized = True

    def _load_config(self, path: str) -> dict:
        """Loads a YAML configuration file."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            # This should only happen if the default config is missing,
            # which is a package error. For user configs, we check for
            # existence first.
            raise RuntimeError(f"Configuration file not found at {path}")
        except yaml.YAMLError as e:
            raise RuntimeError(f"Error parsing YAML file at {path}: {e}")

    def _deep_merge(self, source: dict, destination: dict) -> dict:
        """
        Deeply merges the destination dict into the source dict.
        Overwrites values in source with values from destination.
        """
        for key, value in destination.items():
            if isinstance(value, dict):
                node = source.setdefault(key, {})
                self._deep_merge(value, node)
            else:
                source[key] = value
        return source

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieves a configuration value using dot notation.

        Args:
            key: The key to retrieve, e.g., 'database.neo4j.uri'.
            default: The value to return if the key is not found.

        Returns:
            The configuration value or the default.
        """
        keys = key.split(".")
        value = self.config
        try:
            for k in keys:
                value = value[k]
            return value
        except (TypeError, KeyError):
            return default


# Create a single, global instance of the Config object
# Other modules can simply `from .config import config`
config = Config()
