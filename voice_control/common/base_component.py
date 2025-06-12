#!/usr/bin/env python3
"""
Module containing the BaseComponent class.

This module provides a common base for all components in both ASR and TTS packages.
"""

from .logging_utils import get_logger


class BaseComponent:
    """
    Base class for all components with start/stop methods.

    All components inherit from this class to ensure consistent interface.
    """

    def __init__(self):
        """Initialize the component."""
        self.running = False
        logger.debug(f"{self.__class__.__name__} initialized.")

    def start(self):
        """
        Start the component's processing loop.

        This method should be overridden in subclasses to implement actual functionality.
        """
        if not self.running:
            try:
                self.running = True
                logger.info(f"Starting {self.__class__.__name__}")
                # Subclasses should implement their own processing logic here
            except Exception as e:
                logger.error(f"Error starting {self.__class__.__name__}: {e}")
                raise

    def stop(self):
        """
        Stop the component's processing loop.

        This method should be overridden in subclasses to clean up resources.
        """
        if self.running:
            try:
                self.running = False
                logger.info(f"Stopping {self.__class__.__name__}")
                # Subclasses should implement their own cleanup logic here
            except Exception as e:
                logger.error(f"Error stopping {self.__class__.__name__}: {e}")
                raise

    def __enter__(self):
        """Context manager entry - starts the component."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Context manager exit - stops the component."""
        self.stop()


# Get module logger using our standard logging utility
logger = get_logger(__name__)

# Example usage (can be removed in production)
if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.DEBUG)

    # Create a dummy component to demonstrate functionality
    class DummyComponent(BaseComponent):
        def start(self):
            super().start()
            logger.info("Dummy component started")

        def stop(self):
            logger.info("Dummy component stopped")
            super().stop()

    # Test the component
    try:
        dummy = DummyComponent()
        with dummy:
            logger.info("Component is running in context")
    except Exception as e:
        logger.error(f"Error in test: {e}")
