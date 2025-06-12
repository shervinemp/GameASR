#!/usr/bin/env python3
"""
Core module for Text-to-Speech (TTS) service.

This module provides high-level functions and classes to manage the TTS pipeline.
"""

from ..common.logging_utils import get_logger, setup_logging

from .components.tts_service import TTSService

# Get a logger for this module
logger = get_logger(__name__)


class TTSCore:
    """
    Core class for managing the TTS pipeline.

    This class encapsulates all components of the TTS system and provides a unified interface.
    """

    def __init__(self):
        """Initialize TTS core with default model path."""
        try:
            # Initialize the main TTS service component
            self.tts_service = TTSService()

            logger.debug("TTSCore initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize TTSCore: {e}")
            raise

    def speak(self, text):
        """
        Convert text into speech and play it.

        Args:
            text: The text string to synthesize
        """
        try:
            # Use the TTS service to process and play audio
            self.tts_service.speak(text)
        except Exception as e:
            logger.error(f"Error in TTSCore.speak(): {e}")

    def speak_file(self, text, output_path):
        """
        Convert text into speech and save it to a file.

        Args:
            text: The text string to synthesize
            output_path: Path where the audio file will be saved
        """
        try:
            # Use the TTS service to process and save audio
            self.tts_service.speak_file(text, output_path)
        except Exception as e:
            logger.error(f"Error in TTSCore.speak_file(): {e}")

    def batch_process(self, texts, output_dir):
        """
        Process multiple texts into speech files.

        Args:
            texts: List of text strings to synthesize
            output_dir: Directory where audio files will be saved

        Returns:
            list: List of file paths for the generated audio files
        """
        try:
            # Initialize result container
            output_files = []

            # Process each text in the batch
            logger.info(f"Processing {len(texts)} texts to speech...")

            for i, text in enumerate(texts):
                output_path = f"{output_dir}/output_{i}.wav"
                self.tts_service.speak_file(text, output_path)
                output_files.append(output_path)

            return output_files
        except Exception as e:
            logger.error(f"Error in TTSCore.batch_process(): {e}")
            raise


# Example usage (can be moved to a separate script if needed)
if __name__ == "__main__":
    setup_logging(log_level="DEBUG")

    try:
        # Create TTSCore instance with the TTS model path
        tts_core = TTSCore()

        # Example: Convert text to speech and play it
        sample_text = "Hello, this is a test of the text-to-speech system."
        logger.info("Playing sample audio...")
        tts_core.speak(sample_text)

        # Example: Save text to speech output as file
        output_file = "sample_output.wav"
        tts_core.speak_file(sample_text, output_file)
        logger.info(f"TTS output saved to: {output_file}")

    except Exception as e:
        logger.error(f"Error in TTS core example: {e}")
