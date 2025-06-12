#!/usr/bin/env python3
"""
Module containing the TTSService class.
Part of the TTS components package.

This module provides text-to-speech (TTS) services using a sequence of processors.
"""

import logging

from ...common.base_component import BaseComponent
from ...common.logging_utils import get_logger

from .tts_processor import TTSProcessor
from .audio_player import AudioPlayer

# Get a logger for this module
logger = get_logger(__name__)


class TTSService(BaseComponent):
    """
    Manages the complete TTS pipeline: from text input to audio output.

    Inherits start/stop methods from BaseComponent.
    """

    def __init__(self):
        super().__init__()

        try:
            # Initialize TTS processor and audio player components
            self.tts_processor = TTSProcessor()
            self.audio_player = AudioPlayer()

            logging.debug("TTSService initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize TTSService: {e}")
            raise

    def speak(self, text):
        """
        Convert text into speech and play the audio.

        Args:
            text: The text string to synthesize
        """
        try:
            # Process the text into audio samples
            audio_samples, sample_rate = self.tts_processor.process_text(text)

            # Play the generated audio
            self.audio_player.play_audio(audio_samples, sample_rate)
        except Exception as e:
            logging.error(f"Error in TTS processing: {e}")
            raise

    def speak_file(self, text, output_path):
        """
        Convert text into speech and save to a file.

        Args:
            text: The text string to synthesize
            output_path: Path where the audio file will be saved
        """
        try:
            # Process the text into audio samples
            audio_samples, sample_rate = self.tts_processor.process_text(text)

            # Save the generated audio to a file (requires additional processing)
            import soundfile as sf

            sf.write(output_path, audio_samples, sample_rate)
        except Exception as e:
            logging.error(f"Error saving TTS output: {e}")
            raise
