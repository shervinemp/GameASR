#!/usr/bin/env python3
"""
Module containing the TTSService class.
Part of the TTS components package.

This module provides text-to-speech (TTS) services using a sequence of processors.
"""
from ..common.utils import get_logger

from .components.tts_processor import TTSProcessor
from .components.audio_player import AudioPlayer

# Get a logger for this module
logger = get_logger(__name__)


class TTS:
    """
    Manages the complete TTS pipeline: from text input to audio output.

    Inherits start/stop methods from BaseComponent.
    """

    def __init__(self):
        self.tts_processor = TTSProcessor()
        self.audio_player = AudioPlayer()

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
            logger.error(f"Error in TTS processing: {e}")
            raise
