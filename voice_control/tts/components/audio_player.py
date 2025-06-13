#!/usr/bin/env python3
"""
Module containing the AudioPlayer class.
Part of the TTS components package.

This module provides functionality for playing audio files.
"""

import numpy as np
import simpleaudio as sa

from ...common.base_component import BaseComponent
from ...common.logging_utils import get_logger

# Get a logger for this module
logger = get_logger(__name__)


class AudioPlayer(BaseComponent):
    """
    Plays audio using simpleaudio library.
    Inherits start/stop methods from BaseComponent.

    Can play raw audio data or file paths.
    """

    def __init__(self, device=None, buffer_size=2048):
        """Initialize with default parameters for TTS."""
        super().__init__()
        self.device = device
        self.buffer_size = buffer_size

    def play_audio(self, audio_data, sample_rate=24000):
        """
        Play raw audio data.

        Args:
            audio_data: NumPy array or bytes of audio samples
            sample_rate: Sample rate in Hz (default: 24000)

        Returns:
            simpleaudio.PlayObject: The play object for controlling playback
        """
        try:
            if isinstance(audio_data, np.ndarray) and audio_data.dtype != np.int16:
                if audio_data.dtype != np.int16:
                    audio_data = audio_data / np.max(
                        np.abs(audio_data)
                    )  # Normalize to -1.0 to 1.0
                    audio_data = (audio_data * 32767).astype(
                        np.int16
                    )  # Scale to 16-bit range

                audio_bytes = audio_data.tobytes()
            else:
                audio_bytes = audio_data

            play_obj = sa.play_buffer(
                audio_bytes,
                num_channels=1,
                bytes_per_sample=2,  # 16-bit PCM (2 bytes)
                sample_rate=sample_rate,
            )
            logger.debug("Audio playback started.")
            return play_obj
        except Exception as e:
            logger.error(f"Error during audio playback: {e}")
            raise

    def play_file(self, file_path):
        """
        Play an audio file.

        Args:
            file_path: Path to the audio file

        Returns:
            simpleaudio.PlayObject: The play object for controlling playback
        """
        try:
            # Create a play object and return it for control
            play_obj = sa.play_file(file_path)
            logger.debug(f"Playing audio file: {file_path}")
            return play_obj
        except Exception as e:
            logger.error(f"Error playing audio file: {e}")
            raise
