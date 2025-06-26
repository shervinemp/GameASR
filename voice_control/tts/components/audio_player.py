#!/usr/bin/env python3
"""
Module containing the AudioPlayer class.
Part of the TTS components package.

This module provides functionality for playing audio files and raw data.
"""

import atexit
import numpy as np
import sounddevice as sd

from ...common.utils import get_logger


class AudioPlayer:
    """
    Plays audio using the `sounddevice` library, explicitly managing the
    output device to prevent conflicts.
    """

    def __init__(self, output_device=None):
        self.logger = get_logger(__name__)

        if output_device is None:
            output_device = sd.default.device[1]

        self.output_device = output_device
        device_name = sd.query_devices(self.output_device)["name"]
        self.logger.info(
            f"AudioPlayer initialized. Using output device: '{device_name}' (ID: {self.output_device})"
        )

        atexit.register(self.stop_all)

    def stop_all(self):
        """Stops all currently playing audio."""
        self.logger.info("Stopping all audio playback.")
        sd.stop()

    def play_audio(self, audio_data, sample_rate=24000, wait_done=True):
        """Plays raw audio data, passing the explicit device ID to sd.play()."""
        try:
            # (Normalization and data type checks from previous version)
            if not isinstance(audio_data, np.ndarray):
                raise TypeError("audio_data must be a NumPy array.")
            if audio_data.dtype != np.float32:
                audio_data = (
                    audio_data.astype(np.float32) / 32768.0
                    if audio_data.dtype == np.int16
                    else audio_data.astype(np.float32)
                )
            if np.max(np.abs(audio_data)) > 1.0:
                audio_data /= np.max(np.abs(audio_data))

            sd.play(
                audio_data, sample_rate, blocking=wait_done, device=self.output_device
            )
        except Exception as e:
            self.logger.error(f"Error during audio playback: {e}")
            raise


if __name__ == "__main__":
    # This is an example of how to use the AudioPlayer class
    # To run this, you'll need a WAV file named 'test.wav' in the same directory.

    logger = get_logger("AudioPlayerExample")
    player = AudioPlayer()

    # --- Example 1: Playing a sine wave (blocking) ---
    logger.info("--- Testing blocking playback with a generated sine wave ---")
    sample_rate = 44100
    frequency = 440  # A4 note
    duration = 2.0  # seconds
    t = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False)
    sine_wave = 0.5 * np.sin(2 * np.pi * frequency * t)

    player.play_audio(sine_wave, sample_rate=sample_rate, wait_done=True)
    logger.info("Sine wave playback complete.\n")
