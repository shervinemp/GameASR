#!/usr/bin/env python3
"""
Module containing the AudioPlayer class.
Part of the TTS components package.

This module provides functionality for playing audio files and raw data.
"""

import atexit
from time import sleep
import numpy as np
import sounddevice as sd

from ..common.utils import get_logger


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

    def __call__(self, audio_data: np.ndarray[np.float32 | np.int16], sample_rate: int):
        return self.play(audio_data, sample_rate)

    def play(self, audio_data: np.ndarray[np.float32 | np.int16], sample_rate: int):
        """Plays raw audio data, passing the explicit device ID to sd.play()."""
        try:
            if audio_data.dtype != np.float32:
                audio_data = (
                    audio_data.astype(np.float32) / 32768.0
                    if audio_data.dtype == np.int16
                    else audio_data.astype(np.float32)
                )
            if np.max(np.abs(audio_data)) > 1.0:
                audio_data /= np.max(np.abs(audio_data))

            sd.play(
                data=audio_data,
                samplerate=(sample_rate if sample_rate else self.sample_rate),
                blocking=False,
                device=self.output_device,
            )
        except Exception as e:
            self.logger.error(f"Error during audio playback: {e}")
            raise

    def stop_all(self):
        """Stops all currently playing audio."""
        self.logger.info("Stopping all audio playback.")
        sd.stop()


if __name__ == "__main__":
    logger = get_logger("AudioPlayerExample")

    player = AudioPlayer()

    logger.info("--- Testing blocking playback with a generated sine wave ---")
    sample_rate = 44100
    frequency = 440  # A4 note
    duration = 2.0  # seconds
    t = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False)
    sine_wave = 0.5 * np.sin(2 * np.pi * frequency * t)

    player.play(sine_wave, sample_rate)
    sleep(duration)

    logger.info("Sine wave playback complete.\n")
