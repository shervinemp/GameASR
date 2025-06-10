#!/usr/bin/env python3
"""
Module containing the AudioStreamer class.
Part of the ASR components package.
"""

import logging
import sounddevice as sd
import numpy as np


class AudioStreamer:
    """Manages audio input from the microphone using sounddevice."""

    def __init__(self, samplerate: int, channels: int, chunk_size: int):
        self.samplerate = samplerate
        self.channels = channels
        self.chunk_size = chunk_size
        self.dtype = "int16"  # Fixed for microphone input common to ASR
        self._stream = None

    def __enter__(self):
        """Context manager entry point: opens the audio stream."""
        try:
            self._stream = sd.InputStream(
                samplerate=self.samplerate, channels=self.channels, dtype=self.dtype
            )
            self._stream.start()
            logging.info(
                f"Audio stream started: Sample Rate={self.samplerate}Hz, Chunk Size={self.chunk_size}"
            )
            return self
        except Exception as e:
            logging.error(f"Failed to open audio stream: {e}")
            raise  # Re-raise to be caught by the main application's error handling

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point: stops and closes the audio stream."""
        if self._stream:
            self._stream.stop()
            self._stream.close()
            logging.info("Audio stream stopped and closed.")

    def read_chunk(self) -> tuple[np.ndarray, bool]:
        """
        Reads a chunk of audio from the stream.
        Returns (audio_chunk_int16, overflowed)
        """
        if not self._stream:
            raise RuntimeError("Audio stream is not open. Call __enter__() first.")

        audio_chunk_int16, overflowed = self._stream.read(self.chunk_size)
        audio_chunk_int16 = audio_chunk_int16.flatten()  # Ensure 1D array

        if len(audio_chunk_int16) != self.chunk_size:
            logging.warning(
                f"Audio chunk size mismatch. Expected {self.chunk_size}, "
                f"got {len(audio_chunk_int16)}. Data may be incomplete."
            )
        return audio_chunk_int16, overflowed
