#!/usr/bin/env python3
"""
Module containing the AudioStreamer class.
Part of the ASR components package.
"""

import sounddevice as sd
import numpy as np

from ...common.utils import get_logger


class AudioStreamer:
    """
    Captures audio from microphone input using sounddevice library.
    Inherits start/stop methods from BaseComponent.

    Provides methods to read chunks of audio data.
    """

    def __init__(self, samplerate=16000, channels=1, chunk_size=512):
        """Initialize with default parameters for ASR."""
        self.logger = get_logger(__name__)

        self.samplerate = samplerate
        self.channels = channels
        self.chunk_size = chunk_size
        self.dtype = "int16"  # Fixed for microphone input common to ASR
        self._stream = None

    def __enter__(self):
        """Open audio stream on entering context."""
        try:
            self._stream = sd.InputStream(
                samplerate=self.samplerate, channels=self.channels, dtype=self.dtype
            )
            self._stream.start()

            # Use our standard logger instead of the direct logging module
            self.logger.info(
                f"Audio stream started: Sample Rate={self.samplerate}Hz, Chunk Size={self.chunk_size}"
            )
        except Exception as e:
            # Use our standard logger instead of the direct logging module
            self.logger.error(f"Failed to open audio stream: {e}")
            raise

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close audio stream on exiting context."""
        if self._stream:
            self._stream.stop()
            self._stream.close()

            # Use our standard logger instead of the direct logging module
            self.logger.info("Audio stream stopped and closed.")

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
            # Use our standard logger instead of the direct logging module
            self.logger.warning(
                f"Audio chunk size mismatch. Expected {self.chunk_size}, "
                f"got {len(audio_chunk_int16)}. Data may be incomplete."
            )
        return audio_chunk_int16, overflowed
