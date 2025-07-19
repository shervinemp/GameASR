#!/usr/bin/env python3
"""
Module containing the AudioStreamer class.
Part of the ASR components package.
"""

import atexit
import sounddevice as sd
import numpy as np

from ...common.utils import get_logger


class AudioStreamer:
    """
    Captures audio from microphone, explicitly managing the
    input device to prevent conflicts.
    """

    def __init__(
        self, input_device=None, sample_rate=16000, channels=1, chunk_size=512
    ):
        self.logger = get_logger(__name__)

        if input_device is None:
            input_device = sd.default.device[0]

        self.input_device = input_device
        device_name = sd.query_devices(self.input_device)["name"]
        self.logger.info(
            f"AudioStreamer initialized. Using input device: '{device_name}' (ID: {self.input_device})"
        )

        self.input_device = input_device
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.dtype = "int16"
        self._stream = None

        atexit.register(self.stop_all)

    def __enter__(self):
        """Open audio stream on entering context."""
        try:
            self._stream = sd.InputStream(
                device=self.input_device,
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.dtype,
            )
            self._stream.start()
            self.logger.info(f"Audio stream started: Sample Rate={self.sample_rate}Hz")
        except Exception as e:
            self.logger.error(f"Failed to open audio stream: {e}")
            raise
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close audio stream on exiting context."""
        self.stop_all()

    def stop_all(self):
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self.logger.info("Audio stream stopped and closed.")

    def read_chunk(self) -> tuple[np.ndarray, bool]:
        """Reads a chunk of audio from the stream."""
        if not self._stream:
            raise RuntimeError("Audio stream is not open. Use 'with' statement.")
        audio_chunk, overflowed = self._stream.read(self.chunk_size)
        return audio_chunk.flatten(), overflowed
