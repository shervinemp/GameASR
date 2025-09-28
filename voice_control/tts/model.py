#!/usr/bin/env python3
"""
Module containing the TTSProcessor class.
Part of the TTS components package.

This module provides text-to-speech processing functionality using ONNX models.
"""

from .audio import AudioPlayer
from ..common.utils import get_logger
from ..common.config import config


class TTS:
    def __init__(self):
        self.logger = get_logger(__name__)
        self.audio_player = AudioPlayer()

    @classmethod
    def download(cls):
        pass

    def __call__(
        self,
        text: str,
        voice: str = "af_heart",
        language: str = "en-us",
        speed: float = 1.0,
        interrupt: bool = False,
    ):
        self.logger.warning("TTS is not implemented. Skipping audio playback.")
        return None, None

    def start(self):
        self.audio_player.start()

    def stop(self):
        self.audio_player.stop()

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()