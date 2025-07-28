#!/usr/bin/env python3
"""
Module containing the TTSProcessor class.
Part of the TTS components package.

This module provides text-to-speech processing functionality using ONNX models.
"""

import os
from kokoro_onnx import Kokoro
from kokoro_onnx.tokenizer import Tokenizer

from .audio import AudioPlayer

from ..common.utils import download_file, get_logger


class TTS:
    sample_rate: int = 24_000
    local_dir: str = os.path.join("model_files", "tts")

    def __init__(self):
        """
        Initialize TTS processor with the specified model and configuration files.
        """
        self.logger = get_logger(__name__)

        self.kokoro = Kokoro(
            model_path=os.path.join(self.local_dir, "kokoro-v1.0.onnx"),
            voices_path=os.path.join(self.local_dir, "voices-v1.0.bin"),
        )
        self.tokenizer = Tokenizer()
        self.audio_player = AudioPlayer()

    @classmethod
    def download(cls):
        os.makedirs(cls.local_dir, exist_ok=True)

        required_files = [
            "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
            "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
        ]

        for url in required_files:
            filename = url.split("/")[-1]
            destination = os.path.join(cls.local_dir, filename)
            if not os.path.exists(destination):
                download_file(url, destination)

    def __call__(
        self,
        text: str,
        voice: str = "af_heart",
        language: str = "en-us",
        speed: float = 1.0,
        interrupt: bool = False,
    ):
        """
        Convert text into speech audio.

        Args:
            text: The text string to synthesize

        Returns:
            tuple: (numpy array of audio samples, sample rate)
        """
        phonemes = self.tokenizer.phonemize(text, lang=language)
        samples, sample_rate = self.kokoro.create(
            phonemes,
            voice=voice,
            speed=speed,
            is_phonemes=True,
        )

        self.audio_player(samples, sample_rate, interrupt)

        return samples, sample_rate

    def start(self):
        self.audio_player.start()

    def stop(self):
        self.audio_player.stop()

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
