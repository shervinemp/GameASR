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
    local_dir: str = "model_files/tts"

    def __init__(self):
        """
        Initialize TTS processor with the specified model and configuration files.
        """
        self.logger = get_logger(__name__)

        self.kokoro = Kokoro(
            model_path=f"{self.local_dir}/kokoro-v1.0.onnx",
            voices_path=f"{self.local_dir}/voices-v1.0.bin",
        )
        self.tokenizer = Tokenizer()
        self.audio_player = AudioPlayer()

    def download(self):
        os.makedirs(self.local_dir, exist_ok=True)

        required_files = [
            "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
            "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
        ]

        for url in required_files.items():
            filename = url.split("/")[-1]
            destination = os.path.join(self.local_dir, filename)
            if not os.path.exists(destination):
                download_file(url, destination)
            else:
                self.logger.info(
                    f"File {destination} already exists, skipping download."
                )

        self.logger.info("TTS setup completed successfully.")

    def __call__(
        self,
        text: str,
        voice: str = "af_heart",
        language: str = "en-us",
        speed: float = 1.0,
    ):
        """
        Convert text into speech audio.

        Args:
            text: The text string to synthesize

        Returns:
            tuple: (numpy array of audio samples, sample rate)
        """
        try:
            phonemes = self.tokenizer.phonemize(text, lang=language)
            samples, sample_rate = self.kokoro.create(
                phonemes,
                voice=voice,
                speed=speed,
                is_phonemes=True,
            )

            self.audio_player.play(samples, sample_rate)

        except Exception as e:
            self.logger.error(f"Error speaking text: {e}")
            raise

        return samples, sample_rate
