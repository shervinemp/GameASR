#!/usr/bin/env python3
"""
Module containing the TTSProcessor class.
Part of the TTS components package.

This module provides text-to-speech processing functionality using ONNX models.
"""

from kokoro_onnx import Kokoro
from kokoro_onnx.tokenizer import Tokenizer

from .player import AudioPlayer

from ..common.utils import get_logger


class TTS:
    """
    Processes text to generate speech audio.
    Inherits start/stop methods from BaseComponent.

    Uses Kokoro for TTS.
    """

    def __init__(self):
        """
        Initialize TTS processor with the specified model and configuration files.
        """
        self.logger = get_logger(__name__)

        self.kokoro = Kokoro(
            model_path="models/tts/kokoro-v1.0.onnx",
            voices_path="models/tts/voices-v1.0.bin",
        )
        self.tokenizer = Tokenizer()
        self.audio_player = AudioPlayer()

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

            self.audio_player.play_audio(samples, sample_rate)

        except Exception as e:
            self.logger.error(f"Error speaking text: {e}")
            raise

        return samples, sample_rate
