#!/usr/bin/env python3
"""
Module containing the TTSProcessor class.
Part of the TTS components package.

This module provides text-to-speech processing functionality using ONNX models.
"""

from kokoro_onnx import Kokoro
from kokoro_onnx.tokenizer import Tokenizer

from ...common.utils import get_logger


class TTSProcessor:
    """
    Processes text to generate speech audio.
    Inherits start/stop methods from BaseComponent.

    Uses an ONNX-based model for TTS.
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

    def process_text(self, text, voice="af_heart", language="en-us"):
        """
        Convert text into speech audio.

        Args:
            text: The text string to synthesize

        Returns:
            tuple: (numpy array of audio samples, sample rate)
        """
        try:
            # Preprocess the input text
            phonemes = self.tokenizer.phonemize(text, lang=language)

            samples, sample_rate = self.kokoro.create(
                phonemes, voice=voice, speed=1.0, is_phonemes=True
            )

            return samples, sample_rate
        except Exception as e:
            self.logger.error(f"Error processing text: {e}")
            raise
