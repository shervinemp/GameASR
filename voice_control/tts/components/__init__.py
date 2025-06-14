"""
Package initialization for TTS (Text-to-Speech) components.

This module exposes all the important classes from the TTS components package.
"""

from .audio_player import AudioPlayer
from .tts_processor import TTSProcessor

__all__ = ["AudioPlayer", "TTSProcessor"]
