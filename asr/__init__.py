"""
ASR Module

This module provides real-time automatic speech recognition (ASR) with voice activity detection (VAD).
It includes several components that can be used independently or together to build an ASR system.
"""

from .components import AudioStreamer, VADProcessor, ASRService
from .core import RealtimeASRSystem, parse_asr_args

__all__ = [
    "AudioStreamer",
    "VADProcessor",
    "ASRService",
    "RealtimeASRSystem",
    "parse_asr_args",
]
