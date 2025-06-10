"""
ASR Components Package

This subpackage contains the core components of the ASR system:
- AudioStreamer: For capturing audio from microphones
- VADProcessor: Voice Activity Detection for segmenting speech
- ASRService: Manages transcription services and worker threads

These components can be used independently or together within the RealtimeASRSystem.
"""

from .audio_streamer import AudioStreamer
from .vad_processor import VADProcessor
from .asr_service import ASRService

__all__ = ["AudioStreamer", "VADProcessor", "ASRService"]
