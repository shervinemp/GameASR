"""
Voice Control Bridge Module

This module provides a JSON-RPC interface for integrating with the voice control pipeline.
"""

# Allow direct imports from bridge
from .server import run_server, VoiceControlService

__all__ = ["run_server", "VoiceControlService"]
