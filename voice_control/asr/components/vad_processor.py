#!/usr/bin/env python3
"""
Module containing the VADProcessor class.
Part of the ASR components package.

This module provides voice activity detection (VAD) functionality using Silero VAD.
"""

import numpy as np
from collections import deque

import torch

try:
    from silero_vad import load_silero_vad
except ImportError:
    raise ImportError(
        "Silero VAD is not installed. Please install it using: pip install silero-vad"
    )

from ...common.utils import get_logger


class VADProcessor:
    """
    Manages VAD state and logic to segment continuous audio into utterances.
    Yields complete utterances when speech ends.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        vad_chunk_size_samples: int = 512,
        vad_threshold: float = 0.3,
        end_silence_duration: float = 0.7,
        pre_speech_buffer_duration: float = 1.0,
        device=None,
    ):
        self.logger = get_logger(__name__)

        self.vad_model = load_silero_vad()
        self.logger.info("VAD model loaded.")
        if device is not None:
            self.vad_model.to(device)

        self.sample_rate = sample_rate
        self.vad_chunk_size_samples = vad_chunk_size_samples
        self.vad_threshold = vad_threshold
        self.end_silence_duration = end_silence_duration
        self.pre_speech_buffer_duration = pre_speech_buffer_duration

        self.vad_chunk_duration = self.vad_chunk_size_samples / self.sample_rate
        self.SILENCE_THRESHOLD_CHUNKS = int(
            self.end_silence_duration / self.vad_chunk_duration
        )

        self.is_speech_active = False
        self.silence_counter = 0
        self.current_utterance_audio_chunks = []

        pre_speech_buffer_size = int(self.pre_speech_buffer_duration * self.sample_rate)
        self.pre_speech_buffer_deque = deque(maxlen=pre_speech_buffer_size)

        # Initialize deque with zeros to fill its maxlen
        for _ in range(pre_speech_buffer_size):
            self.pre_speech_buffer_deque.append(0.0)

    def process_audio_chunk(self, audio_chunk_int16: np.ndarray) -> np.ndarray | None:
        """
        Processes an incoming audio chunk for VAD.
        Returns a complete, trimmed utterance (NumPy array) if speech ends, otherwise None.
        """
        audio_chunk_float32 = (
            audio_chunk_int16.astype(np.float32) / np.iinfo(np.int16).max
        )

        # Update pre-speech buffer
        for sample in audio_chunk_float32:
            self.pre_speech_buffer_deque.append(sample)

        audio_tensor = torch.tensor(audio_chunk_float32)
        speech_prob = self.vad_model(audio_tensor, self.sample_rate)

        utterance_to_return = None

        if not self.is_speech_active:
            if speech_prob > self.vad_threshold:
                self.logger.info("Speech detected! Accumulating utterance...")
                self.is_speech_active = True
                self.current_utterance_audio_chunks.append(
                    np.array(self.pre_speech_buffer_deque, dtype=np.float32)
                )
                self.current_utterance_audio_chunks.append(audio_chunk_float32)
                self.silence_counter = 0
        else:
            self.current_utterance_audio_chunks.append(audio_chunk_float32)

            if speech_prob > self.vad_threshold:
                self.silence_counter = 0
            else:
                self.silence_counter += 1

                if self.silence_counter >= self.SILENCE_THRESHOLD_CHUNKS:
                    self.logger.info("End of speech detected.")
                    utterance_to_return = self._finalize_utterance()

        return utterance_to_return

    def _finalize_utterance(self) -> np.ndarray | None:
        """Helper to concatenate and trim the current utterance."""
        self.is_speech_active = False
        self.silence_counter = 0

        if self.current_utterance_audio_chunks:
            full_utterance_audio = np.concatenate(self.current_utterance_audio_chunks)
            self.current_utterance_audio_chunks = []

            trim_samples = int(self.end_silence_duration * self.sample_rate)
            trimmed_utterance_audio = (
                full_utterance_audio[:-trim_samples]
                if len(full_utterance_audio) > trim_samples
                else full_utterance_audio
            )

            if trimmed_utterance_audio.size > 10:
                return trimmed_utterance_audio
            else:
                self.logger.info(
                    "Speech segment was too short or contained only silence after trimming. Not returning."
                )
                return None
        else:
            self.logger.info(
                "No audio accumulated for transcription (empty utterance)."
            )
            return None

    def get_final_utterance_if_active(self) -> np.ndarray | None:
        """
        Call this when the audio stream ends to get any pending active utterance.
        """
        if self.is_speech_active and self.current_utterance_audio_chunks:
            self.logger.info("Stream ended during active speech. Finalizing segment...")

            # No trimming of trailing silence here, as the stream just ended.
            full_utterance_audio = np.concatenate(self.current_utterance_audio_chunks)
            self.current_utterance_audio_chunks = []
            self.is_speech_active = False
            self.silence_counter = 0
            if full_utterance_audio.size > 0:
                return full_utterance_audio
            else:
                self.logger.debug("Final speech segment was empty.")
                return None
        return None
