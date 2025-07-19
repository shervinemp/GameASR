#!/usr/bin/env python3
"""
Core module for continuous ASR with VAD. Designed to be imported and controlled
by external applications. It handles audio processing, VAD, and asynchronous ASR,
delivering transcribed text via a callback.
"""

import time
from typing import Callable
import sounddevice as sd
import threading

from ..common.utils import get_logger

from .components import AudioStreamer, VADProcessor, ASRService


class ASR:
    """
    Orchestrates audio capture, VAD, and ASR services.
    Provides methods to start and stop the system.
    """

    def __init__(
        self,
        vad_threshold: float = 0.3,
        end_silence_duration: float = 0.7,
        pre_speech_duration: float = 1.0,
        vad_chunk_size_samples: int = 512,
        sample_rate: int = 16000,
        queue_size: int = 5,
        transcript_callback: Callable | None = None,
        sound_device: int | None = None,
    ):
        self.logger = get_logger(__name__)

        if transcript_callback is None:
            transcript_callback = self.logger.info

        self.logger.info("Loading ASR/VAD components...")

        self.audio_streamer = AudioStreamer(
            sample_rate=sample_rate,
            channels=1,
            chunk_size=vad_chunk_size_samples,
        )
        self.vad_processor = VADProcessor(
            sample_rate=sample_rate,
            vad_chunk_size_samples=vad_chunk_size_samples,
            vad_threshold=vad_threshold,
            end_silence_duration=end_silence_duration,
            pre_speech_buffer_duration=pre_speech_duration,
            device=sound_device,
        )
        self.asr_service = ASRService(
            sample_rate=sample_rate,
            queue_size=queue_size,
            transcript_callback=transcript_callback,
        )

        self.logger.info("Components initialized.")

        self._running = False
        self._main_loop_thread = None

        threading.current_thread().name = "Main_App_Orchestrator"

    def start(self):
        """Starts the real-time ASR processing system."""
        if self._running:
            self.logger.info("ASR Core is already running.")
            return

        self._running = True
        self.asr_service.start()

        self._main_loop_thread = threading.Thread(
            target=self._main_processing_loop, daemon=True
        )
        self._main_loop_thread.start()
        self.logger.info("ASR Core started.")

    def _main_processing_loop(self):
        """Internal loop for audio capture and VAD processing."""
        while self._running:
            try:
                with self.audio_streamer as streamer:
                    self.logger.info("Listening for speech...")
                    while self._running:
                        audio_chunk_int16, overflowed = streamer.read_chunk()

                        if overflowed:
                            self.logger.warning("Audio input buffer overflowed!")

                        if len(audio_chunk_int16) == 0:
                            self.logger.info(
                                "No audio read. Stream ended unexpectedly."
                            )
                            continue

                        utterance = self.vad_processor.process_audio_chunk(
                            audio_chunk_int16
                        )

                        if utterance is not None:
                            self.asr_service.enqueue_utterance(utterance)
                            self.logger.info("Utterance sent to ASR queue.")
            except sd.PortAudioError as e:
                self.logger.error(
                    f"Audio device error in main loop: {e}. Attempting to restart stream."
                )
                time.sleep(
                    1
                )  # Added a small delay to avoid rapid error loops if mic is disconnected
            except Exception as e:
                self.logger.exception(
                    f"An unexpected error occurred in main processing loop: {e}"
                )
                self.stop()
                break

    def stop(self):
        """Stops the real-time ASR processing system gracefully."""
        if not self._running:
            self.logger.info("ASR Core is already stopped.")
            return

        self.logger.info("Stopping Realtime ASR Core...")
        self._running = False

        if self._main_loop_thread and self._main_loop_thread.is_alive():
            self._main_loop_thread.join(timeout=5)
            if self._main_loop_thread.is_alive():
                self.logger.warning(
                    "Main processing thread did not terminate gracefully."
                )

        final_utterance = self.vad_processor.get_final_utterance_if_active()
        if final_utterance is not None:
            self.asr_service.enqueue_utterance(final_utterance)
            self.asr_service.transcript_queue.join(timeout=5)

        self.asr_service.stop()
        self.logger.info("ASR Core stopped.")

    def process_audio(self):
        """
        Start processing audio from microphone input.
        """
        try:
            self.start()
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.logger.info("Terminating ASR Core...")
        finally:
            self.stop()
