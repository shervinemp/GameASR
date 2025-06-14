#!/usr/bin/env python3
"""
Core module for continuous ASR with VAD. Designed to be imported and controlled
by external applications. It handles audio processing, VAD, and asynchronous ASR,
delivering transcribed text via a callback.
"""

import time
import sounddevice as sd
import threading

from ..common.logging_utils import get_logger

from .components import AudioStreamer, VADProcessor, ASRService

# Get a logger for this module
logger = get_logger(__name__)


class ASRCore:
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
        transcription_callback: callable = print,
        sound_device: int | None = None,
    ):

        logger.info("Loading ASR/VAD components...")

        self.audio_streamer = AudioStreamer(
            samplerate=sample_rate,
            channels=1,
            chunk_size=vad_chunk_size_samples,
        )
        self.vad_processor = VADProcessor(
            samplerate=sample_rate,
            vad_chunk_size_samples=vad_chunk_size_samples,
            vad_threshold=vad_threshold,
            end_silence_duration=end_silence_duration,
            pre_speech_buffer_duration=pre_speech_duration,
            device=sound_device,
        )
        self.asr_service = ASRService(
            samplerate=sample_rate,
            transcription_callback=transcription_callback,
            max_queue_size=queue_size,
            device=sound_device,
        )

        logger.info("Components initialized.")

        self._running = False  # Internal flag to control the main loop
        self._main_loop_thread = None  # Reference to the main loop thread

        threading.current_thread().name = "Main_App_Orchestrator"

    def start(self):
        """Starts the real-time ASR processing system."""
        if self._running:
            logger.info("ASR Core is already running.")
            return

        self._running = True
        self.asr_service.start()  # Start the ASR worker thread

        # Run the main audio capture loop in a separate thread to allow
        # the calling application to potentially do other things or manage UI.
        self._main_loop_thread = threading.Thread(
            target=self._main_processing_loop, daemon=True
        )
        self._main_loop_thread.start()
        logger.info("ASR Core started.")

    def _main_processing_loop(self):
        """Internal loop for audio capture and VAD processing."""
        while self._running:  # Loop controlled by self._running flag
            try:
                with self.audio_streamer as streamer:
                    logger.info("Listening for speech...")
                    while self._running:  # Inner loop also controlled by self._running
                        audio_chunk_int16, overflowed = streamer.read_chunk()

                        if overflowed:
                            logger.warning("Audio input buffer overflowed!")

                        if len(audio_chunk_int16) == 0:
                            logger.info("No audio read. Stream ended unexpectedly.")
                            continue  # Break inner loop

                        # Process chunk with VAD, get utterance if speech ends
                        utterance = self.vad_processor.process_audio_chunk(
                            audio_chunk_int16
                        )

                        if utterance is not None:
                            self.asr_service.enqueue_utterance(utterance)
                            logger.info("Utterance sent to ASR queue.")
            except sd.PortAudioError as e:
                logger.error(
                    f"Audio device error in main loop: {e}. Attempting to restart stream."
                )
                # Could add a short delay here before retrying stream
                time.sleep(
                    1
                )  # Added a small delay to avoid rapid error loops if mic is disconnected
            except Exception as e:
                logger.exception(
                    f"An unexpected error occurred in main processing loop: {e}"
                )
                self.stop()  # Attempt to stop gracefully on unhandled error
                break  # Exit main loop on critical error

    def stop(self):
        """Stops the real-time ASR processing system gracefully."""
        if not self._running:
            logger.info("ASR Core is already stopped.")
            return

        logger.info("Stopping Realtime ASR Core...")
        self._running = False  # Signal main loop thread to stop

        # Wait for the main processing loop to finish (optional, depends on UI responsiveness)
        if self._main_loop_thread and self._main_loop_thread.is_alive():
            self._main_loop_thread.join(timeout=5)  # Wait for thread to finish
            if self._main_loop_thread.is_alive():
                logger.warning("Main processing thread did not terminate gracefully.")

        # Handle any pending utterance if app stops during active speech
        final_utterance = self.vad_processor.get_final_utterance_if_active()
        if final_utterance is not None:
            self.asr_service.enqueue_utterance(final_utterance)
            # Give ASR thread a moment to process the last utterance, if any
            self.asr_service.transcription_queue.join(
                timeout=5
            )  # Wait for queue to clear

        self.asr_service.stop()  # Stop the ASR worker gracefully
        logger.info("ASR Core stopped.")

    def process_audio(self):
        """
        Start processing audio from microphone input.
        """
        try:
            self.start()
            # Keep the main thread alive while processing in background
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Terminating ASR Core...")
        finally:
            self.stop()
