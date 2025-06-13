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


class RealtimeASRSystem:
    """
    Orchestrates audio capture, VAD, and ASR services.
    Provides methods to start and stop the system.
    """

    def __init__(self, args, transcription_callback):
        self.samplerate = 16000  # Fixed for models
        self.vad_chunk_size_samples = 512  # Fixed for Silero VAD at 16kHz

        logger.info("Loading ASR/VAD components...")

        # Initialize components with parameters from args
        self.audio_streamer = AudioStreamer(
            samplerate=self.samplerate,
            channels=1,
            chunk_size=self.vad_chunk_size_samples,
        )
        self.vad_processor = VADProcessor(
            samplerate=self.samplerate,
            vad_chunk_size_samples=self.vad_chunk_size_samples,
            vad_threshold=args.vad_threshold,
            end_silence_duration=args.end_silence_duration,
            pre_speech_buffer_duration=args.pre_speech_duration,
        )
        self.asr_service = ASRService(
            samplerate=self.samplerate,
            transcription_callback=transcription_callback,  # Pass external callback
            max_queue_size=args.queue_size,
        )

        logger.info("Components initialized.")

        self._running = False  # Internal flag to control the main loop
        self._main_loop_thread = None  # Reference to the main loop thread

        threading.current_thread().name = "Main_App_Orchestrator"

    def start(self):
        """Starts the real-time ASR processing system."""
        if self._running:
            logger.info("ASR System is already running.")
            return

        self._running = True
        self.asr_service.start()  # Start the ASR worker thread

        # Run the main audio capture loop in a separate thread to allow
        # the calling application to potentially do other things or manage UI.
        self._main_loop_thread = threading.Thread(
            target=self._main_processing_loop, daemon=True
        )
        self._main_loop_thread.start()
        logger.info("Realtime ASR System started.")

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
            logger.info("ASR System is already stopped.")
            return

        logger.info("Stopping Realtime ASR System...")
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
        logger.info("Realtime ASR System stopped.")


class ASRCore:
    """
    Simplified interface for running ASR from microphone input.

    This class is a wrapper around RealtimeASRSystem that handles default parameter values and
    provides a simple process_audio() method for easy use in scripts.
    """

    def __init__(self):
        self.args = parse_asr_args()

    def process_audio(self, transcription_callback=None):
        """
        Start processing audio from microphone input.

        Args:
            transcription_callback: Optional callback function to receive transcriptions.
                                   If not provided, will use a default no-op callback.
        """
        if transcription_callback is None:
            # Default no-op callback if none is provided
            def transcription_callback(transcription):
                logger.info(f"Transcription received: {transcription}")

        asr_system = RealtimeASRSystem(self.args, transcription_callback)
        try:
            asr_system.start()
            # Keep the main thread alive while processing in background
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Terminating ASR system...")
        finally:
            asr_system.stop()


# ==============================================================================
# Argument Parsing (can be used by both module or external app)
# ==============================================================================
def parse_asr_args():
    """Parses command-line arguments for the ASR system."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Real-time, continuous ASR from microphone using ONNX models and VAD."
    )

    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=0.3,
        help="Probability threshold for Voice Activity Detection (0.0-1.0). Higher values are less sensitive to noise.",
    )
    parser.add_argument(
        "--end-silence-duration",
        type=float,
        default=0.7,
        help="Duration of continuous silence (in seconds) to consider an utterance ended.",
    )
    parser.add_argument(
        "--pre-speech-duration",
        type=float,
        default=0.8,
        help="Duration of audio (in seconds) to include before detected speech starts, for context.",
    )
    parser.add_argument(
        "--queue-size",
        type=int,
        default=5,
        help="Maximum number of utterances to queue for ASR processing. Prevents memory overload.",
    )
    parser.add_argument(
        "--device",
        type=int,
        help="Specific audio input device ID. Use `python -m sounddevice` to list devices.",
    )
    return parser.parse_args()
