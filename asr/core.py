#!/usr/bin/env python3
"""
Core module for continuous ASR with VAD. Designed to be imported and controlled
by external applications. It handles audio processing, VAD, and asynchronous ASR,
delivering transcribed text via a callback.
"""

import time
import sounddevice as sd
import sys
import threading
import logging
import argparse

from components import AudioStreamer, VADProcessor, ASRService

# Configure logging for internal module messages (can be controlled by external app)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(threadName)s - %(levelname)s - %(message)s",
)

try:
    from silero_vad import load_silero_vad
    from onnx_asr import load_model
except ImportError:
    logging.error(
        "Required ASR/VAD libraries not found. Please install 'silero_vad' and 'onnx_asr'."
    )
    sys.exit(1)


class RealtimeASRSystem:
    """
    Orchestrates audio capture, VAD, and ASR services.
    Provides methods to start and stop the system.
    """

    def __init__(self, args, transcription_callback):
        self.samplerate = 16000  # Fixed for models
        self.vad_chunk_size_samples = 512  # Fixed for Silero VAD at 16kHz

        logging.info("Loading ASR/VAD models...")
        self.vad_model = load_silero_vad()
        self.asr_model = load_model("nemo-parakeet-tdt-0.6b-v2", quantization="int8")
        logging.info("Models loaded.")

        # Initialize components
        self.audio_streamer = AudioStreamer(
            samplerate=self.samplerate,
            channels=1,
            chunk_size=self.vad_chunk_size_samples,
        )
        self.vad_processor = VADProcessor(
            vad_model=self.vad_model,
            samplerate=self.samplerate,
            vad_chunk_size_samples=self.vad_chunk_size_samples,
            vad_threshold=args.vad_threshold,
            end_silence_duration=args.end_silence_duration,
            pre_speech_buffer_duration=args.pre_speech_duration,
        )
        self.asr_service = ASRService(
            asr_model=self.asr_model,
            samplerate=self.samplerate,
            transcription_callback=transcription_callback,  # Pass external callback
            max_queue_size=args.queue_size,
        )

        self._running = False  # Internal flag to control the main loop
        self._main_loop_thread = None  # Reference to the main loop thread

        threading.current_thread().name = "Main_App_Orchestrator"

    def start(self):
        """Starts the real-time ASR processing system."""
        if self._running:
            logging.info("ASR System is already running.")
            return

        self._running = True
        self.asr_service.start()  # Start the ASR worker thread

        # Run the main audio capture loop in a separate thread to allow
        # the calling application to potentially do other things or manage UI.
        self._main_loop_thread = threading.Thread(
            target=self._main_processing_loop, daemon=True
        )
        self._main_loop_thread.start()
        logging.info("Realtime ASR System started.")

    def _main_processing_loop(self):
        """Internal loop for audio capture and VAD processing."""
        while self._running:  # Loop controlled by self._running flag
            logging.info("Listening for speech...")
            try:
                with self.audio_streamer as streamer:
                    while self._running:  # Inner loop also controlled by self._running
                        audio_chunk_int16, overflowed = streamer.read_chunk()

                        if overflowed:
                            logging.warning("Audio input buffer overflowed!")

                        if len(audio_chunk_int16) == 0:
                            logging.info("No audio read. Stream ended unexpectedly.")
                            break  # Break inner loop

                        # Process chunk with VAD, get utterance if speech ends
                        utterance = self.vad_processor.process_audio_chunk(
                            audio_chunk_int16
                        )

                        if utterance is not None:
                            self.asr_service.enqueue_utterance(utterance)
                            logging.info(
                                "Utterance sent to ASR queue. Resetting VAD for next speech..."
                            )
                            # Re-enter outer loop to reset VAD state for the next utterance
                            break
            except sd.PortAudioError as e:
                logging.error(
                    f"Audio device error in main loop: {e}. Attempting to restart stream."
                )
                # Could add a short delay here before retrying stream
                time.sleep(
                    1
                )  # Added a small delay to avoid rapid error loops if mic is disconnected
            except Exception as e:
                logging.exception(
                    f"An unexpected error occurred in main processing loop: {e}"
                )
                self.stop()  # Attempt to stop gracefully on unhandled error
                break  # Exit main loop on critical error

    def stop(self):
        """Stops the real-time ASR processing system gracefully."""
        if not self._running:
            logging.info("ASR System is already stopped.")
            return

        logging.info("Stopping Realtime ASR System...")
        self._running = False  # Signal main loop thread to stop

        # Wait for the main processing loop to finish (optional, depends on UI responsiveness)
        if self._main_loop_thread and self._main_loop_thread.is_alive():
            self._main_loop_thread.join(timeout=5)  # Wait for thread to finish
            if self._main_loop_thread.is_alive():
                logging.warning("Main processing thread did not terminate gracefully.")

        # Handle any pending utterance if app stops during active speech
        final_utterance = self.vad_processor.get_final_utterance_if_active()
        if final_utterance is not None:
            self.asr_service.enqueue_utterance(final_utterance)
            # Give ASR thread a moment to process the last utterance, if any
            self.asr_service.transcription_queue.join(
                timeout=5
            )  # Wait for queue to clear

        self.asr_service.stop()  # Stop the ASR worker gracefully
        logging.info("Realtime ASR System stopped.")


# ==============================================================================
# 5. Argument Parsing (can be used by both module or external app)
# ==============================================================================
def parse_asr_args():
    """Parses command-line arguments for the ASR system."""
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
        default=0.75,
        help="Duration of continuous silence (in seconds) to consider an utterance ended.",
    )
    parser.add_argument(
        "--pre-speech-duration",
        type=float,
        default=0.75,
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
