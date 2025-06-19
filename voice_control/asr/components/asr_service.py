#!/usr/bin/env python3
"""
Module containing the ASRService class.
Part of the ASR components package.

This module provides automatic speech recognition (ASR) functionality using the
ParakeetV2 model from ONNX-ASR.
"""

import queue
import threading

from onnx_asr import load_model

from ...common.utils import get_logger


class ASRService:
    """
    Manages the ASR model and a dedicated worker thread for asynchronous transcription.
    Delivers results via a callback function.
    """

    def __init__(
        self,
        samplerate: int = 16000,
        max_queue_size: int = 5,
        transcript_callback: callable | None = None,
    ):
        self.logger = get_logger(__name__)

        if transcript_callback is None:
            transcript_callback = self.logger.info

        # Load the ONNX-ASR model
        self.model = load_model("nemo-parakeet-tdt-0.6b-v2", quantization="int8")
        self.logger.info("ASR model loaded.")

        self.samplerate = samplerate
        self.transcript_callback = transcript_callback  # Callback for results
        self.transcript_queue = queue.Queue(maxsize=max_queue_size)
        self._worker_thread = None

    def start(self):
        """Starts the ASR worker thread."""
        if not self._worker_thread:
            self._worker_thread = threading.Thread(
                target=self._worker_loop,
                daemon=True,  # Daemon thread exits when main program exits
            )
            self._worker_thread.start()
            self.logger.info("ASR worker thread started.")

    def _worker_loop(self):  # No arguments here!
        """The main loop executed by the ASR worker thread."""
        threading.current_thread().name = "ASR_Worker"
        while True:
            try:
                audio_segment = self.transcript_queue.get(
                    timeout=0.1
                )  # Short timeout to check for shutdown

                if audio_segment is None:  # Sentinel for shutdown
                    self.logger.info("Shutdown signal received. Exiting.")
                    break

                self.logger.debug("Processing transcription...")
                transcription = self.model.recognize(
                    audio_segment, sample_rate=self.samplerate
                ).strip()
                self.logger.debug("Transcription: " + transcription)

                if len(transcription):
                    self.transcript_callback(transcription)  # Call instance callback
                else:
                    self.logger.debug("No transcription generated for this segment.")

                self.transcript_queue.task_done()

            except queue.Empty:
                continue  # Keep trying if queue is empty
            except Exception as e:
                self.logger.error(
                    f"An unexpected error occurred during transcription: {e}"
                )
                if audio_segment is not None:
                    self.transcript_queue.task_done()

    def enqueue_utterance(self, audio_segment):
        """Adds an audio segment to the queue for transcription."""
        try:
            self.transcript_queue.put_nowait(audio_segment)
            self.logger.debug("Utterance sent to ASR queue.")
        except queue.Full:
            self.logger.warning(
                "ASR queue is full. Dropping utterance to maintain real-time performance."
            )

    def stop(self):
        """Sends shutdown signal to the ASR worker and waits for it to finish."""
        if self._worker_thread and self._worker_thread.is_alive():
            self.logger.info("Sending shutdown signal to ASR thread...")
            self.transcript_queue.put(None)  # Send sentinel
            self._worker_thread.join(timeout=5)  # Wait for thread to finish
            if self._worker_thread.is_alive():
                self.logger.warning(
                    "ASR thread did not terminate gracefully within timeout."
                )
