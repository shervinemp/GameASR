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

from ...common.base_component import BaseComponent
from ...common.logging_utils import get_logger

# Get a logger for this module
logger = get_logger(__name__)


class ASRService(BaseComponent):
    """
    Manages the ASR model and a dedicated worker thread for asynchronous transcription.
    Delivers results via a callback function.
    """

    def __init__(
        self,
        samplerate: int = 16000,
        max_queue_size: int = 5,
        transcription_callback: callable = print,
        device=None,
    ):
        super().__init__()
        # Load the ONNX-ASR model
        self.asr_model = load_model("nemo-parakeet-tdt-0.6b-v2", quantization="int8")
        logger.info("ASR model loaded.")

        if device is not None:
            self.asr_model.set_device(device)

        self.samplerate = samplerate
        self.transcription_callback = transcription_callback  # Callback for results
        self.transcription_queue = queue.Queue(maxsize=max_queue_size)
        self._worker_thread = None

    def start(self):
        """Starts the ASR worker thread."""
        if not self._worker_thread:
            self._worker_thread = threading.Thread(
                target=self._asr_worker_loop,
                args=(
                    self.asr_model,
                    self.transcription_queue,
                    self.samplerate,
                    self.transcription_callback,
                ),
                daemon=True,  # Daemon thread exits when main program exits
            )
            self._worker_thread.start()
            logger.info("ASR worker thread started.")

    def _asr_worker_loop(self, model, input_queue, sample_rate, callback):
        """The main loop executed by the ASR worker thread."""
        threading.current_thread().name = "ASR_Worker"
        while True:
            try:
                audio_segment = input_queue.get(
                    timeout=0.1
                )  # Short timeout to check for shutdown

                if audio_segment is None:  # Sentinel for shutdown
                    logger.info("Shutdown signal received. Exiting.")
                    break

                logger.info("Processing transcription...")
                transcriptions = model.recognize(audio_segment, sample_rate=sample_rate)

                if transcriptions:
                    callback(transcriptions)
                else:
                    logger.info("No transcription generated for this segment.")

                input_queue.task_done()

            except queue.Empty:
                continue  # Keep trying if queue is empty
            except Exception as e:
                logger.error(f"An unexpected error occurred during transcription: {e}")
                if audio_segment is not None:
                    input_queue.task_done()

    def enqueue_utterance(self, audio_segment):
        """Adds an audio segment to the queue for transcription."""
        try:
            self.transcription_queue.put_nowait(audio_segment)
            logger.info("Utterance sent to ASR queue.")
        except queue.Full:
            logger.warning(
                "ASR queue is full. Dropping utterance to maintain real-time performance."
            )

    def stop(self):
        """Sends shutdown signal to the ASR worker and waits for it to finish."""
        if self._worker_thread and self._worker_thread.is_alive():
            logger.info("Sending shutdown signal to ASR thread...")
            self.transcription_queue.put(None)  # Send sentinel
            self._worker_thread.join(timeout=5)  # Wait for thread to finish
            if self._worker_thread.is_alive():
                logger.warning(
                    "ASR thread did not terminate gracefully within timeout."
                )
