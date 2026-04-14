import queue
import threading
import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, Generator, Iterable

from sounddevice import InputStream

from ...common.base import ConsumerProducer


class ModelBase(ConsumerProducer, ABC):

    def __init__(
        self,
        sound_device: int,
    ):
        self.audio_queue = queue.Queue(maxsize=100)
        self._is_muted = threading.Event()
        self._is_running = threading.Event()
        self._is_running.set()

        self._vad_thread = threading.Thread(target=self._vad_worker, daemon=True)
        self._vad_thread.start()

        def sound_cb(in_data, frames, time, status):
            if status:
                print(f"Audio Status: {status}")
            try:
                # Strictly wait-free. Copy to avoid memory corruption from C.
                self.audio_queue.put_nowait(in_data.copy())
            except queue.Full:
                pass # Drop frame gracefully rather than crashing

        self._input_stream = self._inputstream(sound_device, sound_cb)

    def _vad_worker(self):
        while self._is_running.is_set():
            try:
                chunk = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if self._is_muted.is_set():
                # Safely evaluate state inside the consumer thread
                # Shape of silence_frame should match the chunk shape
                silence_frame = np.zeros(chunk.shape, dtype=chunk.dtype)
                self.__call__(silence_frame)
            else:
                self.__call__(chunk)

    def enable(self):
        super().enable()
        self._is_muted.clear()

    def disable_w_passthrough(self, value=None):
        super().disable_w_passthrough(value)
        self._is_muted.set()

    @abstractmethod
    def _consume(self, chunk: Iterable[float]): ...

    @abstractmethod
    def _produce(self) -> Generator[str, None, None]: ...

    @abstractmethod
    def _inputstream(self, device: int, callback: Callable) -> InputStream: ...

    def start(self):
        self._input_stream.start()

    def stop(self):
        self._input_stream.stop()
        self._is_running.clear()
        if hasattr(self, "_vad_thread"):
            self._vad_thread.join(timeout=1.0)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
