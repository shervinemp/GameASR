from collections import deque
import threading
import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, Generator, Iterable

from sounddevice import InputStream

from ...common.base import ConsumerProducer


class ModelBase(ConsumerProducer, ABC):

    def __init__(
        self,
        sound_device: int | str,
    ):
        self.audio_queue = deque(maxlen=100)
        self._audio_event = threading.Event()
        self._is_muted = threading.Event()
        self._is_running = threading.Event()
        self._is_running.set()

        self._vad_thread = threading.Thread(target=self._vad_worker, daemon=True)
        self._vad_thread.start()

        def sound_cb(in_data, frames, time, status):
            if status:
                print(f"Audio Status: {status}")
            self.audio_queue.append(in_data.copy())
            self._audio_event.set()

        self._input_stream = self._inputstream(sound_device, sound_cb)

    def _vad_worker(self):
        while self._is_running.is_set():
            try:
                chunk = self.audio_queue.popleft()
            except IndexError:
                self._audio_event.wait(timeout=0.05)
                self._audio_event.clear()
                continue

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
    def _inputstream(self, device: int | str, callback: Callable) -> InputStream: ...

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
