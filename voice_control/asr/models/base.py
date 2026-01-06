from abc import ABC, abstractmethod
from queue import Queue, Empty
import threading
from typing import Callable, Generator, Iterable

from sounddevice import InputStream

from ...common.base import ConsumerProducer


class ModelBase(ConsumerProducer, ABC):

    def __init__(
        self,
        sound_device: int,
    ):
        self._audio_queue = Queue()
        self._worker_thread = None
        self._running = False

        def sound_cb(in_data, frames, time, status):
            self._audio_queue.put(in_data.copy())

        self._input_stream = self._inputstream(sound_device, sound_cb)

    def _process_audio(self):
        while self._running:
            try:
                data = self._audio_queue.get(timeout=0.5)
                self.__call__(data)
            except Empty:
                pass

    @abstractmethod
    def _consume(self, chunk: Iterable[float]): ...

    @abstractmethod
    def _produce(self) -> Generator[str, None, None]: ...

    @abstractmethod
    def _inputstream(self, device: int, callback: Callable) -> InputStream: ...

    def start(self):
        if not self._running:
            self._running = True
            self._worker_thread = threading.Thread(
                target=self._process_audio, daemon=True
            )
            self._worker_thread.start()
            self._input_stream.start()

    def stop(self):
        if self._running:
            self._running = False
            self._input_stream.stop()
            if self._worker_thread:
                self._worker_thread.join(timeout=1.0)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
