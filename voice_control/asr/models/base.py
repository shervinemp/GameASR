from abc import ABC, abstractmethod
from typing import Callable, Generator, Iterable

import sounddevice


class ConsumerProducer(ABC):

    def __call__(self, value):
        self._consume(value)

    def __iter__(self):
        yield from self._produce()

    @abstractmethod
    def _consume(self, value): ...

    @abstractmethod
    def _produce(self) -> Generator: ...


class ModelBase(ConsumerProducer, ABC):

    def __init__(
        self,
        sound_device: int,
    ):
        def sound_cb(in_data, frames, time, status):
            self._consume(in_data)

        self._input_stream = self._inputstream(sound_device, sound_cb)

    @abstractmethod
    def _consume(self, chunk: Iterable[float]): ...

    @abstractmethod
    def _produce(self) -> Generator[str, None, None]: ...

    @abstractmethod
    def _inputstream(
        self, device: int, callback: Callable
    ) -> sounddevice.InputStream: ...

    def start(self):
        self._input_stream.start()

    def stop(self):
        self._input_stream.stop()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
