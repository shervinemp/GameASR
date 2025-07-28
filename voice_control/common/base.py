from abc import ABC, abstractmethod
from typing import Generator


class ConsumerProducer(ABC):

    def __call__(self, value):
        self._consume(value)

    def __iter__(self):
        yield from self._produce()

    @abstractmethod
    def _consume(self, value): ...

    @abstractmethod
    def _produce(self) -> Generator: ...
