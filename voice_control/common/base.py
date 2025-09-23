from abc import ABC, abstractmethod
import re
from typing import Any, Generator, Iterable


class ConsumerProducer(ABC):
    __allow_consume: bool = True
    __allow_passthrough: bool = False
    __passthrough: Any

    def __call__(self, value):
        if self.__allow_consume:
            self._consume(value)
        elif self.__allow_passthrough:
            self._consume(self.__passthrough)

    def __iter__(self):
        yield from self._produce()

    def enable(self):
        self.__allow_consume = True
        self.__allow_passthrough = False

    def disable(self):
        self.__allow_consume = False
        self.__allow_passthrough = False

    def disable_w_passthrough(self, value: Any = None):
        self.__allow_consume = False
        self.__allow_passthrough = True
        self.__passthrough = value

    @abstractmethod
    def _consume(self, value): ...

    @abstractmethod
    def _produce(self) -> Generator: ...


def stream_splitter(
    text_stream: Iterable[str], min_len: int = 0
) -> Generator[str, None, None]:
    """
    Splits a text into sentences, ensuring each sentence is at least `min_len`
    characters long.
    """
    sentences = re.compile(r"[^.][.!?]\s+")
    buffer = ""
    for chunk in text_stream:
        buffer += chunk
        if len(buffer) >= min_len:
            if matches := sentences.finditer(buffer):
                for match in matches:
                    start, end = (match.start() + 2, match.end())
                    sentence = buffer[:start]
                    buffer = buffer[end:]
                    yield sentence.strip()
    else:
        if s := buffer.strip():
            yield s
