from abc import ABC, abstractmethod
import re
from typing import Generator, Iterable


class ConsumerProducer(ABC):

    def __call__(self, value):
        self._consume(value)

    def __iter__(self):
        yield from self._produce()

    @abstractmethod
    def _consume(self, value): ...

    @abstractmethod
    def _produce(self) -> Generator: ...


def stream_splitter(text_stream: Iterable[str], min_len: int = 0) -> Generator[str, None, None]:
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
