from queue import Empty, Queue
import threading
from time import sleep
from typing import Callable, Generator, Iterable
from collections import deque
import numpy as np

from .base import ModelBase
from ...common.base import ConsumerProducer
from ...common.utils import get_logger


_vad_lock = threading.Lock()


class ParakeetV2(ModelBase):

    def __init__(self, sound_device: int = 0):
        from onnx_asr import load_model

        self._model = load_model(
            "nemo-parakeet-tdt-0.6b-v2", quantization="int8"
        )
        self._vad = Silero()

        self._lock = _vad_lock

        super().__init__(sound_device)

    def _consume(self, chunk: Iterable[float]):
        self._vad(chunk)

    def _produce(self) -> Generator[str, None, None]:
        for e in self._vad:
            r = None
            with self._lock:
                r = self._model.recognize(
                    e, sample_rate=self._vad._model.SAMPLE_RATE
                )
            yield r

    def _inputstream(self, sound_device: int, callback: Callable):
        import sounddevice as sd

        return sd.InputStream(
            samplerate=self._vad._model.SAMPLE_RATE,
            blocksize=self._vad._model.HOP_SIZE,
            device=sound_device,
            channels=1,
            callback=callback,
        )


class Silero(ConsumerProducer):
    def __init__(
        self,
        vad_threshold: float = 0.4,
        leading_silence_duration: float = 1.0,
        trailing_silence_duration: float = 2.4,
        trailing_buffer_duration: float = 1.2,
    ):
        from onnx_asr import load_vad

        self.logger = get_logger(__name__)

        self._model = load_vad("silero")
        self._queue = Queue(maxsize=1000)

        self._lock = _vad_lock

        self.vad_threshold = vad_threshold
        self.pre_speech_dur = leading_silence_duration
        self.post_speech_dur = trailing_silence_duration
        self.post_speech_keep = trailing_buffer_duration

        self.reset()

    def reset(self):
        self._is_speech_segment = False
        self._silence_counter = 0

        coeff_ = self._model.SAMPLE_RATE / self._model.HOP_SIZE

        pre_speech_chunks = int(self.pre_speech_dur * coeff_)
        self._pre_speech_buffer = deque(maxlen=pre_speech_chunks)

        self._trailing_silent_chunks = int(self.post_speech_dur * coeff_ + 1)

        self._trailing_buffer_chunks = int(self.post_speech_keep * coeff_ + 1)

        self._model_input_frame = np.zeros(
            self._model.CONTEXT_SIZE + self._model.HOP_SIZE, dtype=np.float32
        )

    def _produce(self) -> Generator[np.ndarray, None, None]:
        buffer = deque()
        while True:
            try:
                if (c := self._queue.get(timeout=1)) is not None:
                    buffer.append(c)
                elif len(buffer) >= 10:
                    yield np.concatenate(buffer)
                    buffer.clear()
            except Empty:
                sleep(0.01)

    def _consume(self, chunk: Iterable[np.ndarray]):
        acquired = self._lock.acquire(timeout=2)
        if not acquired:
            return

        try:
            chunk = np.mean(chunk, axis=1)
            self._model_input_frame = np.concatenate(
                [self._model_input_frame[-self._model.CONTEXT_SIZE :], chunk]
            )

            speech_prob, *_ = self._model._encode(
                self._model_input_frame[np.newaxis, :]
            )
            speech_prob = speech_prob[0]
        finally:
            self._lock.release()

        is_loud = speech_prob > self.vad_threshold

        if not self._is_speech_segment and is_loud:
            self._is_speech_segment = True
            if self._pre_speech_buffer:
                r = np.concatenate(self._pre_speech_buffer, dtype=np.float32)
                self._queue.put(r)
                self._pre_speech_buffer.clear()

        if self._is_speech_segment:
            if is_loud:
                self._queue.put(chunk)
                self._silence_counter = 0
            else:
                self._silence_counter += 1
                if self._silence_counter <= self._trailing_buffer_chunks:
                    self._queue.put(chunk)
                if self._silence_counter >= self._trailing_silent_chunks:
                    self._is_speech_segment = False
                    self._silence_counter = 0
                    self._queue.put(None)

        if not is_loud:
            self._pre_speech_buffer.append(chunk)
