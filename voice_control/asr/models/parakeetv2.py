from queue import Queue
import threading
from typing import Callable, Generator, Iterable
from collections import deque
import numpy as np
import sounddevice as sd

from .base import ModelBase
from ...common.utils import get_logger
from ...common.base import ConsumerProducer

try:
    from onnx_asr import load_model, load_vad
except ImportError:
    raise ImportError(
        "ONNX-ASR is not installed. Please install it using: pip install onnx-asr"
    )

_provider_lock = threading.Lock()


class ParakeetV2(ModelBase):

    def __init__(self, sound_device: int = 0):
        self._model = load_model("nemo-parakeet-tdt-0.6b-v2", quantization="int8")
        self._vad = Silero()

        global _provider_lock
        self._lock = _provider_lock

        super().__init__(sound_device)

    def _consume(self, chunk: Iterable[float]):
        self._vad(chunk)

    def _produce(self) -> Generator[str, None, None]:
        for e in self._vad:
            r = None
            with self._lock:
                r = self._model.recognize(e, sample_rate=self._vad._model.SAMPLE_RATE)
            yield r

    def _inputstream(self, sound_device: int, callback: Callable):
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
        post_speech_silence_dur: float = 0.75,
        pre_speech_dur: float = 0.75,
    ):
        self.logger = get_logger(__name__)

        self._model = load_vad("silero")
        self._queue = Queue(maxsize=1000)

        global _provider_lock
        self._lock = _provider_lock

        self.vad_threshold = vad_threshold
        self.post_speech_silence_dur = post_speech_silence_dur
        self.pre_speech_dur = pre_speech_dur

        self.reset()

    def reset(self):
        self._is_speech_segment = False
        self._silence_counter = 0

        pre_speech_chunks = int(
            self.pre_speech_dur * self._model.SAMPLE_RATE / self._model.HOP_SIZE
        )
        self._pre_speech_buffer = deque(maxlen=pre_speech_chunks)

        self._trailing_silent_chunks = int(
            self.post_speech_silence_dur
            * self._model.SAMPLE_RATE
            / self._model.HOP_SIZE
            + 1
        )

        self._model_input_frame = np.zeros(
            self._model.CONTEXT_SIZE + self._model.HOP_SIZE, dtype=np.float32
        )

    def _produce(self) -> Generator[np.ndarray, None, None]:
        buffer = deque()
        while True:
            if (c := self._queue.get()) is not None:
                buffer.append(c)
            elif len(buffer) >= 10:
                yield np.concatenate(buffer)
                buffer.clear()

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
            self._queue.put(chunk)
            if not is_loud:
                self._silence_counter += 1
                if self._silence_counter >= self._trailing_silent_chunks:
                    self._is_speech_segment = False
                    self._silence_counter = 0
                    self._queue.put(None)

        if not is_loud:
            self._pre_speech_buffer.append(chunk)
