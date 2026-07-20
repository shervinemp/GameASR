from queue import Empty, Queue
import threading
import time
from typing import Any, Callable, Generator, Iterable
from collections import deque
import numpy as np

from .base import ModelBase
from ...common.base import ConsumerProducer
from ...common.utils import get_logger
from ...exceptions import ASRError

_vad_lock = threading.Lock()


class ParakeetV2(ModelBase):

    def __init__(self, sound_device: int | str = 0):
        from onnx_asr import load_model

        # Resolve string device name to integer index
        if isinstance(sound_device, str):
            import sounddevice as sd
            devices = sd.query_devices()
            for i, dev in enumerate(devices):
                if sound_device.lower() in dev["name"].lower():
                    sound_device = i
                    break
            else:
                self.logger.warning(
                    f"Sound device '{sound_device}' not found. Using default (0)."
                )
                sound_device = 0

        self._model = load_model(
            "nemo-parakeet-tdt-0.6b-v2", quantization="int8"
        )

        # Read VAD settings from config before initializing Silero
        from ...common.config import config as _cfg
        vad_threshold = _cfg.get("asr.vad_threshold", 0.4)
        trailing_ms = _cfg.get("asr.trailing_silence_ms", 800)
        leading_ms = _cfg.get("asr.leading_silence_ms", 1000)
        max_segment = _cfg.get("asr.max_segment_duration", 30.0)
        self._vad = Silero(
            vad_threshold=vad_threshold,
            trailing_silence_duration=trailing_ms / 1000.0,
            leading_silence_duration=leading_ms / 1000.0,
            max_segment_duration=max_segment,
        )
        self._lock = _vad_lock

        super().__init__(sound_device)

        # Warm up ONNX model to avoid cold start latency on first utterance
        dummy = np.zeros(self._vad._model.SAMPLE_RATE, dtype=np.float32)
        self._model.recognize(dummy, sample_rate=self._vad._model.SAMPLE_RATE)

    def _consume(self, chunk: Iterable[float]):
        if self._is_muted.is_set():
            return
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

    def disable_w_passthrough(self, value: Any = None):
        value = np.zeros(
            (self._input_stream.blocksize, self._input_stream.channels),
            dtype=np.float32,
        )
        super().disable_w_passthrough(value)
        self._vad.flush()


class Silero(ConsumerProducer):
    def __init__(
        self,
        vad_threshold: float = 0.4,
        leading_silence_duration: float = 1.0,
        trailing_silence_duration: float = 0.8,
        trailing_buffer_duration: float = 1.2,
        max_segment_duration: float = 30.0,
        on_speech_onset: Callable | None = None,
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
        self.max_segment_duration = max_segment_duration
        self.on_speech_onset = on_speech_onset

        self.reset()

    def reset(self):
        self._is_speech_segment = False
        self._silence_counter = 0
        self._segment_start_time = time.monotonic()

        coeff_ = self._model.SAMPLE_RATE / self._model.HOP_SIZE

        pre_speech_chunks = int(self.pre_speech_dur * coeff_)
        self._pre_speech_buffer = deque(maxlen=pre_speech_chunks)

        self._trailing_silent_chunks = int(self.post_speech_dur * coeff_ + 1)

        self._trailing_buffer_chunks = int(self.post_speech_keep * coeff_ + 1)

        self._model_input_frame = np.zeros(
            self._model.CONTEXT_SIZE + self._model.HOP_SIZE, dtype=np.float32
        )

    def flush(self):
        with self._lock:
            if self._is_speech_segment:
                self._is_speech_segment = False
                self._silence_counter = 0
                self._queue.put(None)

    def _produce(self) -> Generator[np.ndarray, None, None]:
        buffer = deque()
        while True:
            try:
                c = self._queue.get(timeout=1)
                if c is not None:
                    buffer.append(c)
                else:
                    if len(buffer) >= 10:
                        yield np.concatenate(buffer)
                        buffer.clear()
            except Empty:
                pass

    def _consume(self, chunk: Iterable[np.ndarray]):
        acquired = self._lock.acquire(blocking=False)
        if not acquired:
            return

        try:
            # Force-flush if segment exceeds max duration (safety net for
            # sustained noise that keeps speech probability above threshold).
            if self._is_speech_segment and self.max_segment_duration > 0:
                elapsed = time.monotonic() - self._segment_start_time
                if elapsed > self.max_segment_duration:
                    self._is_speech_segment = False
                    self._silence_counter = 0
                    self._segment_start_time = time.monotonic()
                    self._queue.put(None)
                    return

            if len(chunk.shape) > 1 and chunk.shape[1] > 1:
                chunk = np.mean(chunk, axis=1)
            elif len(chunk.shape) > 1:
                chunk = chunk[:, 0]

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
            self._segment_start_time = time.monotonic()
            if self.on_speech_onset:
                self.on_speech_onset()
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
