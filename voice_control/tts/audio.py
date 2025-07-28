import atexit
from queue import Empty, Queue
import threading
from time import sleep
import numpy as np
import sounddevice as sd

from ..common.utils import get_logger


class AudioPlayer:

    def __init__(self, output_device: int | None = None):
        self.logger = get_logger(__name__)

        if output_device is None:
            output_device = sd.default.device[1]

        self.output_device = output_device
        device_name = sd.query_devices(output_device)["name"]
        self.logger.info(
            f"AudioPlayer initialized. Using output device: '{device_name}' (ID: {output_device})"
        )

        self._queue = Queue(maxsize=1000)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._running = False
        atexit.register(self.stop)

    def _run(self):
        while self._running:
            try:
                audio_data, sample_rate = self._queue.get(timeout=0.85)
                self.play(audio_data, sample_rate)
            except Empty:
                pass
            finally:
                sleep(0.15)

    def __call__(
        self,
        audio_data: np.ndarray[np.float32 | np.int16],
        sample_rate: int,
        interrupt: bool = False,
    ):
        return self._consume(audio_data, sample_rate, interrupt)

    def _consume(
        self,
        audio_data: np.ndarray[np.float32 | np.int16],
        sample_rate: int,
        interrupt: bool = False,
    ):
        if interrupt:
            with self._queue.mutex:
                self._queue.queue.clear()
            sd.stop()
        self._queue.put((audio_data, sample_rate))

    def play(
        self,
        audio_data: np.ndarray[np.float32 | np.int16],
        sample_rate: int,
    ):
        if audio_data.dtype != np.float32:
            audio_data = (
                audio_data.astype(np.float32) / 32768.0
                if audio_data.dtype == np.int16
                else audio_data.astype(np.float32)
            )
        if np.max(np.abs(audio_data)) > 1.0:
            audio_data /= np.max(np.abs(audio_data))

        sd.play(
            audio_data,
            samplerate=sample_rate,
            device=self.output_device,
            blocking=True,
        )

    def start(self):
        if not self._running:
            self._running = True
            self._thread.start()

    def stop(self):
        with self._queue.mutex:
            self._queue.queue.clear()
        self._running = False
        self._thread.join(timeout=5)
        sd.stop()


def main():
    logger = get_logger("AudioPlayerExample")

    player = AudioPlayer()

    logger.info("--- Testing interrupt playback with a generated sine wave ---")
    sample_rate = 44100
    frequency = 440  # A4 note
    duration = 2.0  # seconds
    t = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False)
    sine_wave = 0.5 * np.sin(2 * np.pi * frequency * t)

    player(sine_wave, sample_rate)
    sleep(duration)

    logger.info("Sine wave playback complete.\n")


if __name__ == "__main__":
    main()
