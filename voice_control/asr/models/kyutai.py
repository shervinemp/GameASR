from typing import Callable


from ..model import get_model_class
from ...common.utils import get_logger


class KyutaiSTT:
    def __init__(
        self,
        callback: Callable,
        sound_device: int = 0,
    ):
        self.callback = callback
        self.sound_device = sound_device
        self.logger = get_logger(__name__)
        self.model = self._load_model()

    def _load_model(self):
        # This is a placeholder for the actual model loading logic
        self.logger.info("Loading KyutaiSTT model...")
        return None

    def __call__(self, samples, sample_rate):
        # This is a placeholder for the actual transcription logic
        if samples.size == 0:
            return ""
        self.logger.info("Transcribing audio with KyutaiSTT...")
        return "this is a test sentence for the voice detection system"
