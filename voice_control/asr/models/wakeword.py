from abc import ABC, abstractmethod

import numpy as np


class WakeWordState:
    IDLE = "idle"
    WAKING = "waking"
    ACTIVE = "active"


class WakeWordDetector(ABC):
    """Interface for a wake-word detection stage.

    A concrete implementation (e.g. Porcupine) processes raw audio chunks
    and signals when the wake word is spoken. The pipeline uses this to
    gate audio flow to the VAD — audio is dropped until the wake word
    fires.
    """

    @abstractmethod
    def process(self, chunk: np.ndarray) -> str:
        """Process an audio chunk. Return one of WakeWordState values.

        IDLE   — no wake word detected, continue monitoring
        WAKING — wake word detected but not yet confirmed (transitional)
        ACTIVE — wake word confirmed, audio should now flow to VAD
        """
        ...
