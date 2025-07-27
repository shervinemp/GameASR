from typing import Callable


class KyutaiSTT:
    def __init__(
        self,
        callback: Callable,
        sound_device: int = 0,
    ):
        super().__init__(
            callback=callback,
            sound_device=sound_device,
        )
