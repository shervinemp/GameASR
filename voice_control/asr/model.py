from typing import Literal

model_types = Literal["parakeetv2"]


def get_model_class(model_name: model_types) -> type:
    if model_name == "parakeetv2":
        from .models import ParakeetV2

        return ParakeetV2
