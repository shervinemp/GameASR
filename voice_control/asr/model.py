from typing import Literal

model_types = Literal["parakeetv2", "kyutai"]


def get_model_class(model_name: model_types) -> type:
    if model_name == "parakeetv2":
        from .models import ParakeetV2

        return ParakeetV2
    elif model_name == "kyutai":
        from .models import KyutaiSTT

        return KyutaiSTT
