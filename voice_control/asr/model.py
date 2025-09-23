from ..common.config import config
from .models import ParakeetV2


# ----------------------------------------------------------------------

asr_providers = {
    "parakeetv2": ParakeetV2,
}

provider = config.get("asr.default_provider", "parakeetv2")
default_class = asr_providers.get(provider)

# ----------------------------------------------------------------------
