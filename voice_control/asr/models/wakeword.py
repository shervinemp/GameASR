# Wake-word scaffolding: the Silero gate (wake_word_detector param + _ww_active
# flag in parakeetv2.py:Silero._consume) is the full extent of what we need.
# A concrete detector implementing the process(chunk) -> "active" | str
# protocol can be plugged in when available.
