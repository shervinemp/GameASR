_HANGING = frozenset({
    "and", "but", "or", "nor", "yet", "so", "for",
    "if", "when", "while", "because", "although", "unless",
    "since", "until", "after", "before", "though",
    "the", "a", "an", "in", "on", "at", "to", "by",
    "with", "about", "into", "through", "during",
    "that", "which", "who", "whom",
    "is", "are", "was", "were", "has", "have", "had",
    "would", "could", "should", "can", "will",
})


def qualify_transcript(text: str) -> tuple[str | None, str | None]:
    """Returns (cleaned_text_or_None, annotation_or_None).

    Cleans and classifies raw ASR output before it reaches the LLM.
    """
    t = (text or "").strip()
    words = t.split()

    # Noise / fragment — drop silently
    if not words or len(words) < 2 or not any(c.isalpha() for c in t):
        return None, None

    # Incomplete — ends with a hanging word that strongly suggests more speech
    last = words[-1].lower().rstrip(".!?\"')")
    if last in _HANGING:
        return t, f"(partial: {t})"

    return t, None
