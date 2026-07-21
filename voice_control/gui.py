"""PyQt6 always-on-top mic overlay factory."""


def MicButton(pipeline):
    from .gui_qt import MicOverlay
    return MicOverlay(pipeline)
