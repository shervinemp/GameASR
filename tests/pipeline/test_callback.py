"""Pipeline._callback flow: gate -> context -> LLM -> TTS."""
import unittest
from unittest.mock import MagicMock


class TestPipelineCallback(unittest.TestCase):
    """Pipeline._callback flow: gate -> context -> LLM -> TTS."""

    def _make(self, **kw):
        from voice_control.pipeline import Pipeline
        p = Pipeline.__new__(Pipeline)
        p.logger = MagicMock()
        p.events = MagicMock()
        p._rag = None
        p._conv_bank = None
        p._conv_history_enabled = False
        p._conv_threshold = 0.75
        p._conv_top_k = 2
        p._embedder = MagicMock()
        p._response_parts = []
        p._llm_busy = False
        p._interrupt_event = MagicMock()
        p._interrupt_event.is_set.return_value = False
        p._interrupted_at = None
        p._match_command = MagicMock(return_value=False)
        p.session = MagicMock()
        p.session.return_value = iter([])
        p.tts = None
        for k, v in kw.items():
            setattr(p, k, v)
        return p

    def test_callback_gate_filters_noise(self):
        pipe = self._make()
        pipe._callback("...")
        pipe.session.assert_not_called()

    def test_callback_valid_text_calls_session(self):
        pipe = self._make()
        pipe.session.return_value = iter(["hello world"])
        pipe._callback("hello world")
        pipe.session.assert_called_once()

    def test_callback_match_command_skips_llm(self):
        pipe = self._make(_match_command=MagicMock(return_value=True))
        pipe._callback("stop")
        pipe.session.assert_not_called()
