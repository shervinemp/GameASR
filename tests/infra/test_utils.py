"""Tests for utility functions and exception hierarchy."""
import unittest


class TestSafeJsonLoads(unittest.TestCase):
    def test_valid(self):
        from voice_control.common.utils import safe_json_loads
        self.assertEqual(safe_json_loads('{"a": 1}', fallback={}), {"a": 1})

    def test_invalid(self):
        from voice_control.common.utils import safe_json_loads
        self.assertEqual(safe_json_loads("{bad}", fallback=[1]), [1])

    def test_empty(self):
        from voice_control.common.utils import safe_json_loads
        self.assertEqual(safe_json_loads("", fallback="x"), "x")


class TestExceptions(unittest.TestCase):
    def test_imports(self):
        from voice_control.exceptions import (
            VoiceControlError, ASRError, LLMError, TTSError,
            ConfigError, ToolError, ProviderError, StorageError,
        )
        for exc in [ASRError, LLMError, TTSError, ConfigError, ToolError, ProviderError, StorageError]:
            self.assertTrue(issubclass(exc, VoiceControlError))

    def test_raise_catch(self):
        from voice_control.exceptions import LLMError, VoiceControlError
        try:
            raise LLMError("test")
        except VoiceControlError as e:
            self.assertEqual(str(e), "test")
