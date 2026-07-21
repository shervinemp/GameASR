"""Tests for configuration models and defaults."""
import unittest


class TestConfigModels(unittest.TestCase):
    def test_asr_config(self):
        from voice_control.common.config import config
        self.assertIsInstance(config.get("asr.provider"), str)

    def test_llm_config(self):
        from voice_control.common.config import config
        self.assertIsInstance(config.get("llm.provider"), str)

    def test_tts_config(self):
        from voice_control.common.config import config
        self.assertIsInstance(config.get("tts.provider"), str)

    def test_get_nonexistent(self):
        from voice_control.common.config import config
        self.assertIsNone(config.get("nonexistent.key"))

    def test_get_default(self):
        from voice_control.common.config import config
        self.assertEqual(config.get("nonexistent", default=42), 42)


class TestConfigModelsAdvanced(unittest.TestCase):
    """Configuration model validation."""

    def test_asr_config_fields(self):
        from voice_control.common.config_models import ASRConfig
        cfg = ASRConfig(provider="test", weights_dir="/tmp")
        self.assertEqual(cfg.provider, "test")

    def test_llm_config_fields(self):
        from voice_control.common.config_models import LLMConfig, LLMModelsConfig
        cfg = LLMConfig(provider="test",
                        models=LLMModelsConfig(default="", extraction_heavy="", embedding=""),
                        providers={})
        self.assertEqual(cfg.provider, "test")

    def test_tts_config_fields(self):
        from voice_control.common.config_models import TTSConfig
        cfg = TTSConfig(provider="test", weights_dir="/tmp")
        self.assertEqual(cfg.provider, "test")

    def test_rag_config_fields(self):
        from voice_control.common.config_models import RAGConfig
        cfg = RAGConfig()
        self.assertTrue(hasattr(cfg, "runtime"))

    def test_hotkey_config(self):
        from voice_control.common.config_models import HotkeyConfig
        cfg = HotkeyConfig()
        self.assertTrue(hasattr(cfg, "enable"))
