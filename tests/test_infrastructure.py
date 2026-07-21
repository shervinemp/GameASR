"""Tests for pipeline infrastructure: gate, splitter, events, context, registry."""
import os
import re
import time
import threading
import unittest
from unittest.mock import MagicMock, patch, PropertyMock


class TestTranscriptGate(unittest.TestCase):
    from voice_control.transcript_gate import qualify_transcript
    _gate = staticmethod(qualify_transcript)

    def test_returns_none_for_empty(self):
        self.assertEqual(self._gate(""), (None, None))

    def test_returns_none_for_noise(self):
        self.assertEqual(self._gate("..."), (None, None))
        self.assertEqual(self._gate("   "), (None, None))

    def test_returns_text_for_valid(self):
        text, ann = self._gate("hello world")
        self.assertEqual(text, "hello world")
        self.assertIsNone(ann)

    def test_single_word_passes(self):
        text, ann = self._gate("yes")
        self.assertEqual(text, "yes")

    def test_annotation_for_hanging_word(self):
        text, ann = self._gate("I think that")
        self.assertIsNotNone(text)
        self.assertIsNotNone(ann)
        self.assertIn("partial", ann)

    def test_no_annotation_for_complete_sentence(self):
        text, ann = self._gate("I think that is correct.")
        self.assertEqual(text, "I think that is correct.")
        self.assertIsNone(ann)


class TestStreamSplitter(unittest.TestCase):
    def _split(self, chunks, min_len=0):
        from voice_control.common.base import stream_splitter
        return list(stream_splitter(iter(chunks), min_len=min_len))

    def test_single_sentence(self):
        self.assertEqual(self._split(["Hello world."]), ["Hello world."])

    def test_sentences_across_chunks(self):
        self.assertEqual(self._split(["Hello. How ", "are ", "you? Fine."]),
                         ["Hello.", "How are you?", "Fine."])

    def test_abbreviation_not_split(self):
        self.assertEqual(self._split(["Dr. Smith is here. He came."]),
                         ["Dr. Smith is here.", "He came."])

    def test_no_trailing_punctuation(self):
        self.assertEqual(self._split(["Hello without punctuation"]),
                         ["Hello without punctuation"])

    def test_empty_chunks(self):
        self.assertEqual(self._split([]), [])

    def test_empty_string_chunk(self):
        self.assertEqual(self._split([""]), [])

    def test_min_len_triggers_scan(self):
        r = self._split(["Hi. Bye."], min_len=8)
        self.assertEqual(len(r), 2)
        self.assertEqual(r[0], "Hi.")

    def test_one_sentence_per_iteration(self):
        r = self._split(["Hello. How are you? I'm fine."])
        self.assertEqual(r[0], "Hello.")


class TestEventEmitter(unittest.TestCase):
    def setUp(self):
        from voice_control.events import EventEmitter
        self.em = EventEmitter()

    def test_on_and_emit(self):
        events = []
        self.em.on("test", lambda *a, **kw: events.append((a, kw)))
        self.em.emit("test", 1, key="val")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0][0][0], 1)

    def test_off_removes_handler(self):
        count = 0
        def cb():
            nonlocal count
            count += 1
        self.em.on("test", cb)
        self.em.off("test", cb)
        self.em.emit("test")
        self.assertEqual(count, 0)

    def test_multiple_handlers(self):
        results = []
        self.em.on("ev", lambda: results.append("a"))
        self.em.on("ev", lambda: results.append("b"))
        self.em.emit("ev")
        self.assertEqual(results, ["a", "b"])

    def test_emit_no_handlers(self):
        self.em.emit("nonexistent")

    def test_async_handler(self):
        import time
        results = []
        def slow():
            time.sleep(0.02)
            results.append("done")
        self.em.on("ev", slow, async_=True)
        self.em.emit("ev")
        time.sleep(0.05)
        self.assertIn("done", results)


class TestDropOldestStrategy(unittest.TestCase):
    def setUp(self):
        from voice_control.llm.context import DropOldestStrategy
        self.strat = DropOldestStrategy(max_turns=3)

    def _conv(self, n_msgs):
        from voice_control.llm.conversation import Conversation
        c = Conversation()
        for i in range(n_msgs):
            c.add_user_message(f"msg{i}")
        return c

    def _llm(self):
        m = MagicMock()
        m.count_tokens.return_value = 10
        type(m).n_ctx = PropertyMock(return_value=4096)
        type(m).max_tokens = PropertyMock(return_value=512)
        return m

    def test_does_not_trim_below_max_turns(self):
        c = self._conv(3)
        self.strat.trim(c, self._llm())
        self.assertEqual(len(c._messages), 3)

    def test_trims_to_max_turns(self):
        c = self._conv(10)
        self.strat.trim(c, self._llm())
        self.assertLessEqual(len(c._messages), 6)

    def test_empty_conversation(self):
        from voice_control.llm.conversation import Conversation
        c = Conversation()
        self.strat.trim(c, self._llm())
        self.assertEqual(len(c._messages), 0)


class TestConsumerProducer(unittest.TestCase):
    def test_enable_disable(self):
        from voice_control.common.base import ConsumerProducer
        class Impl(ConsumerProducer):
            def _consume(self, value): self.last = value
            def _produce(self): yield from [1, 2, 3]
        impl = Impl()
        impl("hello")
        self.assertEqual(impl.last, "hello")
        impl.disable()
        impl("world")
        self.assertEqual(impl.last, "hello")
        impl.enable()
        impl("again")
        self.assertEqual(impl.last, "again")

    def test_passthrough(self):
        from voice_control.common.base import ConsumerProducer
        class Impl(ConsumerProducer):
            def _consume(self, value): self.last = value
            def _produce(self): yield from []
        impl = Impl()
        impl.disable_w_passthrough("fixed")
        impl("input")
        self.assertEqual(impl.last, "fixed")

    def test_iter(self):
        from voice_control.common.base import ConsumerProducer
        class Impl(ConsumerProducer):
            def _consume(self, value): pass
            def _produce(self): yield from [10, 20]
        impl = Impl()
        self.assertEqual(list(impl), [10, 20])


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


class TestLLMProvidersRegistry(unittest.TestCase):
    def test_get_known(self):
        from voice_control.llm.model import LLMProviders
        self.assertIsNotNone(LLMProviders.get("Qwen3"))

    def test_get_unknown_raises(self):
        from voice_control.llm.model import LLMProviders, ProviderError
        with self.assertRaises(ProviderError):
            LLMProviders.get("nonexistent")

    def test_get_empty_raises(self):
        from voice_control.llm.model import LLMProviders, ProviderError
        with self.assertRaises(ProviderError):
            LLMProviders.get("")

    def test_registry_contains_expected(self):
        from voice_control.llm.model import LLMProviders
        for name in ["Qwen3", "Gemma4E2B", "Gemma4_12B", "Gemma4E4B", "LiteLLM"]:
            self.assertIsNotNone(LLMProviders.get(name), f"Missing: {name}")


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


class TestHotkeyDispatcher(unittest.TestCase):
    def test_register_unregister(self):
        from voice_control.hotkey_dispatcher import HotkeyDispatcher
        d = HotkeyDispatcher()
        results = []
        def cb():
            results.append("fired")
        d.register("<ctrl>+r", cb)
        self.assertEqual(len(d.hotkeys), 1)
        d.unregister("<ctrl>+r")
        self.assertEqual(len(d.hotkeys), 0)
        d.stop()

    def test_multiple_registrations(self):
        from voice_control.hotkey_dispatcher import HotkeyDispatcher
        d = HotkeyDispatcher()
        d.register("<ctrl>+a", lambda: None)
        d.register("<ctrl>+b", lambda: None)
        self.assertEqual(len(d.hotkeys), 2)
        d.stop()

    def test_register_invalid_hotkey(self):
        from voice_control.hotkey_dispatcher import HotkeyDispatcher
        d = HotkeyDispatcher()
        with self.assertRaises(ValueError):
            d.register("invalid!!", lambda: None)
        d.stop()


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


class TestModelManager(unittest.TestCase):
    def test_ensure_downloaded_unknown(self):
        from voice_control.common.model_manager import ensure_downloaded
        from voice_control.exceptions import ModelError
        with self.assertRaises(ModelError):
            ensure_downloaded("nonexistent")

    def test_manifest_path_exists(self):
        from voice_control.common.model_manager import _MANIFEST_PATH
        self.assertTrue(os.path.exists(_MANIFEST_PATH))

    def test_manifest_yaml_loads(self):
        import yaml
        from voice_control.common.model_manager import _MANIFEST_PATH
        with open(_MANIFEST_PATH) as f:
            data = yaml.safe_load(f)
        self.assertIn("Gemma4E4B", data)
        self.assertIn("repo", data["Gemma4E4B"])


if __name__ == "__main__":
    unittest.main()
