"""Integration tests that run the actual LLM model.

These are non-deterministic and require the GGUF file to be downloaded.
They're separated from the unit tests (test_llm.py) because model output
varies between runs.

Run with: python -m pytest tests/test_llm_integration.py -v --tb=short
Skip with: python -m pytest tests/ -v --ignore=tests/test_llm_integration.py
"""
import os
import sys
import time
import json
import unittest

import pytest

from voice_control.llm.model import Gemma4E4B
from voice_control.llm.conversation import Conversation
from voice_control.llm.tools import Tool, ToolCall


def _has_model():
    """Check if the GGUF model file exists on disk."""
    from voice_control.common.model_manager import ensure_downloaded
    try:
        ensure_downloaded("Gemma4E4B")
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_model(), reason="Gemma4E4B GGUF not downloaded")


class TestModelToolBehavior(unittest.TestCase):
    """Tests that require the actual model to verify tool-calling behavior."""

    @classmethod
    def setUpClass(cls):
        t0 = time.monotonic()
        cls.model = Gemma4E4B()
        cls.logger = type("Logger", (), {"info": print, "warning": print})()
        cls.model.logger = cls.logger
        print(f"\nModel loaded in {time.monotonic()-t0:.1f}s")

    def _run(self, message, tools=None):
        """Run inference and return (text, tool_calls, elapsed)."""
        conv = Conversation()
        conv.set_system_message(
            "You are a voice-controlled game assistant. Respond conversationally and naturally.\n\nRules:\n"
            "- Call 'retrieve' when the user asks about entities, relationships, or facts.\n"
            "- If the user's message seems incomplete or cut off, ask what they meant.\n"
            "- If you're about to perform an important action, confirm briefly."
        )
        if tools:
            conv.tools = tools
        conv.add_user_message(message)

        t0 = time.monotonic()
        text_parts = []
        calls = []
        for chunk in cls.model(conv, session_state={}):
            if isinstance(chunk, ToolCall):
                calls.append(chunk)
            elif isinstance(chunk, str):
                text_parts.append(chunk)
        elapsed = time.monotonic() - t0
        return "".join(text_parts), calls, elapsed

    # --- Greetings (should NOT call tools) ---

    def test_greeting_hi(self):
        text, calls, elapsed = self._run("Hi.")
        print(f"  [{elapsed:.1f}s] {len(calls)} calls, {len(text)} chars")
        self.assertEqual(len(calls), 0, f"Model called tool for greeting: {calls}")
        self.assertLess(elapsed, 10, f"Greeting took {elapsed:.1f}s — too slow")
        self.assertTrue(len(text) > 0, "Model returned empty response")

    def test_greeting_how_are_you(self):
        text, calls, elapsed = self._run("Hey, how are you?")
        print(f"  [{elapsed:.1f}s] {len(calls)} calls, {len(text)} chars")
        self.assertEqual(len(calls), 0)
        self.assertLess(elapsed, 10)
        self.assertTrue(len(text) > 0)

    # --- Knowledge queries (SHOULD call retrieve) ---

    def test_knowledge_elden_ring(self):
        retrieve_cb = lambda query: "Elden Ring is a 2022 action RPG."
        retrieve_tool = Tool.from_callable("retrieve", retrieve_cb)
        retrieve_tool.instruction = "Call when asked about entities or facts."
        text, calls, elapsed = self._run(
            "What do you know about the Elden Ring?",
            tools={"retrieve": retrieve_tool},
        )
        print(f"  [{elapsed:.1f}s] {len(calls)} calls, {len(text)} chars")
        self.assertGreater(len(calls), 0, "Model should call retrieve for knowledge query")
        self.assertLess(elapsed, 30)

    def test_knowledge_god_of_war(self):
        retrieve_cb = lambda query: "Kratos is the main character."
        retrieve_tool = Tool.from_callable("retrieve", retrieve_cb)
        retrieve_tool.instruction = "Call when asked about entities or facts."
        text, calls, elapsed = self._run(
            "Who is the main character in God of War?",
            tools={"retrieve": retrieve_tool},
        )
        print(f"  [{elapsed:.1f}s] {len(calls)} calls, {len(text)} chars")
        self.assertGreater(len(calls), 0, "Model should call retrieve for knowledge query")
        self.assertLess(elapsed, 30)

    # --- Tool call format verification ---

    def test_tool_call_format_is_parseable(self):
        """Verify the model's tool call format is parseable by both decoders."""
        from voice_control.llm.decoders import GeneralDecoder, GemmaE2BDecoder

        retrieve_cb = lambda query: "result"
        retrieve_tool = Tool.from_callable("retrieve", retrieve_cb)
        retrieve_tool.instruction = "Call when asked about entities or facts."

        conv = Conversation()
        conv.set_system_message("You are a game assistant.")
        conv.tools["retrieve"] = retrieve_tool
        conv.add_user_message("Tell me about Skyrim.")

        raw_tokens = []
        for chunk in cls.model(conv, session_state={}):
            if isinstance(chunk, str):
                raw_tokens.append(chunk)

        raw_text = "".join(raw_tokens)
        print(f"  Raw output ({len(raw_text)} chars): {raw_text[:200]}")

        # Both decoders should parse the tool call
        gd_ok = any(isinstance(x, ToolCall) for x in GeneralDecoder()(iter(raw_tokens)))
        ge_ok = any(isinstance(x, ToolCall) for x in GemmaE2BDecoder()(iter(raw_tokens)))

        self.assertTrue(gd_ok or ge_ok,
                        f"Neither decoder parsed tool call from:\n{raw_text[:300]}")


if __name__ == "__main__":
    unittest.main()
