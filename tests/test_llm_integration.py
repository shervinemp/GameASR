"""Integration tests that run the actual LLM model.

Non-deterministic (model output varies).  Uses best-of-5: each scenario
runs 5 times; passes if >= 3 agree on the expected outcome.

Separated from unit tests because the GGUF must be downloaded.

Run with:
  python -m pytest tests/test_llm_integration.py -v --tb=short
"""
import time
import unittest

import pytest

from voice_control.llm.model import Gemma4E4B
from voice_control.llm.conversation import Conversation
from voice_control.llm.tools import Tool, ToolCall


def _has_model():
    from voice_control.common.model_manager import ensure_downloaded
    try:
        ensure_downloaded("Gemma4E4B")
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_model(), reason="Gemma4E4B GGUF not downloaded")


def _decode_tool(raw_tokens: list[str]) -> list[ToolCall]:
    """Parse raw tokens through GeneralDecoder and return ToolCalls found."""
    from voice_control.llm.decoders import GeneralDecoder
    return [x for x in GeneralDecoder()(iter(raw_tokens)) if isinstance(x, ToolCall)]


class TestModelBehavior(unittest.TestCase):
    """Best-of-5 non-deterministic model behavior tests.

    Each scenario runs 5 times.  Passes if >= 3 runs match the expected
    tool-call pattern (0 calls for greetings, 1+ calls for knowledge).
    Tools are NEVER executed — we only check if the decoder can parse
    the model's output format.
    """

    _model = None

    @classmethod
    def setUpClass(cls):
        t0 = time.monotonic()
        cls._model = Gemma4E4B()
        cls._model.logger = type("L", (), {"info": print, "warning": print})()
        print(f"\nModel loaded in {time.monotonic()-t0:.1f}s")

    def _run_once(self, message: str, with_tools: bool,
                  demanding_prompt: bool = True) -> list[ToolCall]:
        """Run the model once, return parsed ToolCalls (never executed)."""
        conv = Conversation()
        if demanding_prompt:
            conv.set_system_message(
                "You are a voice-controlled game assistant. "
                "Respond conversationally and naturally.\n\nRules:\n"
                "- Call 'retrieve' when the user asks about entities, "
                "relationships, or facts.\n"
                "- If the user's message seems incomplete or cut off, "
                "ask what they meant before proceeding."
            )
        else:
            conv.set_system_message(
                "You are a helpful voice-controlled assistant. "
                "Respond conversationally and naturally."
            )
        if with_tools:
            t = Tool.from_callable("retrieve", lambda q: "")
            t.instruction = "Call when asked about entities or facts."
            conv.tools["retrieve"] = t
        conv.add_user_message(message)

        raw = []
        for chunk in self._model(conv, session_state={}):
            if isinstance(chunk, str):
                raw.append(chunk)
        return _decode_tool(raw)

    def _best_of_5(self, message: str, with_tools: bool,
                   demanding_prompt: bool = True) -> list[int]:
        """Run 5 times.  Returns list of call counts per run."""
        counts = []
        for i in range(5):
            calls = self._run_once(message, with_tools, demanding_prompt)
            counts.append(len(calls))
        return counts

    # 2x2 matrix: tools × prompt
    #   tools=yes + prompt=demanding  → model has tool + told to use it
    #   tools=yes + prompt=plain      → model has tool but not told to use it
    #   tools=no  + prompt=demanding  → model told to use tool but none available
    #   tools=no  + prompt=plain      → model has neither

    # --- Greetings (expect 0 calls in all 4 combos) ---

    def test_greeting_tools_demanding(self):
        counts = self._best_of_5("Hi.", with_tools=True, demanding_prompt=True)
        ok = sum(1 for c in counts if c == 0)
        print(f"  tools+demand [{ok}/5]: {counts}")
        self.assertGreaterEqual(ok, 3)

    def test_greeting_tools_plain(self):
        counts = self._best_of_5("Hi.", with_tools=True, demanding_prompt=False)
        ok = sum(1 for c in counts if c == 0)
        print(f"  tools+plain [{ok}/5]: {counts}")
        self.assertGreaterEqual(ok, 3)

    def test_greeting_no_tools_demanding(self):
        counts = self._best_of_5("Hi.", with_tools=False, demanding_prompt=True)
        ok = sum(1 for c in counts if c == 0)
        print(f"  no_tools+demand [{ok}/5]: {counts}")
        self.assertGreaterEqual(ok, 3)

    def test_greeting_no_tools_plain(self):
        counts = self._best_of_5("Hi.", with_tools=False, demanding_prompt=False)
        ok = sum(1 for c in counts if c == 0)
        print(f"  no_tools+plain [{ok}/5]: {counts}")
        self.assertGreaterEqual(ok, 3)

    # --- Knowledge (expect >=1 calls only when tools=yes + prompt=demanding) ---

    def test_knowledge_tools_demanding(self):
        counts = self._best_of_5("What is the Elden Ring?", with_tools=True, demanding_prompt=True)
        ok = sum(1 for c in counts if c >= 1)
        print(f"  tools+demand [{ok}/5]: {counts}")
        self.assertGreaterEqual(ok, 3)

    def test_knowledge_tools_plain(self):
        counts = self._best_of_5("What is the Elden Ring?", with_tools=True, demanding_prompt=False)
        ok = sum(1 for c in counts if c >= 1)
        print(f"  tools+plain [{ok}/5]: {counts}")
        # Model has the tool but wasn't told to use it — may or may not call
        # This is informational, not a hard assertion
        print(f"  (informational: called in {ok}/5 runs with tool but no prompt instruction)")

    def test_knowledge_no_tools_demanding(self):
        counts = self._best_of_5("What is the Elden Ring?", with_tools=False, demanding_prompt=True)
        ok = sum(1 for c in counts if c == 0)
        print(f"  no_tools+demand [{ok}/5]: {counts}")
        self.assertGreaterEqual(ok, 3)

    def test_knowledge_no_tools_plain(self):
        counts = self._best_of_5("What is the Elden Ring?", with_tools=False, demanding_prompt=False)
        ok = sum(1 for c in counts if c == 0)
        print(f"  no_tools+plain [{ok}/5]: {counts}")
        self.assertGreaterEqual(ok, 3)

    # --- Decoder format verification ---

    def test_tool_call_format_is_parseable(self):
        """When model calls a tool, verify both decoders can parse the format."""
        from voice_control.llm.decoders import GeneralDecoder, GemmaE2BDecoder

        t = Tool.from_callable("retrieve", lambda q: "")
        t.instruction = "Call when asked about entities or facts."

        parseable_by = {"general": 0, "gemma": 0}

        for i in range(5):
            conv = Conversation()
            conv.set_system_message("You are a game assistant.")
            conv.tools["retrieve"] = t
            conv.add_user_message("Tell me about Skyrim.")

            raw = [x for x in self._model(conv, session_state={}) if isinstance(x, str)]
            gd_ok = any(isinstance(x, ToolCall) for x in GeneralDecoder()(iter(raw)))
            ge_ok = any(isinstance(x, ToolCall) for x in GemmaE2BDecoder()(iter(raw)))
            if gd_ok:
                parseable_by["general"] += 1
            if ge_ok:
                parseable_by["gemma"] += 1

        print(f"  Parseable by GeneralDecoder: {parseable_by['general']}/5")
        print(f"  Parseable by GemmaE2BDecoder: {parseable_by['gemma']}/5")
        self.assertGreaterEqual(parseable_by["general"], 3,
                                "GeneralDecoder couldn't parse output in enough runs")
        self.assertGreaterEqual(parseable_by["gemma"], 3,
                                "GemmaE2BDecoder couldn't parse output in enough runs")


if __name__ == "__main__":
    unittest.main()
