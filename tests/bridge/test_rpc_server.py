"""RPC server integration tests."""
import unittest
from unittest.mock import MagicMock, patch

from voice_control.bridge.llm_server import LLMService, LLMServer
from voice_control.llm.session import Session
from voice_control.llm.conversation import Conversation
from voice_control.llm.tools import ToolCall


class _FakeLLM:
    """Minimal LLM stand-in that yields text tokens."""
    decoder = None

    def __init__(self, tokens=None):
        self.tokens = tokens or ["response text"]
        self.logger = MagicMock()

    def __call__(self, conversation, session_state=None, **kwargs):
        yield from self.tokens

    def create_context_strategy(self, max_turns=20):
        from voice_control.llm.context import DropOldestStrategy
        return DropOldestStrategy(max_turns)

    def count_tokens(self, text):
        return max(1, len(text) // 2)


class TestLLMService(unittest.TestCase):
    """LLMService.query() validation and integration."""

    def _make_service(self, tokens=None):
        conv = Conversation()
        llm = _FakeLLM(tokens)
        sess = Session(llm=llm, conversation=conv)
        return LLMService(sess), conv

    def test_query_returns_string(self):
        service, _ = self._make_service()
        result = service.query("hello")
        self.assertIsInstance(result, str)

    def test_query_empty_raises(self):
        service, _ = self._make_service()
        with self.assertRaises(Exception):
            service.query("")
        with self.assertRaises(Exception):
            service.query("   ")

    def test_query_non_string_raises(self):
        service, _ = self._make_service()
        with self.assertRaises(Exception):
            service.query(123)

    def test_query_too_long_raises(self):
        service, _ = self._make_service()
        with self.assertRaises(Exception):
            service.query("x" * 20_000)

    def test_query_wrong_role_raises(self):
        service, _ = self._make_service()
        with self.assertRaises(Exception):
            service.query("hello", role="assistant")

    def test_query_adds_user_message(self):
        service, conv = self._make_service(["ok"])
        service.query("test query")
        user_msgs = [m for m in conv.messages if m.get("role") == "user"]
        self.assertTrue(any("test query" in m.get("content", "") for m in user_msgs),
                        "user message should be in conversation")

    def test_query_with_empty_response(self):
        """Empty model response returns empty string, no crash."""
        service, _ = self._make_service(tokens=[])
        result = service.query("test")
        self.assertIsInstance(result, str)


class TestLLMServerValidation(unittest.TestCase):
    """LLMServer constructor validation."""

    def test_requires_rpc_methods(self):
        service = MagicMock()
        service.rpc_methods = frozenset()
        with self.assertRaises(Exception):
            LLMServer(service, "tcp://127.0.0.1:0")

    def test_rejects_small_max_bytes(self):
        service = MagicMock()
        service.rpc_methods = frozenset({"query"})
        with self.assertRaises(Exception):
            LLMServer(service, "tcp://127.0.0.1:0", max_request_bytes=512)

    def test_rejects_large_max_bytes(self):
        service = MagicMock()
        service.rpc_methods = frozenset({"query"})
        with self.assertRaises(Exception):
            LLMServer(service, "tcp://127.0.0.1:0", max_request_bytes=2_000_000)

    def test_loopback_no_auth(self):
        service = MagicMock()
        service.rpc_methods = frozenset({"query"})
        server = LLMServer(service, "tcp://127.0.0.1:0")
        self.assertIsNone(server.auth_token)

    def test_non_loopback_requires_auth(self):
        service = MagicMock()
        service.rpc_methods = frozenset({"query"})
        with self.assertRaises(Exception):
            LLMServer(service, "tcp://0.0.0.0:0")
