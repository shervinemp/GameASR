import unittest
from unittest.mock import MagicMock
from voice_control.llm.context import ContextManager
from voice_control.llm.conversation import Conversation
from voice_control.llm.model import LLM


class MockLLM(LLM):
    def __init__(self, n_ctx=100, max_tokens=10):
        super().__init__()
        self.n_ctx = n_ctx
        self.max_tokens = max_tokens
        self.logger = MagicMock()

    def _infer(self, conversation, *, session_state, **kwargs):
        yield "response"

    def count_tokens(self, text):
        # 1 char = 1 token for simplicity in test
        return len(text)


class TestContextPruning(unittest.TestCase):
    def test_pruning(self):
        llm = MockLLM(n_ctx=50, max_tokens=10)  # History limit = 40
        cm = ContextManager()

        conv = Conversation()
        # Add messages.
        # "msg1" = 4 tokens + 4 overhead = 8 tokens
        # We want to exceed 40 tokens. 5 messages * 8 = 40. 6th message should trigger pruning.

        for i in range(6):
            conv.add_user_message(f"msg{i}")

        # Verify initial state
        self.assertEqual(len(conv.messages), 6)

        # Run manager
        cm.manage_context(conv, llm)

        # Total tokens = 6 * 8 = 48 > 40.
        # It should prune until <= 40.
        # Pruning 1 message: 40 tokens. Perfect.
        # Pruning 2 messages: 32 tokens.

        # Depending on implementation loop, it stops when <= limit.
        # Pruning 1st message (msg0) -> 40 tokens remaining. 40 <= 40. Stop.
        # So we expect 5 messages remaining (msg1..msg5)

        self.assertEqual(len(conv.messages), 5)
        self.assertEqual(conv.messages[0]["content"], "msg1")
        self.assertEqual(conv.messages[-1]["content"], "msg5")

    def test_no_pruning_needed(self):
        llm = MockLLM(n_ctx=100, max_tokens=10)  # History limit = 90
        cm = ContextManager()

        conv = Conversation()
        for i in range(5):
            conv.add_user_message(f"msg{i}")

        # Total tokens = 5 * 8 = 40 <= 90.
        cm.manage_context(conv, llm)

        self.assertEqual(len(conv.messages), 5)


if __name__ == "__main__":
    unittest.main()
