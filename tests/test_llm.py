import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

from voice_control.llm.conversation import Conversation
from voice_control.llm.model import (
    LLMProviders,
    LiteLLMProvider,
    Qwen3,
)
from voice_control.llm.tools import ToolCall


class TestLLM(unittest.TestCase):
    @patch("voice_control.llm.model.os.path.exists", return_value=True)
    @patch("voice_control.llm.model.GGUFLLM.__init__", return_value=None)
    @patch("voice_control.llm.model.Llama", create=True)
    def test_gguf_llm(self, mock_llama, mock_init, mock_exists):
        """
        Test a GGUF-backed LLM.
        """
        # Mock the Llama model
        mock_model = MagicMock()
        mock_model.create_chat_completion.return_value = iter(
            [{"choices": [{"delta": {"content": "This is a test."}}]}]
        )
        mock_llama.return_value = mock_model

        # Initialize the LLM
        llm = Qwen3()
        llm.model = mock_model
        llm.max_tokens = 128
        llm._last_state = None
        llm._lock = MagicMock()
        llm._parse = MagicMock(side_effect=lambda x: x)
        llm.logger = MagicMock()

        # Create a conversation
        conversation = Conversation()
        conversation.add_user_message("Hello")

        # Get the response
        response = "".join(llm._infer(conversation, session_state={}))

        # Check the response
        self.assertEqual(response, "This is a test.")

    def test_empty_conversation(self):
        """
        Test that the LLM returns an empty string for an empty conversation.
        """
        completion = MagicMock(return_value=iter([]))

        # Initialize the LLM
        llm = LiteLLMProvider(model="test", provider="ollama", completion_fn=completion)

        # Create an empty conversation
        conversation = Conversation()

        # Get the response
        response = "".join(llm(conversation))

        # Check the response
        self.assertEqual(response, "")

    @patch("voice_control.llm.model.os.path.exists", return_value=True)
    @patch("voice_control.llm.model.GGUFLLM.__init__", return_value=None)
    @patch("voice_control.llm.model.Llama", create=True)
    def test_qwen_llm(self, mock_llama, mock_init, mock_exists):
        """
        Test the QwenLLM.
        """
        # Mock the Llama model
        mock_model = MagicMock()
        mock_model.create_chat_completion.return_value = iter(
            [{"choices": [{"delta": {"content": "This is a test."}}]}]
        )
        mock_llama.return_value = mock_model

        # Initialize the LLM
        llm = Qwen3()
        llm.model = mock_model
        llm.max_tokens = 128
        llm._last_state = None
        llm._lock = MagicMock()
        llm._parse = MagicMock(side_effect=lambda x: x)
        llm.logger = MagicMock()

        # Create a conversation
        conversation = Conversation()
        conversation.add_user_message("Hello")

        # Get the response
        response = "".join(llm._infer(conversation, session_state={}))

        # Check the response
        self.assertEqual(response, "This is a test.")

    def test_ollama_llm(self):
        """
        Test the OllamaLLM.
        """
        completion = MagicMock(
            return_value=iter(
                [
                    {
                        "choices": [
                            {"delta": {"content": "This is a test."}}
                        ]
                    }
                ]
            )
        )

        # Initialize the LLM
        llm = LiteLLMProvider(
            model="test", provider="ollama",
            api_base="http://localhost:11434", completion_fn=completion,
        )

        # Create a conversation
        conversation = Conversation()
        conversation.add_user_message("Hello")

        # Get the response
        response = "".join(llm(conversation))

        # Check the response
        self.assertEqual(response, "This is a test.")
        request = completion.call_args.kwargs
        self.assertEqual(request["model"], "ollama/test")
        self.assertEqual(request["api_base"], "http://localhost:11434")
        self.assertEqual(request["timeout"], 60.0)
        self.assertEqual(request["num_retries"], 0)

    def test_litellm_model_prefixing(self):
        completion = MagicMock(return_value=iter([]))
        openai = LiteLLMProvider(
            model="gpt-test", provider="openai",
            api_key="test-openai-key", completion_fn=completion,
        )
        gemini = LiteLLMProvider(
            model="gemini-test", provider="gemini",
            api_key="test-gemini-key", completion_fn=completion,
        )

        self.assertEqual(openai.model, "openai/gpt-test")
        self.assertEqual(gemini.model, "gemini/gemini-test")

    def test_remote_plaintext_provider_endpoint_is_rejected(self):
        with self.assertRaises(ValueError):
            LiteLLMProvider(
                model="test", provider="ollama",
                api_base="http://192.0.2.10:11434",
                completion_fn=MagicMock(),
            )

    def test_litellm_stream_reassembles_fragmented_tool_calls(self):
        completion = MagicMock(
            return_value=iter(
                [
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "function": {
                                                "name": "pi",
                                                "arguments": '{"value":',
                                            },
                                        }
                                    ]
                                }
                            }
                        ]
                    },
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "function": {
                                                "name": "ng",
                                                "arguments": "1}",
                                            },
                                        }
                                    ]
                                }
                            }
                        ]
                    },
                ]
            )
        )
        llm = LiteLLMProvider(model="test", provider="ollama", completion_fn=completion)

        result = list(llm._infer(Conversation(), session_state={}))

        self.assertEqual(result, [ToolCall(name="ping", arguments={"value": 1})])

    def test_litellm_malformed_tool_arguments_are_not_executed(self):
        completion = MagicMock(
            return_value=iter(
                [
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "function": {
                                                "name": "ping",
                                                "arguments": "[1, 2]",
                                            },
                                        }
                                    ]
                                }
                            }
                        ]
                    }
                ]
            )
        )
        llm = LiteLLMProvider(model="test", provider="ollama", completion_fn=completion)

        result = list(llm._infer(Conversation(), session_state={}))

        self.assertEqual(
            result,
            [ToolCall(name="_parse_error", arguments={"tool_name": "ping"})],
        )

    def test_provider_factory_uses_allowlisted_provider_configuration(self):
        completion = MagicMock(return_value=iter([]))

        provider = LLMProviders.create(
            "LiteLLM",
            {
                "litellm": {
                    "provider": "ollama",
                    "model": "test",
                    "api_base": "http://127.0.0.1:11434",
                    "completion_fn": completion,
                }
            },
        )

        self.assertIsInstance(provider, LiteLLMProvider)
        self.assertEqual(provider.model, "ollama/test")
        with self.assertRaises(ValueError):
            LLMProviders.get("__class__")

    def test_litellm_import_uses_local_cost_map_by_default(self):
        fake_litellm = types.ModuleType("litellm")
        fake_litellm.completion = MagicMock()

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.dict(sys.modules, {"litellm": fake_litellm}),
        ):
            LiteLLMProvider(model="test", provider="ollama")
            self.assertEqual(
                os.environ["LITELLM_LOCAL_MODEL_COST_MAP"], "True"
            )


class TestDecoders(unittest.TestCase):

    def _collect(self, decoder, chunks):
        calls, text = [], []
        for item in decoder(iter(chunks)):
            if isinstance(item, ToolCall):
                calls.append((item.name, item.arguments))
            elif isinstance(item, str):
                text.append(item)
        return calls, "".join(text)

    def _check(self, decoder, chunks, expect_calls, expect_text=""):
        calls, text = self._collect(decoder, chunks)
        self.assertEqual(calls, expect_calls)
        self.assertEqual(text, expect_text)

    # --- GeneralDecoder ---

    def test_gd_plain_text(self):
        from voice_control.llm.decoders import GeneralDecoder
        d = GeneralDecoder()
        self._check(d, ["Hello! How are you?"], [], "Hello! How are you?")

    def test_gd_gemma_call_format(self):
        from voice_control.llm.decoders import GeneralDecoder
        d = GeneralDecoder()
        self._check(d, [
            "<|tool_call>",
            'call:retrieve{query:<|"|>Elden Ring<|"|>}',
            "<tool_call|>",
        ], [("retrieve", {"query": "Elden Ring"})])

    def test_gd_gemma_with_pretext(self):
        from voice_control.llm.decoders import GeneralDecoder
        d = GeneralDecoder()
        self._check(d, [
            "I'll check! ",
            "<|tool_call>",
            'call:search{query:<|"|>hello<|"|>}',
            "<tool_call|>",
        ], [("search", {"query": "hello"})], "I'll check! ")

    def test_gd_standard_toolcall(self):
        from voice_control.llm.decoders import GeneralDecoder
        d = GeneralDecoder()
        self._check(d, [
            "<toolcall>",
            '{"name": "retrieve", "arguments": {"q": "test"}}',
            "</toolcall>",
        ], [("retrieve", {"q": "test"})])

    def test_gd_html_escaped(self):
        from voice_control.llm.decoders import GeneralDecoder
        d = GeneralDecoder()
        self._check(d, [
            "&lt;toolcall&gt;",
            '{"function": "search", "arguments": {"x": 1}}',
            "&lt;/toolcall&gt;",
        ], [("search", {"x": 1})])

    def test_gd_numeric_args(self):
        from voice_control.llm.decoders import GeneralDecoder
        d = GeneralDecoder()
        self._check(d, [
            "<|tool_call>",
            "call:get{id:42,limit:10}",
            "<tool_call|>",
        ], [("get", {"id": 42, "limit": 10})])

    def test_gd_bool_and_null(self):
        from voice_control.llm.decoders import GeneralDecoder
        d = GeneralDecoder()
        self._check(d, [
            "<|tool_call>",
            "call:check{active:true,data:null}",
            "<tool_call|>",
        ], [("check", {"active": True, "data": None})])

    def test_gd_empty_args(self):
        from voice_control.llm.decoders import GeneralDecoder
        d = GeneralDecoder()
        self._check(d, [
            "<|tool_call>", "call:status{}", "<tool_call|>",
        ], [("status", {})])

    def test_gd_no_tool(self):
        from voice_control.llm.decoders import GeneralDecoder
        d = GeneralDecoder()
        self._check(d, ["Just a regular response."], [], "Just a regular response.")

    def test_gd_halts_after_first_tool(self):
        from voice_control.llm.decoders import GeneralDecoder
        d = GeneralDecoder()
        self._check(d, [
            "<|tool_call>", 'call:a{x:1}', "<tool_call|>",
            "should not appear",
            "<|tool_call>", 'call:b{y:2}', "<tool_call|>",
        ], [("a", {"x": 1})])

    def test_gd_chunked_streaming(self):
        from voice_control.llm.decoders import GeneralDecoder
        d = GeneralDecoder()
        self._check(d, [
            "pre<|tool_cal",
            "l>call:retrieve{query:<|",
            '"|>hi<|"|>}<tool_call|>post',
        ], [("retrieve", {"query": "hi"})], "pre")

    # --- GemmaE2BDecoder ---

    def test_ge_plain_text(self):
        from voice_control.llm.decoders import GemmaE2BDecoder
        d = GemmaE2BDecoder()
        self._check(d, ["Hello!"], [], "Hello!")

    def test_ge_gemma_call(self):
        from voice_control.llm.decoders import GemmaE2BDecoder
        d = GemmaE2BDecoder()
        self._check(d, [
            "pre ",
            "<|tool_call>",
            'call:retrieve{query:<|"|>hi<|"|>}',
            "<tool_call|>",
            "post",
        ], [("retrieve", {"query": "hi"})], "pre ")

    def test_ge_halts_after_first_tool(self):
        from voice_control.llm.decoders import GemmaE2BDecoder
        d = GemmaE2BDecoder()
        self._check(d, [
            "<|tool_call>", 'call:a{x:1}', "<tool_call|>", "more",
        ], [("a", {"x": 1})])

    # --- LegacyXMLDecoder ---

    def test_lxd_standard(self):
        from voice_control.llm.decoders import LegacyXMLDecoder
        d = LegacyXMLDecoder()
        self._check(d, [
            "pre ",
            "<toolcall>",
            '{"name": "retrieve", "arguments": {"q": "test"}}',
            "</toolcall>",
        ], [("retrieve", {"q": "test"})], "pre ")

    def test_lxd_function_key(self):
        from voice_control.llm.decoders import LegacyXMLDecoder
        d = LegacyXMLDecoder()
        self._check(d, [
            "<toolcall>",
            '{"function": "search", "arguments": {"x": 1}}',
            "</toolcall>",
        ], [("search", {"x": 1})])

    # --- Cross-decoder consistency ---

    def test_cross_decoder_gemma_format(self):
        from voice_control.llm.decoders import GeneralDecoder, GemmaE2BDecoder
        chunks = [
            "<|tool_call>",
            'call:retrieve{query:<|"|>test<|"|>}',
            "<tool_call|>",
        ]
        r1 = list(GeneralDecoder()(iter(chunks)))
        r2 = list(GemmaE2BDecoder()(iter(chunks)))
        self.assertEqual(len(r1), len(r2))
        if r1 and r2:
            self.assertEqual(r1[0].name, r2[0].name)
            self.assertEqual(r1[0].arguments, r2[0].arguments)


class TestStreamDecoders(unittest.TestCase):
    """Comprehensive edge-case tests for all decoders."""

    def _collect(self, decoder, chunks):
        calls, text = [], []
        for item in decoder(iter(chunks)):
            if isinstance(item, ToolCall):
                calls.append((item.name, item.arguments))
            elif isinstance(item, str):
                text.append(item)
        return calls, "".join(text)

    def _check(self, decoder, chunks, expect_calls, expect_text=""):
        calls, text = self._collect(decoder, chunks)
        self.assertEqual(calls, expect_calls)
        self.assertEqual(text, expect_text)

    # --- GeneralDecoder edge cases ---

    def test_gd_custom_formats(self):
        from voice_control.llm.decoders import GeneralDecoder
        d = GeneralDecoder(formats=[
            {"open": "<custom>", "close": "</custom>", "parse": "json"},
        ])
        self._check(d, [
            "<custom>",
            '{"name": "my_tool", "arguments": {}}',
            "</custom>",
        ], [("my_tool", {})])

    def test_gd_empty_formats_list(self):
        from voice_control.llm.decoders import GeneralDecoder
        d = GeneralDecoder(formats=[])
        self._check(d, ["some plain text"], [], "some plain text")

    def test_gd_thought_tag_dropped(self):
        from voice_control.llm.decoders import GeneralDecoder
        d = GeneralDecoder()
        self._check(d, [
            "before ",
            "<|channel>thought\n",
            "internal monologue here",
            "<channel|>",
            " after",
        ], [], "before  after")

    def test_gd_thought_no_close(self):
        from voice_control.llm.decoders import GeneralDecoder
        d = GeneralDecoder()
        self._check(d, [
            "pre<|channel>thought\n",
            "open thought never closed",
        ], [], "pre")

    def test_gd_tag_in_regular_text(self):
        from voice_control.llm.decoders import GeneralDecoder
        d = GeneralDecoder()
        # <toolcall> without </toolcall> — opener detected, decoder waits for
        # close that never arrives.  Text after the opener is accumulated as
        # body and dropped when the stream ends.
        self._check(d, ["see <toolcall> in docs"], [], "see ")

    # --- GemmaE2BDecoder edge cases ---

    def test_ge_malformed_json(self):
        from voice_control.llm.decoders import GemmaE2BDecoder
        d = GemmaE2BDecoder()
        calls, text = self._collect(d, [
            "<|tool_call>call:bad{invalid json!!!}<tool_call|>",
        ])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "_parse_error")

    def test_ge_quotes_in_args(self):
        from voice_control.llm.decoders import GemmaE2BDecoder
        d = GemmaE2BDecoder()
        self._check(d, [
            "<|tool_call>",
            'call:search{query:<|"|>it'+chr(39)+'s fine<|"|>}',
            "<tool_call|>",
        ], [("search", {"query": "it's fine"})])

    def test_ge_nested_channel_references(self):
        from voice_control.llm.decoders import GemmaE2BDecoder
        d = GemmaE2BDecoder()
        self._check(d, [
            "text<|channel>thought\n",
            "thinking<channel|>",
            " out",
        ], [], "text out")

    def test_ge_unicode_in_args(self):
        from voice_control.llm.decoders import GemmaE2BDecoder
        d = GemmaE2BDecoder()
        self._check(d, [
            "<|tool_call>",
            'call:find{name:<|"|>caf\\u00e9<|"|>}',
            "<tool_call|>",
        ], [("find", {"name": "caf\u00e9"})])

    def test_ge_empty_stream(self):
        from voice_control.llm.decoders import GemmaE2BDecoder
        d = GemmaE2BDecoder()
        self._check(d, [], [])

    # --- LegacyXMLDecoder edge cases ---

    def test_lxd_buffer_overflow(self):
        from voice_control.llm.decoders import LegacyXMLDecoder
        d = LegacyXMLDecoder()
        big_body = "{" + "x" * 10_001 + "}"
        calls, text = self._collect(d, [
            "<toolcall>",
            big_body,
            "</toolcall>",
        ])
        # Decoder discards the oversized buffer and yields _parse_error
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "_parse_error")

    def test_lxd_malformed_json(self):
        from voice_control.llm.decoders import LegacyXMLDecoder
        d = LegacyXMLDecoder()
        calls, text = self._collect(d, [
            "<toolcall>{bad</toolcall>",
        ])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "_parse_error")

    def test_lxd_partial_tag_at_boundary(self):
        from voice_control.llm.decoders import LegacyXMLDecoder
        d = LegacyXMLDecoder()
        self._check(d, [
            "pre<toolcal",
            'l>{"name":"t","arguments":{}}',
            "</toolcall>",
        ], [("t", {})], "pre")

    def test_lxd_empty_stream(self):
        from voice_control.llm.decoders import LegacyXMLDecoder
        d = LegacyXMLDecoder()
        self._check(d, [], [])

    # --- NativeDecoder ---

    def test_nd_passthrough(self):
        from voice_control.llm.decoders import NativeDecoder
        d = NativeDecoder()
        chunks = ["hello", " world"]
        result = list(d(iter(chunks)))
        self.assertEqual(result, ["hello", " world"])

    def test_nd_mixed_content(self):
        from voice_control.llm.decoders import NativeDecoder
        d = NativeDecoder()
        tc = ToolCall(name="test", arguments={"x": 1})
        chunks = ["a", tc, "b"]
        result = list(d(iter(chunks)))
        self.assertEqual(result, ["a", tc, "b"])

    def test_nd_empty(self):
        from voice_control.llm.decoders import NativeDecoder
        d = NativeDecoder()
        self.assertEqual(list(d(iter([]))), [])

    # --- StreamDecoder ABC ---

    def test_abc_cannot_instantiate(self):
        from voice_control.llm.decoders import StreamDecoder
        with self.assertRaises(TypeError):
            StreamDecoder()

    # --- ToolCall passthrough in all decoders ---

    def test_toolcall_passthrough_gd(self):
        from voice_control.llm.decoders import GeneralDecoder
        d = GeneralDecoder()
        tc = ToolCall(name="direct", arguments={"from": "test"})
        result = list(d(iter([tc])))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "direct")


class TestSessionIntegration(unittest.TestCase):
    """End-to-end Session flow with a mocked LLM streaming output."""

    def _make_mock_llm(self, tokens: list[str], decoder=None):
        from voice_control.llm.decoders import GeneralDecoder
        from voice_control.llm.model import LLM
        from voice_control.llm.context import DropOldestStrategy
        _decoder = decoder or GeneralDecoder()

        class MockLLM(LLM):
            decoder = _decoder
            def _infer(self, conversation, *, session_state, **kwargs):
                yield from tokens
            def create_context_strategy(self, max_turns=20):
                return DropOldestStrategy(max_turns)
            def count_tokens(self, text):
                return max(1, len(text) // 2)

        llm = MockLLM()
        llm.logger = MagicMock()
        return llm

    def _collect(self, gen):
        text, calls = [], []
        for item in gen:
            if isinstance(item, ToolCall):
                calls.append((item.name, item.arguments))
            elif isinstance(item, str):
                text.append(item)
        return text, calls

    def test_session_plain_text(self):
        from voice_control.llm.session import Session
        llm = self._make_mock_llm(["Hello! ", "How ", "can ", "I ", "help?"])
        sess = Session(llm=llm)
        text, calls = self._collect(sess("Hi."))
        self.assertEqual("".join(text), "Hello! How can I help?")
        self.assertEqual(calls, [])
        sess.close()

    def test_session_with_tool_call(self):
        from voice_control.llm.session import Session
        from voice_control.llm.tools import Tool
        from voice_control.llm.conversation import Conversation
        tool_results = []
        def my_tool(query: str) -> str:
            tool_results.append(query)
            return f"result for {query}"
        tokens = [
            "Let me check...",
            "<|tool_call>", 'call:my_tool{query:<|"|>hello<|"|>}', "<tool_call|>",
        ]
        llm = self._make_mock_llm(tokens)
        conv = Conversation()
        conv.set_system_message("You are a helpful assistant.")
        conv.tools["my_tool"] = Tool.from_callable("my_tool", my_tool)
        conv.add_user_message("find hello")
        sess = Session(llm=llm, conversation=conv)
        text, calls = self._collect(sess())
        self.assertEqual(len(tool_results), 1)
        self.assertEqual(tool_results[0], "hello")
        msgs = sess.conversation.messages
        tool_msgs = [m for m in msgs if m["role"] == "tool"]
        self.assertIn("result for hello", tool_msgs[0]["content"])
        sess.close()

    def test_session_second_pass_uses_tool_results(self):
        from voice_control.llm.session import Session
        from voice_control.llm.tools import Tool
        from voice_control.llm.conversation import Conversation
        from voice_control.llm.decoders import GeneralDecoder

        class TwoPassLLM:
            decoder = GeneralDecoder()
            _call_count = 0
            def create_context_strategy(self, max_turns=20):
                from voice_control.llm.context import DropOldestStrategy
                return DropOldestStrategy(max_turns)
            def count_tokens(self, text):
                return max(1, len(text) // 2)
            def __call__(self, conversation, session_state=None, **kwargs):
                self._call_count += 1
                if self._call_count == 1:
                    tokens = ["<|tool_call>", 'call:retrieve{query:<|"|>test<|"|>}', "<tool_call|>"]
                else:
                    tokens = ["Based on retrieval, the answer is 42."]
                yield from self.decoder(iter(tokens))

        def retrieve(query: str) -> str:
            return "42"
        conv = Conversation()
        conv.set_system_message("You are a helpful assistant.")
        conv.tools["retrieve"] = Tool.from_callable("retrieve", retrieve)
        conv.add_user_message("what is the answer?")
        llm = TwoPassLLM()
        llm.logger = MagicMock()
        sess = Session(llm=llm, conversation=conv)
        text, calls = self._collect(sess())
        self.assertIn("Based on retrieval, the answer is 42.", "".join(text))
        sess.close()

    def test_session_recovers_from_tool_error(self):
        from voice_control.llm.session import Session
        from voice_control.llm.tools import Tool
        from voice_control.llm.conversation import Conversation
        def broken_tool(**kwargs):
            raise RuntimeError("tool exploded")
        tokens = ["<|tool_call>", 'call:broken_tool{x:1}', "<tool_call|>"]
        llm = self._make_mock_llm(tokens)
        conv = Conversation()
        conv.tools["broken_tool"] = Tool.from_callable("broken_tool", broken_tool)
        conv.add_user_message("do something")
        sess = Session(llm=llm, conversation=conv)
        text, calls = self._collect(sess())
        msgs = sess.conversation.messages
        self.assertTrue(any("Tool Error" in m["content"] for m in msgs if m["role"] == "tool"))
        sess.close()

    def test_session_unknown_tool(self):
        from voice_control.llm.session import Session
        from voice_control.llm.conversation import Conversation
        # Tool name not in conversation.tools — triggers retry, then fallback
        tokens = ["<|tool_call>", 'call:nonexistent{args:{}}', "<tool_call|>"]
        llm = self._make_mock_llm(tokens)
        conv = Conversation()
        conv.add_user_message("test")
        sess = Session(llm=llm, conversation=conv)
        text, calls = self._collect(sess())
        combined = "".join(text)
        self.assertIn("persistent error", combined)

    def test_session_lock_serializes_access(self):
        from voice_control.llm.session import Session
        import threading, time
        call_order = []

        class LockTestLLM:
            def create_context_strategy(self, max_turns=20):
                from voice_control.llm.context import DropOldestStrategy
                return DropOldestStrategy(max_turns)
            def count_tokens(self, text):
                return max(1, len(text) // 2)
            def __call__(self, conversation, session_state=None, **kwargs):
                call_order.append("enter")
                time.sleep(0.05)
                call_order.append("exit")
                return iter(["x"])

        llm = LockTestLLM()
        llm.logger = MagicMock()
        sess = Session(llm=llm)
        results = []
        def run():
            for chunk in sess("hello"):
                results.append(chunk)
        t1 = threading.Thread(target=run)
        t2 = threading.Thread(target=run)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        self.assertEqual(call_order, ["enter", "exit", "enter", "exit"])
        sess.close()

    def test_session_retry_on_parse_error(self):
        from voice_control.llm.session import Session
        from voice_control.llm.conversation import Conversation
        tokens = ["<|tool_call>", "call:bad_tool{invalid!!!}}", "<tool_call|>"]
        llm = self._make_mock_llm(tokens)
        conv = Conversation()
        conv.add_user_message("do it")
        sess = Session(llm=llm, conversation=conv)
        text, calls = self._collect(sess())
        self.assertIn("persistent error", "".join(text))
        sess.close()

    def test_session_empty_query(self):
        from voice_control.llm.session import Session
        llm = self._make_mock_llm([])
        sess = Session(llm=llm)
        text, calls = self._collect(sess())
        sess.close()

    def test_session_gather_timeout(self):
        from voice_control.llm.session import Session
        from voice_control.llm.tools import Tool
        from voice_control.llm.conversation import Conversation
        import time

        def slow_tool(**kwargs):
            time.sleep(20)
            return "too late"

        tokens = ["<|tool_call>", 'call:slow_tool{x:1}', "<tool_call|>"]
        llm = self._make_mock_llm(tokens)
        conv = Conversation()
        conv.tools["slow_tool"] = Tool.from_callable("slow_tool", slow_tool)
        conv.add_user_message("do slow")
        sess = Session(llm=llm, conversation=conv)
        t0 = time.monotonic()
        text, calls = self._collect(sess())
        elapsed = time.monotonic() - t0
        self.assertLess(elapsed, 15)
        msgs = sess.conversation.messages
        self.assertTrue(any("Tool Error" in m["content"] for m in msgs if m["role"] == "tool"))
        sess.close()

    def test_toolcall_passthrough_ge(self):
        from voice_control.llm.decoders import GemmaE2BDecoder
        d = GemmaE2BDecoder()
        tc = ToolCall(name="direct", arguments={"from": "test"})
        result = list(d(iter([tc])))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "direct")
