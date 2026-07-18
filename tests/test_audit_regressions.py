import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import zipfile

from voice_control.bridge.scaffold import BridgeLanguage, scaffold_bridge
from voice_control.bridge.llm_server import LLMServer, LLMService
from voice_control.common.utils import download_file, verify_file_sha256
from voice_control.llm.conversation import Conversation
from voice_control.llm.model import LLM, Ollama
from voice_control.llm.session import Session
from voice_control.llm.tools import Tool
from voice_control.rag.validation import normalize_triplets, queue_triplets
from voice_control.rag.data import CodexDataLoader


class EchoLLM(LLM):
    n_ctx = 4_096
    max_tokens = 128

    def __init__(self):
        super().__init__()
        self.logger = MagicMock()
        self.received = []

    def _infer(self, conversation, *, session_state, **kwargs):
        self.received.append(conversation.messages)
        yield "ok"


class EchoService:
    rpc_methods = frozenset({"echo"})

    def echo(self, value):
        return value


class TestAuditRegressions(unittest.TestCase):
    def test_generated_tool_schema_is_loadable(self):
        spec_path = Path(__file__).parents[1] / "api_spec.json"
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        tools = [Tool.from_dict(item) for item in spec["functions"]]

        self.assertEqual(
            [tool.name for tool in tools],
            ["move_unit", "command_unit", "set_mic_status", "get_squad_status"],
        )
        self.assertEqual(tools[0].parameters.properties["x"].type, "number")

    def test_legacy_tool_schema_remains_supported(self):
        tool = Tool.from_dict(
            {"name": "move", "description": "Move", "params": ["x", "y"]}
        )
        self.assertEqual(tool.parameters.required, ["x", "y"])
        with self.assertRaises(ValueError):
            Tool.from_dict({"function": {"name": "invalid-name!"}})
        with self.assertRaises(TypeError):
            Tool.from_dict({"type": "function", "function": "invalid"})

    def test_tool_arguments_are_strictly_checked(self):
        tool = Tool.from_dict(
            {
                "function": {
                    "name": "flag",
                    "parameters": {
                        "type": "object",
                        "properties": {"active": {"type": "boolean"}},
                        "required": ["active"],
                    },
                }
            }
        )
        tool.callback = lambda active: active
        self.assertFalse(tool(active="false"))
        with self.assertRaises(TypeError):
            tool(extra=True)

    def test_rpc_query_adds_a_plain_user_message(self):
        llm = EchoLLM()
        session = Session(llm)
        try:
            result = LLMService(session).query("move alpha")
            self.assertEqual(result, "ok")
            self.assertEqual(
                llm.received[0][-1],
                {"role": "user", "content": "move alpha"},
            )
        finally:
            session.close()

    def test_rpc_public_bind_requires_a_strong_token(self):
        with self.assertRaises(ValueError):
            LLMServer(EchoService(), "tcp://0.0.0.0:5555")
        with self.assertRaises(ValueError):
            LLMServer(EchoService(), "tcp://192.0.2.1:5555", auth_token="short")
        with self.assertRaises(ValueError):
            LLMServer(EchoService(), "tcp://127.0.0.1:5555", auth_token="short")
        LLMServer(EchoService(), "tcp://127.0.0.1:5555")
        LLMServer(
            EchoService(),
            "tcp://0.0.0.0:5555",
            auth_token="x" * 32,
        )

    def test_rpc_auth_shape_size_allowlist_and_rate_limit(self):
        server = LLMServer(
            EchoService(),
            "tcp://127.0.0.1:5555",
            auth_token="x" * 32,
            max_request_bytes=1_024,
            requests_per_minute=1,
        )
        unauthorized = json.loads(
            server._handle_request(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "echo",
                        "params": {"value": "hello"},
                    }
                )
            )
        )
        self.assertEqual(unauthorized["error"]["code"], -32001)

        accepted = json.loads(
            server._handle_request(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 2,
                        "method": "echo",
                        "params": {"value": "hello"},
                        "auth_token": "x" * 32,
                    }
                )
            )
        )
        self.assertEqual(accepted["result"], "hello")

        limited = json.loads(
            server._handle_request(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 3,
                        "method": "echo",
                        "params": {"value": "hello"},
                        "auth_token": "x" * 32,
                    }
                )
            )
        )
        self.assertEqual(limited["error"]["code"], -32002)

        oversized = json.loads(server._handle_request(" " * 1_025))
        self.assertEqual(oversized["error"]["code"], -32600)

    def test_ollama_tool_followup_omits_tools(self):
        completion = MagicMock()
        completion.side_effect = [
            iter(
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
                                                "arguments": {},
                                            },
                                        }
                                    ]
                                }
                            }
                        ]
                    }
                ]
            ),
            iter(
                [
                    {
                        "choices": [
                            {
                                "delta": {
                                    "content": "done",
                                    "tool_calls": None,
                                }
                            }
                        ]
                    }
                ]
            ),
        ]
        conversation = Conversation()
        conversation.tools = [
            Tool(name="ping", description="Ping", callback=lambda: "pong")
        ]
        session = Session(
            Ollama(model="test", completion_fn=completion), conversation
        )
        try:
            self.assertEqual("".join(session("go")), "done")
        finally:
            session.close()

        self.assertIn("tools", completion.call_args_list[0].kwargs)
        self.assertNotIn("tools", completion.call_args_list[1].kwargs)
        self.assertNotIn("tool_choice", completion.call_args_list[1].kwargs)

    def test_session_reset_reuses_one_tool_thread(self):
        session = Session(EchoLLM())
        try:
            original_thread = session.tool_caller._loop_thread
            session.conversation.add_user_message("old")
            session.reset()
            self.assertIs(session.tool_caller._loop_thread, original_thread)
            self.assertTrue(original_thread.is_alive())
            self.assertEqual(session.conversation.messages, [])
        finally:
            session.close()

    def test_one_shot_completion_does_not_mutate_conversation(self):
        session = Session(EchoLLM())
        try:
            session.conversation.add_user_message("persistent")
            before = list(session.conversation.messages)

            result = session.complete_once("isolated")

            self.assertEqual(result, "ok")
            self.assertEqual(
                session.llm.received[-1],
                [{"role": "user", "content": "isolated"}],
            )
            self.assertEqual(session.conversation.messages, before)
        finally:
            session.close()

    def test_triplets_are_validated_and_quarantined(self):
        normalized = normalize_triplets(
            [{"subject": "Alpha", "predicate": "squad member", "object": "Bravo"}]
        )
        self.assertEqual(normalized[0]["predicate"], "SQUAD_MEMBER")
        with self.assertRaises(ValueError):
            normalize_triplets(
                [
                    {
                        "subject": "Alpha",
                        "predicate": "KNOWS",
                        "object": "Bravo",
                        "untrusted": "field",
                    }
                ]
            )

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            queue_path = Path(temp_dir) / "pending.jsonl"
            written = queue_triplets(
                normalized,
                str(queue_path),
                query="Who is in the squad?",
                provenance="graph",
            )
            record = json.loads(written.read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "pending_review")
            self.assertEqual(record["triplets"], normalized)

        with self.assertRaises(ValueError):
            queue_triplets(
                normalized,
                str(Path.cwd().parent / "escape.jsonl"),
                query="test",
                provenance="graph",
            )

    def test_sha256_verification_rejects_tampering(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            artifact = Path(temp_dir) / "asset.bin"
            artifact.write_bytes(b"trusted")
            digest = hashlib.sha256(b"trusted").hexdigest()
            verify_file_sha256(str(artifact), digest)
            artifact.write_bytes(b"tampered")
            with self.assertRaises(ValueError):
                verify_file_sha256(str(artifact), digest)

    @patch("voice_control.common.utils.requests.get")
    def test_download_is_bounded_verified_and_atomic(self, requests_get):
        payload = b"verified artifact"
        response = MagicMock()
        response.is_redirect = False
        response.is_permanent_redirect = False
        response.headers = {"Content-Length": str(len(payload))}
        response.iter_content.return_value = iter([payload])
        requests_get.return_value = response

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            destination = Path(temp_dir) / "asset.bin"
            digest = hashlib.sha256(payload).hexdigest()
            download_file(
                "https://downloads.example.test/asset.bin",
                str(destination),
                expected_sha256=digest,
                allowed_hosts={"downloads.example.test"},
                max_bytes=1_024,
            )
            self.assertEqual(destination.read_bytes(), payload)
            self.assertEqual(requests_get.call_args.kwargs["timeout"], (5, 60))
            self.assertFalse(requests_get.call_args.kwargs["allow_redirects"])

            destination.write_bytes(b"existing")
            with self.assertRaises(ValueError):
                download_file(
                    "https://downloads.example.test/asset.bin",
                    str(destination),
                    expected_sha256="0" * 64,
                    allowed_hosts={"downloads.example.test"},
                    max_bytes=1_024,
                )
            self.assertEqual(destination.read_bytes(), b"existing")

        with self.assertRaises(ValueError):
            download_file(
                "http://downloads.example.test/asset.bin",
                "unused.bin",
                expected_sha256="0" * 64,
                allowed_hosts={"downloads.example.test"},
                max_bytes=1_024,
            )

    def test_bridge_scaffolder_creates_destination_with_templates(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            destination = Path(temp_dir) / "nested" / "bridge"
            scaffold_bridge(BridgeLanguage.LUA, str(destination))
            self.assertTrue((destination / "llm_client.lua").is_file())
            self.assertTrue((destination / "tool_server.lua").is_file())
            self.assertTrue((destination / "rpc_api.lua").is_file())

    def test_dataset_zip_extraction_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            root = Path(temp_dir)
            archive_path = root / "dataset.zip"
            destination = root / "extract"
            destination.mkdir()
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("../escape.txt", "untrusted")

            with self.assertRaises(ValueError):
                CodexDataLoader()._extract_archive_safely(
                    str(archive_path), str(destination)
                )
            self.assertFalse((root / "escape.txt").exists())


if __name__ == "__main__":
    unittest.main()
