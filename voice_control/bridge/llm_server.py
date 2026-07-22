from collections import deque
import hmac
import ipaddress
import json
import threading
import time

from ..common.utils import get_logger
from ..exceptions import VoiceControlError
from ..llm import LLMProviders, Session
from ..common.config import config


class LLMService:
    rpc_methods = frozenset({"query"})
    max_content_chars = 16_384
    max_response_chars = 65_536

    def __init__(self, session: Session):
        if not isinstance(session, Session):
            raise VoiceControlError("Session must be an instance of Session class.")
        self.session = session

    def query(self, content: str, role: str = "user") -> str:
        # ASVS 2.2.1 / 8.2.1: external consumers can submit user content only;
        # they cannot manufacture system, assistant, or tool messages.
        if role != "user":
            raise VoiceControlError("Only the user role is accepted.")
        if not isinstance(content, str):
            raise VoiceControlError("Content must be a string.")
        if not content.strip():
            raise VoiceControlError("Content must not be empty.")
        if len(content) > self.max_content_chars:
            raise VoiceControlError(
                f"Content exceeds the {self.max_content_chars}-character limit."
            )

        response_parts = []
        response_length = 0
        try:
            for part in self.session(content):
                response_length += len(part)
                if response_length > self.max_response_chars:
                    raise VoiceControlError(
                        "Model response exceeded the configured limit."
                    )
                response_parts.append(part)
        except VoiceControlError:
            # Rollback: remove the user message that self.session() already
            # added to conversation history.
            conv = self.session.conversation
            if conv.visible_count() > 0:
                last = conv._get_raw_message(-1)
                if last.role.value == "user":
                    conv._messages.pop()
                    conv._token_counts.pop()
            raise
        return "".join(response_parts)


class LLMServer:
    def __init__(
        self,
        service_api,
        endpoint: str,
        auth_token: str = None,
        *,
        max_request_bytes: int = 65_536,
        requests_per_minute: int = 60,
    ):
        self.logger = get_logger(__name__)
        self.endpoint = endpoint
        self.service_api = service_api
        self.auth_token = auth_token
        self.max_request_bytes = max_request_bytes
        self.requests_per_minute = requests_per_minute
        self.context = None
        self.socket = None
        self._worker_thread = None
        self._running = False
        self._request_times = deque()
        self._rate_limit_lock = threading.Lock()
        self._allowed_methods = frozenset(
            getattr(service_api, "rpc_methods", ())
        )

        if not self._allowed_methods:
            raise VoiceControlError("The RPC service must declare an explicit rpc_methods allowlist.")
        if not isinstance(max_request_bytes, int) or not 1_024 <= max_request_bytes <= 1_048_576:
            raise VoiceControlError("max_request_bytes must be between 1024 and 1048576.")
        if not isinstance(requests_per_minute, int) or not 1 <= requests_per_minute <= 10_000:
            raise VoiceControlError("requests_per_minute must be between 1 and 10000.")

        # ASVS 6.3.4 / 13.1: local-only operation may omit a token. Any TCP
        # listener reachable beyond loopback fails closed with a strong token.
        if auth_token is not None and (
            not isinstance(auth_token, str)
            or len(auth_token.strip()) < 32
        ):
            raise VoiceControlError(
                "Configured RPC authentication tokens must contain at least "
                "32 non-blank characters."
            )
        if not self._is_loopback_endpoint(endpoint):
            if auth_token is None:
                raise VoiceControlError(
                    "Non-loopback RPC endpoints require an authentication token "
                    "of at least 32 characters."
                )
        self.logger.info(f"LLM Server will bind to {self.endpoint}")
        if self.auth_token:
            self.logger.info("LLM Server authentication is enabled.")

    def _is_authenticated(self, request: dict) -> bool:
        if not self.auth_token:
            return True
        candidate = request.get("auth_token")
        if not isinstance(candidate, str):
            return False
        # ASVS 6.3: avoid leaking token information through comparison timing.
        return hmac.compare_digest(candidate, self.auth_token)

    @staticmethod
    def _is_loopback_endpoint(endpoint: str) -> bool:
        if not isinstance(endpoint, str):
            return False
        if endpoint.startswith("ipc://") or endpoint.startswith("inproc://"):
            return True
        if not endpoint.startswith("tcp://"):
            return False

        authority = endpoint[6:].rsplit(":", 1)[0].strip("[]").lower()
        if authority == "localhost":
            return True
        try:
            return ipaddress.ip_address(authority).is_loopback
        except ValueError:
            return False

    def _is_rate_limited(self) -> bool:
        # ASVS 2.4.1 / 15.2.2: bound access to expensive model inference.
        now = time.monotonic()
        cutoff = now - 60.0
        with self._rate_limit_lock:
            while self._request_times and self._request_times[0] <= cutoff:
                self._request_times.popleft()
            if len(self._request_times) >= self.requests_per_minute:
                return True
            self._request_times.append(now)
            return False

    def _dispatch_method(self, method_name: str, params: dict):
        if method_name not in self._allowed_methods:
            raise VoiceControlError("Method not found")
        method_func = getattr(self.service_api, method_name, None)

        if method_func is None or not callable(method_func):
            raise VoiceControlError("Method not found")
        if not isinstance(params, dict):
            raise VoiceControlError("Parameters must be a dictionary.")

        return method_func(**params)

    def _handle_request(self, request_body_str: str):
        response_obj = {"jsonrpc": "2.0"}
        request_id = None

        try:
            if not isinstance(request_body_str, str):
                raise VoiceControlError("Invalid request encoding.")
            if len(request_body_str.encode("utf-8")) > self.max_request_bytes:
                raise VoiceControlError("Request exceeds the configured size limit.")

            request = json.loads(request_body_str)
            # ASVS 1.5.2 / 2.2.1: accept only the documented JSON object shape.
            if not isinstance(request, dict):
                raise VoiceControlError("Request must be a JSON object.")
            allowed_fields = {"jsonrpc", "id", "method", "params", "auth_token"}
            if set(request) - allowed_fields:
                raise VoiceControlError("Request contains unsupported fields.")
            request_id = request.get("id")
            if not isinstance(request_id, (str, int, type(None))):
                raise VoiceControlError("Request id must be a string, integer, or null.")
            if isinstance(request_id, str) and len(request_id) > 128:
                raise VoiceControlError("Request id is too long.")
            response_obj["id"] = request_id

            if request.get("jsonrpc") != "2.0" or "method" not in request:
                raise VoiceControlError("Invalid JSON-RPC request format.")

            if not self._is_authenticated(request):
                self.logger.warning("RPC authentication failed.")
                response_obj["error"] = {
                    "code": -32001,
                    "message": "Authentication failed",
                }
                return json.dumps(response_obj)

            if self._is_rate_limited():
                self.logger.warning("RPC rate limit exceeded.")
                response_obj["error"] = {
                    "code": -32002,
                    "message": "Rate limit exceeded",
                }
                return json.dumps(response_obj)

            method_name = request["method"]
            params = request.get("params", {})

            try:
                result = self._dispatch_method(method_name, params)
                response_obj["result"] = result
            except VoiceControlError as e:
                msg = str(e)
                if "Method not found" in msg:
                    response_obj["error"] = {
                        "code": -32601,
                        "message": "Method not found",
                    }
                else:
                    self.logger.warning("Invalid RPC parameters: %s", e)
                    response_obj["error"] = {
                        "code": -32602,
                        "message": "Invalid method parameters",
                    }
            except Exception as e:
                # ASVS 16.5.1: keep internal exception details in logs only.
                response_obj["error"] = {
                    "code": -32000,
                    "message": "Internal server error",
                }
                self.logger.error(
                    f"Error executing method '{method_name}': {e}"
                )

        except json.JSONDecodeError:
            response_obj["error"] = {
                "code": -32700,
                "message": "Parse error: invalid JSON",
            }
            response_obj["id"] = None
        except ValueError as e:
            response_obj["error"] = {
                "code": -32600,
                "message": f"Invalid request: {str(e)}",
            }
            response_obj["id"] = None
        except Exception as e:
            response_obj["error"] = {
                "code": -32000,
                "message": "Internal server error",
            }
            response_obj["id"] = None
            self.logger.error(
                f"Unexpected error during request processing: {e}"
            )

        return json.dumps(response_obj)

    def _worker_loop(self):
        import asyncio
        import zmq.asyncio

        async def _async_worker_loop():
            self.context = zmq.asyncio.Context()
            self.socket = self.context.socket(zmq.REP)
            self.socket.setsockopt(zmq.MAXMSGSIZE, self.max_request_bytes)
            self.socket.bind(self.endpoint)
            self.logger.info(f"LLM Server bound to {self.endpoint}")

            self.logger.info("RPC Server worker thread started.")
            self._running = True
            try:
                while self._running:
                    if await self.socket.poll(1000) & zmq.POLLIN:
                        raw_message = await self.socket.recv(zmq.DONTWAIT)
                        if len(raw_message) > self.max_request_bytes:
                            response = json.dumps(
                                {
                                    "jsonrpc": "2.0",
                                    "id": None,
                                    "error": {
                                        "code": -32600,
                                        "message": "Request exceeds the configured size limit",
                                    },
                                }
                            )
                            await self.socket.send_string(response)
                            continue
                        try:
                            message = raw_message.decode("utf-8", errors="strict")
                        except UnicodeDecodeError:
                            await self.socket.send_json(
                                {
                                    "jsonrpc": "2.0",
                                    "id": None,
                                    "error": {
                                        "code": -32700,
                                        "message": "Request must be valid UTF-8 JSON",
                                    },
                                }
                            )
                            continue

                        # Handle the request in a separate thread so it doesn't block other ZMQ operations
                        response = await asyncio.to_thread(self._handle_request, message)

                        await self.socket.send_string(response)
                        self.logger.debug("RPC response sent.")
            except KeyboardInterrupt:
                self.logger.info(
                    "RPC Server shutting down due to KeyboardInterrupt."
                )
            except Exception as e:
                self.logger.error(
                    f"Unhandled error in RPC Server worker loop: {e}"
                )
            finally:
                self._running = False
                self.socket.setsockopt(zmq.LINGER, 0)
                self.socket.close()
                self.context.term()
                self.logger.info("ZeroMQ resources cleaned up.")

        import sys
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(_async_worker_loop())

    def start(self):
        if not self._worker_thread:
            self._worker_thread = threading.Thread(
                target=self._worker_loop, daemon=True, name="RpcServerWorker"
            )
            self._worker_thread.start()
            self.logger.info("RPC Server worker thread initiated.")
        else:
            self.logger.info("RPC Server worker thread is already running.")

    def stop(self):
        if self._worker_thread and self._worker_thread.is_alive():
            self.logger.info("Attempting to stop RPC Server worker thread...")
            self._running = False
            self._worker_thread.join(timeout=2)
            if self._worker_thread.is_alive():
                self.logger.warning(
                    "RPC Server thread did not terminate gracefully within timeout."
                )
            else:
                self.logger.info(
                    "RPC Server worker thread stopped successfully."
                )
            self._worker_thread = None
        else:
            self.logger.info(
                "RPC Server worker thread is not running or already stopped."
            )


if __name__ == "__main__":
    llm_session = Session(
        LLMProviders.create(
            config.get("llm.backend"),
            config.get("llm.model"),
        )
    )
    llm_service_instance = LLMService(llm_session)

    server_endpoint = "tcp://127.0.0.1:5555"
    auth_token = config.get("rpc_server.auth_token")
    llm_server = LLMServer(
        llm_service_instance,
        server_endpoint,
        auth_token=auth_token,
        max_request_bytes=config.get("rpc_server.max_request_bytes", 65_536),
        requests_per_minute=config.get("rpc_server.requests_per_minute", 60),
    )

    llm_server.start()
    print(f"LLM Server started on {server_endpoint}. Press Ctrl+C to stop.")

    try:
        while True:
            threading.Event().wait(1)
    except KeyboardInterrupt:
        print(
            "\nMain thread received KeyboardInterrupt. Stopping LLM server..."
        )
        llm_server.stop()
        print("Application exiting.")
