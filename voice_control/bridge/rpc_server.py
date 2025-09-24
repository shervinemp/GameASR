import threading
import zmq
import json

from ..common.utils import get_logger
from ..llm import Session
from ..common.config import config


class LLMService:
    def __init__(self, session: Session):
        if not isinstance(session, Session):
            raise TypeError("Session must be an instance of Session class.")
        self.session = session

    def query(self, content: str, role: str) -> str:
        if not isinstance(role, str):
            raise TypeError("Role must be a string.")
        if not isinstance(content, str):
            raise TypeError("Content must be a string.")

        message = {"role": role, "content": content}
        response_parts = self.session([message])
        return "".join(response_parts)


class LLMServer:
    def __init__(self, service_api, endpoint: str, auth_token: str = None):
        self.logger = get_logger(__name__)
        self.endpoint = endpoint
        self.service_api = service_api
        self.auth_token = auth_token
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(self.endpoint)
        self._worker_thread = None
        self._running = False
        self.logger.info(f"LLM Server bound to {self.endpoint}")
        if self.auth_token:
            self.logger.info("LLM Server authentication is enabled.")

    def _is_authenticated(self, request: dict) -> bool:
        if not self.auth_token:
            return True
        return request.get("auth_token") == self.auth_token

    def _dispatch_method(self, method_name: str, params: dict):
        method_func = getattr(self.service_api, method_name, None)

        if method_func is None or not callable(method_func):
            raise ValueError(f"Method not found: {method_name}")
        if not isinstance(params, dict):
            raise TypeError("Parameters must be a dictionary.")

        return method_func(**params)

    def _handle_request(self, request_body_str: str):
        response_obj = {"jsonrpc": "2.0"}
        request_id = None

        try:
            request = json.loads(request_body_str)
            request_id = request.get("id")
            response_obj["id"] = request_id

            if request.get("jsonrpc") != "2.0" or "method" not in request:
                raise ValueError("Invalid JSON-RPC request format.")

            if not self._is_authenticated(request):
                response_obj["error"] = {"code": -32001, "message": "Authentication failed"}
                return json.dumps(response_obj)

            method_name = request["method"]
            params = request.get("params", {})

            try:
                result = self._dispatch_method(method_name, params)
                response_obj["result"] = result
            except ValueError as e:
                response_obj["error"] = {"code": -32601, "message": str(e)}
            except TypeError as e:
                response_obj["error"] = {
                    "code": -32602,
                    "message": f"Invalid params for method '{method_name}': {str(e)}",
                }
            except Exception as e:
                response_obj["error"] = {
                    "code": -32000,
                    "message": f"Server error executing '{method_name}': {str(e)}",
                }
                self.logger.error(
                    f"Error executing method '{method_name}': {e}"
                )

        except json.JSONDecodeError as e:
            response_obj["error"] = {
                "code": -32700,
                "message": f"Parse error: Invalid JSON - {str(e)}",
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
                "message": f"Unexpected server error: {str(e)}",
            }
            response_obj["id"] = None
            self.logger.error(
                f"Unexpected error during request processing: {e}"
            )

        return json.dumps(response_obj)

    def _worker_loop(self):
        self.logger.info("RPC Server worker thread started.")
        self._running = True
        try:
            while self._running:
                if self.socket.poll(100) & zmq.POLLIN:
                    message = self.socket.recv_string()
                    self.logger.debug(f"Received: {message}")
                    response = self._handle_request(message)
                    self.socket.send_string(response)
                    self.logger.debug(f"Sent: {response}")
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
            self.socket.close()
            self.context.term()
            self.logger.info("ZeroMQ resources cleaned up.")

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
    llm_session = Session()
    llm_service_instance = LLMService(llm_session)

    server_endpoint = "tcp://127.0.0.1:5555"
    auth_token = config.get("rpc_server.auth_token")
    llm_server = LLMServer(llm_service_instance, server_endpoint, auth_token=auth_token)

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
