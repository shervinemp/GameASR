import threading
import zmq
import json
import sys

from ..common.utils import get_logger
from ..llm.core import LLMCore


class LLMService:
    def __init__(self, llm: LLMCore):
        self.llm = llm

    def set_contexts(self, contexts: list[str]) -> None:
        if not isinstance(contexts, list):
            raise TypeError("Contexts must be a list of strings.")
        self.llm.contexts = contexts

    def query(self, content: str, role: str) -> str:
        if not isinstance(role, str):
            raise TypeError("Role must be a string.")
        if not isinstance(content, str):
            raise TypeError(content, str)

        message = {"role": role, "content": content}
        response = self.llm.generate_response([message])

        return response


class RpcServer:
    def __init__(self, service_api, endpoint: str):
        self.logger = get_logger(__name__)

        self.endpoint = endpoint
        self.service_api = service_api
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(self.endpoint)
        self._worker_thread = None
        self.logger.info(f"[RpcServer] Bound to {self.endpoint}")

    def start(self):
        """Starts the server worker thread."""
        if not self._worker_thread:
            self._worker_thread = threading.Thread(
                target=self._worker_loop,
                daemon=True,  # Daemon thread exits when main program exits
            )
            self._worker_thread.start()
            self.logger.info("Server worker thread started.")

    def _dispatch_method(self, method_name: str, params: dict):
        # Dynamically get the method from the service api
        method_func = getattr(self.service_api, method_name, None)

        if method_func is None or not callable(method_func):
            raise ValueError(f"Method not found: {method_name}")

        # Call the method with unpacked parameters
        return method_func(**params)

    def handle_request(self, request_body_str: str):
        response_obj = {"jsonrpc": "2.0"}
        try:
            request = json.loads(request_body_str)

            if request.get("jsonrpc") != "2.0" or "method" not in request:
                raise ValueError("Invalid JSON-RPC request format.")

            method_name = request["method"]
            params = request.get("params", {})
            request_id = request.get("id")

            response_obj["id"] = request_id

            try:
                result = self._dispatch_method(method_name, params)
                response_obj["result"] = result
            except ValueError as e:  # Catch method not found from _dispatch_method
                response_obj["error"] = {
                    "code": -32601,
                    "message": str(e),
                }  # Method not found
            except TypeError as e:  # Catch argument mismatch (Invalid params)
                response_obj["error"] = {
                    "code": -32602,
                    "message": f"Invalid params for method '{method_name}': {str(e)}",
                }
            except Exception as e:  # Catch any other errors from the service method
                response_obj["error"] = {
                    "code": -32000,
                    "message": f"Server error: {str(e)}",
                }
                self.logger.error(
                    f"[Server] Error executing method '{method_name}': {e}",
                    file=sys.stderr,
                )

        except json.JSONDecodeError as e:
            response_obj["error"] = {
                "code": -32700,
                "message": f"Parse error: Invalid JSON received - {str(e)}",
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
                "message": f"An unexpected server error occurred: {str(e)}",
            }
            response_obj["id"] = None

            self.logger.error(
                f"[Server] Unexpected error during request processing: {e}",
                file=sys.stderr,
            )

        return json.dumps(response_obj)

    def _worker_loop(self):
        try:
            while True:
                message = self.socket.recv_string()
                self.logger.debug(f"\n[Server] Received: {message}")

                response = self.handle_request(message)

                self.socket.send_string(response)
                self.logger.debug(f"[Server] Sent: {response}")

        except KeyboardInterrupt:
            self.logger.info("\n[Server] Shutting down.")
        except Exception as e:
            self.logger.error(
                f"\n[Server] An unhandled error occurred: {e}", file=sys.stderr
            )
        finally:
            self.socket.close()
            self.context.term()
            self.logger.debug("[Server] ZeroMQ resources cleaned up.")

    def stop(self):
        """Sends shutdown signal to the server worker and waits for it to finish."""
        if self._worker_thread and self._worker_thread.is_alive():
            self.logger.info("Sending shutdown signal to server thread...")
            self.transcription_queue.put(None)  # Send sentinel
            self._worker_thread.join(timeout=5)  # Wait for thread to finish
            if self._worker_thread.is_alive():
                self.logger.warning(
                    "Server thread did not terminate gracefully within timeout."
                )
