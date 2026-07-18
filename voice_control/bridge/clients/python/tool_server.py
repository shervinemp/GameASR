import zmq
from collections import deque
import hmac
import time
import os

from voice_control.bridge.llm_server import LLMServer


# --- Dummy Tool Implementations ---
def move_player(direction):
    print(f"[ToolServer] Moving player: {direction}")
    return {"status": "success", "message": f"Moved {direction}"}


def get_player_position():
    print("[ToolServer] Getting player position")
    return {"x": 10, "y": 20}


def set_game_pause(is_paused):
    print(f"[ToolServer] Setting game pause state to: {is_paused}")
    return {"status": "success"}


def get_game_time():
    print("[ToolServer] Getting game time")
    return {"time": "12:30 PM"}


# --- RPC Method Dispatcher ---
RPC_METHODS = {
    "move_player": move_player,
    "get_player_position": get_player_position,
    "set_game_pause": set_game_pause,
    "get_game_time": get_game_time,
}


class ToolServer:
    def __init__(
        self,
        endpoint="tcp://127.0.0.1:8080",
        auth_token=None,
        *,
        max_request_bytes=65_536,
        requests_per_minute=120,
    ):
        self.endpoint = endpoint
        self.auth_token = auth_token
        self.max_request_bytes = max_request_bytes
        self.requests_per_minute = requests_per_minute
        self._request_times = deque()
        if not LLMServer._is_loopback_endpoint(endpoint):
            if not isinstance(auth_token, str) or len(auth_token) < 32:
                raise ValueError(
                    "Non-loopback tool endpoints require a token of at least 32 characters."
                )
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.setsockopt(zmq.MAXMSGSIZE, max_request_bytes)

    def start(self):
        """Starts the server loop."""
        self.socket.bind(self.endpoint)
        print(f"[ToolServer] Listening on {self.endpoint}")
        if self.auth_token:
            print("[ToolServer] Authentication is enabled.")

        while True:
            try:
                message = self.socket.recv_json()
                response = self._handle_request(message)
                self.socket.send_json(response)
            except Exception as e:
                print(f"Error handling request: {e}")
                error_response = {
                    "jsonrpc": "2.0",
                    "error": {"code": -32000, "message": "Internal server error"},
                    "id": None,
                }
                self.socket.send_json(error_response)

    def _handle_request(self, request):
        # ASVS 1.5.2 / 2.2.1: validate untrusted JSON before game dispatch.
        if not isinstance(request, dict):
            return self._error(-32600, "Invalid request")
        if len(str(request).encode("utf-8")) > self.max_request_bytes:
            return self._error(-32600, "Request too large", request.get("id"))

        supplied_token = request.get("auth_token")
        authenticated = not self.auth_token or (
            isinstance(supplied_token, str)
            and hmac.compare_digest(supplied_token, self.auth_token)
        )
        if not authenticated:
            return {
                "jsonrpc": "2.0",
                "error": {"code": -32001, "message": "Authentication failed"},
                "id": request.get("id"),
            }

        now = time.monotonic()
        while self._request_times and self._request_times[0] <= now - 60:
            self._request_times.popleft()
        if len(self._request_times) >= self.requests_per_minute:
            return self._error(-32002, "Rate limit exceeded", request.get("id"))
        self._request_times.append(now)

        method_name = request.get("method")
        params = request.get("params", {})
        if not isinstance(method_name, str) or not isinstance(params, dict):
            return self._error(-32600, "Invalid request", request.get("id"))

        method = RPC_METHODS.get(method_name)
        if not method:
            return {
                "jsonrpc": "2.0",
                "error": {"code": -32601, "message": "Method not found"},
                "id": request.get("id"),
            }

        try:
            result = method(**params)
            return {
                "jsonrpc": "2.0",
                "result": result,
                "id": request.get("id"),
            }
        except Exception:
            return {
                "jsonrpc": "2.0",
                "error": {"code": -32602, "message": "Invalid method parameters"},
                "id": request.get("id"),
            }

    @staticmethod
    def _error(code, message, request_id=None):
        return {
            "jsonrpc": "2.0",
            "error": {"code": code, "message": message},
            "id": request_id,
        }


if __name__ == "__main__":
    # Example usage
    # You can run this script and it will listen for requests.
    # To test authentication, set the AUTH_TOKEN environment variable.
    auth_token = os.environ.get("TOOLS_AUTH_TOKEN")
    server = ToolServer(auth_token=auth_token)
    server.start()
