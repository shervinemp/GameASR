import zmq
import json
import time
import os


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
    def __init__(self, endpoint="tcp://0.0.0.0:8080", auth_token=None):
        self.endpoint = endpoint
        self.auth_token = auth_token
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)

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
                    "error": {"code": -32000, "message": str(e)},
                    "id": None,
                }
                self.socket.send_json(error_response)

    def _handle_request(self, request):
        if self.auth_token and request.get("auth_token") != self.auth_token:
            return {
                "jsonrpc": "2.0",
                "error": {"code": -32001, "message": "Authentication failed"},
                "id": request.get("id"),
            }

        method_name = request.get("method")
        params = request.get("params", {})

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
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "error": {"code": -32602, "message": str(e)},
                "id": request.get("id"),
            }


if __name__ == "__main__":
    # Example usage
    # You can run this script and it will listen for requests.
    # To test authentication, set the AUTH_TOKEN environment variable.
    auth_token = os.environ.get("TOOLS_AUTH_TOKEN")
    server = ToolServer(auth_token=auth_token)
    server.start()
