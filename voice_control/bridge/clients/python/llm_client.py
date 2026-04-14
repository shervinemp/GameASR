import zmq
import json


class LLMClient:
    """
    A Python client for the LLMServer.
    """

    def __init__(self, endpoint="tcp://0.0.0.0:8000", auth_token=None):
        self.endpoint = endpoint
        self.auth_token = auth_token
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)

    def connect(self):
        """Connects to the server."""
        self.socket.connect(self.endpoint)
        print(f"[LLMClient] Connected to {self.endpoint}")

    def disconnect(self):
        """Disconnects from the server."""
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.close()
        self.context.term()
        print("[LLMClient] Disconnected")

    def _request(self, method, params):
        request = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": 1,  # Simple ID for example
        }
        if self.auth_token:
            request["auth_token"] = self.auth_token

        self.socket.send_json(request)
        response = self.socket.recv_json()

        if "error" in response:
            raise Exception(f"RPC Error: {response['error']}")

        return response.get("result")

    def query(self, content, role="user"):
        """Sends a query to the LLM."""
        return self._request("query", {"content": content, "role": role})


if __name__ == "__main__":
    # Example usage
    client = LLMClient()
    client.connect()
    try:
        response = client.query("Hello, world!")
        print("Response from LLM:", response)
    except Exception as e:
        print(e)
    finally:
        client.disconnect()
