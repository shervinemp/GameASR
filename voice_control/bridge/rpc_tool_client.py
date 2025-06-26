import zmq
import json

from ..llm.tool_client import Tool, ToolClient


class RpcToolClient(ToolClient):
    """
    A dynamic RPC client that configures its available methods based on
    a provided api_spec.json file path.
    """

    def __init__(self, spec_path: str, host="localhost", port=5556):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{host}:{port}")

        with open(spec_path, "r") as f:
            self.spec = json.load(f)

    def _call_tool(self, tool: Tool, **kwargs):
        """The generic handler for all RPC calls."""
        request = {"method": tool.name, "params": kwargs}
        self.socket.send_json(request)
        response = self.socket.recv_json()
        return response
