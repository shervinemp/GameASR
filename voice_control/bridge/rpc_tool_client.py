from functools import partial
from typing import List
import zmq

from ..llm.tools import Tool


class RpcToolClient:
    """
    A dynamic RPC client that configures its available methods based on
    a provided api specification.
    """

    def __init__(self, host="localhost", port=5556):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{host}:{port}")

    def from_spec(self, spec_dict: dict) -> List[Tool]:
        tools = []
        for method in spec_dict["methods"]:
            t_ = Tool.from_dict(method)
            t_.callback = partial(self._call_tool, t_.name)
            tools.append(t_)
        return tools

    def _call_tool(self, tool_name: str, **kwargs):
        """The generic handler for all RPC calls."""
        request = {"method": tool_name, "params": kwargs}
        self.socket.send_json(request)
        response = self.socket.recv_json()
        return response
