from functools import partial
from typing import List
import zmq.asyncio

from ..llm.tools import Tool


class ToolClient:
    """
    A dynamic client that configures its available methods based on
    a provided api specification.
    """

    def __init__(self, endpoint: str, auth_token: str = None):
        self.endpoint = endpoint
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(endpoint)
        self.auth_token = auth_token

    def from_spec(self, spec_dict: dict) -> List[Tool]:
        tools = []
        for method in spec_dict["methods"]:
            t_ = Tool.from_dict(method)
            t_.callback = partial(self.call_tool_async, t_.name)
            tools.append(t_)
        return tools

    async def call_tool_async(self, tool_name: str, **kwargs):
        """The generic handler for all RPC calls."""
        request = {"method": tool_name, "params": kwargs}
        if self.auth_token:
            request["auth_token"] = self.auth_token
        await self.socket.send_json(request)

        # Add timeout handling to prevent infinite hangs
        if await self.socket.poll(3000) != 0:
            return await self.socket.recv_json()
        else:
            # Cleanup socket on timeout to reset state machine
            self.socket.close()
            self.socket = self.context.socket(zmq.REQ)
            self.socket.connect(self.endpoint)
            return {"error": "Game engine RPC timeout"}
