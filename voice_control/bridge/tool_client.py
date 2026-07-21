from functools import partial
from typing import List
import zmq.asyncio

from ..llm.tools import Tool, ToolResult


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
        methods = spec_dict.get("methods") or spec_dict.get("functions", [])
        for method in methods:
            t_ = Tool.from_dict(method)
            t_.callback = partial(self.call_tool_async, t_.name)
            tools.append(t_)
        return tools

    async def call_tool_async(self, tool_name: str, max_retries: int = 3, **kwargs) -> ToolResult:
        """The generic handler for all RPC calls."""
        last_error = None
        for attempt in range(max_retries):
            request = {"method": tool_name, "params": kwargs}
            if self.auth_token:
                request["auth_token"] = self.auth_token
            try:
                await self.socket.send_json(request)
                if await self.socket.poll(3000 * (attempt + 1)) != 0:
                    return ToolResult(result=await self.socket.recv_json())
                else:
                    raise TimeoutError(f"Timeout on attempt {attempt + 1}")
            except Exception as e:
                last_error = str(e)
                self.socket.setsockopt(zmq.LINGER, 0)
                self.socket.close()
                self.socket = self.context.socket(zmq.REQ)
                self.socket.connect(self.endpoint)

        return ToolResult(result={"error": f"Game engine RPC failed after {max_retries} attempts: {last_error}"})

    def close(self):
        """Release ZMQ resources."""
        try:
            self.socket.setsockopt(zmq.LINGER, 0)
            self.socket.close()
        except Exception:
            pass
        try:
            self.context.term()
        except Exception:
            pass

    def __del__(self):
        self.close()
