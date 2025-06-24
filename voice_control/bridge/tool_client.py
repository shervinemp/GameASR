import zmq
import json
from functools import partial


class ToolClient:
    """
    A dynamic RPC client that configures its available methods based on
    a provided api_spec.json file path.
    """

    def __init__(self, spec_path: str, host="localhost", port=5556):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{host}:{port}")

        self._load_methods_from_spec(spec_path)

    def _load_methods_from_spec(self, spec_path):
        """Loads the API spec and dynamically creates methods on this object."""
        print(f"Loading API specification from: {spec_path}")
        try:
            with open(spec_path, "r") as f:
                spec = json.load(f)

            for func_info in spec.get("functions", []):
                func_name = func_info["name"]
                method = partial(self._call_rpc, func_name)
                setattr(self, func_name, method)
                print(f"  - Registered tool: {func_name}")

        except FileNotFoundError:
            print(
                f"CRITICAL ERROR: API spec file not found at '{spec_path}'. The tool client will not have any methods. Please run the generation script."
            )
        except json.JSONDecodeError:
            print(
                f"CRITICAL ERROR: Could not decode API spec file at '{spec_path}'. Check for syntax errors in the JSON."
            )

    def _call_rpc(self, method_name, **kwargs):
        """The generic handler for all RPC calls."""
        print(f"Calling remote tool '{method_name}' with params: {kwargs}")
        request = {"method": method_name, "params": kwargs}
        self.socket.send_json(request)
        response = self.socket.recv_json()
        return response


class ToolCaller:
    """Callable class that dispatches calls to the ToolClient instance."""

    def __init__(self, tool_client):
        self.tool_client = tool_client

    def __call__(self, tool_name, **kwargs):
        """Calls the method on the tool_client if it exists. Fixes previous NameError."""
        if hasattr(self.tool_client, tool_name):
            rpc_method = getattr(self.tool_client, tool_name)
            return rpc_method(**kwargs)
        else:
            raise AttributeError(
                f"Tool '{tool_name}' not found. Ensure it is defined in the client's API source file and that the API spec has been regenerated."
            )
