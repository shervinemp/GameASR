from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from functools import partial


@dataclass
class Parameter:
    type: str
    description: str = None
    properties: dict[str, "Parameter"] | None = field(default_factory=dict)
    enum: list | None = None
    required: list | None = None

    def __str__(self):
        d = {
            k: (str(v) if isinstance(v, Parameter) else v)
            for k, v in asdict(self).items()
        }
        return str(d)


@dataclass
class Tool:
    name: str
    description: str
    parameters: list[Parameter] = field(default_factory=list)

    def __str__(self):
        tool_dict = asdict(self)
        return f"<tool> {tool_dict} </tool>"


class ToolClient(ABC):
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def __call__(self, method_name, **kwargs):
        self._call_tool(self._tools[method_name], **kwargs)

    def _register_tool(self, tool: Tool):
        key = tool.name
        self._tools[key] = tool
        method = partial(self._call_tool, tool)
        setattr(self, key, method)

    @abstractmethod
    def _call_tool(self, tool: Tool, **kwargs): ...
