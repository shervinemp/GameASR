from dataclasses import dataclass, field
import inspect
import re
from typing import Callable, List, Dict, Optional, Any, Type, Union, Literal, get_origin, get_args


@dataclass
class ToolCall:
    """Represents a requested tool call from the LLM."""
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Tool:
    """
    Represents a tool callable by an AI model, structured according to OpenAI's function calling schema.
    """

    @dataclass
    class Parameter:
        """
        Represents a parameter schema definition within a Tool. Corresponds to a JSON Schema object for a property.
        """

        type: str
        description: Optional[str] = None
        properties: Optional[Dict[str, "Tool.Parameter"]] = None
        enum: Optional[List[Any]] = None
        required: Optional[List[str]] = None

        @classmethod
        def from_dict(cls, data: Dict[str, Any]) -> "Tool.Parameter":
            """Recursively creates a Parameter from a dictionary."""
            return cls(
                type=data["type"],
                description=data.get("description"),
                properties=(
                    {
                        k: cls.from_dict(v)
                        for k, v in data.get("properties", {}).items()
                    }
                    if data.get("type") == "object" and data.get("properties")
                    else None
                ),
                enum=data.get("enum"),
                required=(
                    data.get("required")
                    if data.get("type") == "object"
                    else None
                ),
            )

        def to_dict(self) -> Dict[str, Any]:
            """Converts the Parameter to a dictionary compatible with from_dict."""
            d = {"type": self.type}

            if self.description:
                d["description"] = self.description

            if self.type == "object":
                if self.properties:
                    d["properties"] = {
                        k: v.to_dict() for k, v in self.properties.items()
                    }

                if self.required:
                    d["required"] = self.required

            if self.enum:
                d["enum"] = self.enum

            return d

        def __str__(self) -> str:
            return str(self.to_dict())

    name: str
    description: str
    parameters: Optional[Parameter] = field(default=None)
    callback: Optional[Callable] = None

    def __call__(self, **kwargs) -> Any:
        return self.callback(**self._parse_args(**kwargs))

    def _parse_args(self, **kwargs) -> Dict[str, Any]:
        if not self.parameters or not hasattr(self.parameters, "properties"):
            return kwargs

        casted_args = {}
        properties = self.parameters.properties

        for arg_name, arg_value in kwargs.items():
            casted_args[arg_name] = arg_value
            if arg_name in properties:
                json_type = properties[arg_name].type
                if json_type == "integer":
                    casted_args[arg_name] = int(arg_value)
                elif json_type == "number":
                    casted_args[arg_name] = float(arg_value)
                elif json_type == "boolean":
                    casted_args[arg_name] = bool(arg_value)
                elif json_type == "string":
                    casted_args[arg_name] = str(arg_value)

        return casted_args

    @staticmethod
    def from_callable(name: str, fn: Callable) -> "Tool":
        """
        Transforms a callable into a Tool instance.
        """
        doc_info = _parse_method_docstring(inspect.getdoc(fn))
        description = doc_info["description"] or "No description provided."

        signature = inspect.signature(fn)
        method_properties: Dict[str, Tool.Parameter] = {}
        required_property_names: List[str] = []

        for param_name, param_obj in signature.parameters.items():
            if param_name == "self":
                continue

            json_type = _get_json_type(param_obj.annotation)
            param_desc = doc_info["params"].get(param_name, None)

            enum_values = None
            if get_origin(param_obj.annotation) is Literal:
                enum_values = list(get_args(param_obj.annotation))

            if param_obj.default is inspect.Parameter.empty:
                required_property_names.append(param_name)

            method_properties[param_name] = Tool.Parameter(
                type=json_type,
                description=param_desc,
                enum=enum_values
            )

        tool_parameters_obj: Optional[Tool.Parameter] = None
        if method_properties:
            tool_parameters_obj = Tool.Parameter(
                type="object",
                properties=method_properties,
                required=(
                    required_property_names
                    if required_property_names
                    else None
                ),
            )

        tool = Tool(
            name=name,
            description=description,
            parameters=tool_parameters_obj,
            callback=fn,
        )

        return tool

    @staticmethod
    def from_class(cls: Type) -> Callable[..., List["Tool"]]:
        """
        A class decorator that transforms the decorated class into a callable which, when instantiated,
        returns a list of Tool instances.

        Each tool's callback is a bound method of the internally created instance of the original class.
        Only public methods (not starting with '_') are considered. Method descriptions and parameter details
        are parsed from docstrings.
        """

        class ToolFactory:
            _original_class = cls

            def __call__(self, *args, **kwargs) -> List["Tool"]:
                instance = self._original_class(*args, **kwargs)
                tools: List["Tool"] = []
                for name, member in inspect.getmembers(
                    instance, predicate=inspect.ismethod
                ):
                    if name.startswith("_"):  # Skip private methods
                        continue

                    tool = self.from_callable(name, member)
                    tools.append(tool)

                return tools

        return ToolFactory

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the Tool instance to a dictionary that conforms to the OpenAI
        tool schema, which is used by llama-cpp-python.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": (
                    self.parameters.to_dict()
                    if self.parameters
                    else {"type": "object", "properties": {}}
                ),
            },
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Tool":
        """
        Creates a Tool instance from a dictionary definition conforming to the
        OpenAI tool schema.
        """
        function_data = d.get("function", {})
        parameters_data = function_data.get("parameters")

        return cls(
            name=function_data.get("name"),
            description=function_data.get("description"),
            parameters=(
                cls.Parameter.from_dict(parameters_data)
                if parameters_data
                else None
            ),
        )


def _parse_method_docstring(docstring: Optional[str]) -> Dict[str, Any]:
    """
    Parses a method's docstring to extract overall description, parameter descriptions,
    and return description. Supports reStructuredText-like ':param:' and ':returns:' and
    simple Google-style 'Args:', 'Returns:'.
    """
    if not docstring:
        return {"description": "", "params": {}, "returns": ""}

    lines = docstring.strip().split("\n")
    main_description_lines = []
    param_descriptions = {}
    returns_description = ""

    in_params_section = False
    in_returns_section = False

    param_re_rst = re.compile(r"^\s*:param\s+([a-zA-Z0-9_]+):(.*)$")
    returns_re_rst = re.compile(r"^\s*:returns:\s*(.*)$")
    param_re_google = re.compile(r"^\s*(\w+)\s*(?:\([^\)]+\))?:\s*(.*)$")
    args_header_re = re.compile(r"^\s*Args:\s*$")
    returns_header_re = re.compile(r"^\s*Returns:\s*$")

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if args_header_re.match(line):
            in_params_section = True
            in_returns_section = False
            i += 1
            continue
        elif returns_header_re.match(line):
            in_returns_section = True
            in_params_section = False
            i += 1
            continue

        if in_params_section:
            match_rst = param_re_rst.match(line)
            match_google = param_re_google.match(line)
            if match_rst:
                param_name, param_desc = match_rst.groups()
                param_descriptions[param_name.strip()] = param_desc.strip()
            elif match_google and not line.strip().startswith(":"):
                param_name, param_desc = match_google.groups()
                param_descriptions[param_name.strip()] = param_desc.strip()
            elif line and not line.startswith(" "):
                in_params_section = False
                main_description_lines.append(line)
            else:
                if param_descriptions and line.startswith(" "):
                    last_param_name = list(param_descriptions.keys())[-1]
                    param_descriptions[last_param_name] += " " + line.strip()

        elif in_returns_section:
            match_rst = returns_re_rst.match(line)
            if match_rst:
                returns_description = match_rst.group(1).strip()
            elif line and not line.startswith(" "):
                in_returns_section = False
                main_description_lines.append(line)
            else:
                if returns_description and line.startswith(" "):
                    returns_description += " " + line.strip()

        else:
            main_description_lines.append(line)

        i += 1

    main_desc = "\n".join(main_description_lines).strip()
    main_desc = inspect.cleandoc(main_desc)

    return {
        "description": main_desc,
        "params": param_descriptions,
        "returns": returns_description,
    }


def _get_json_type(py_type) -> str:
    """Converts a Python type hint to a JSON schema type string."""
    origin = get_origin(py_type)

    if origin is Union:
        non_none_types = [
            arg for arg in get_args(py_type) if arg is not type(None)
        ]
        if non_none_types:
            py_type = non_none_types[0]
            origin = get_origin(py_type)
        else:
            return "string"

    if origin is Literal:
        args = get_args(py_type)
        if args:
            return _get_json_type(type(args[0]))
        return "string"

    if py_type is int:
        return "integer"
    if py_type is float:
        return "number"
    if py_type is str:
        return "string"
    if py_type is bool:
        return "boolean"

    if origin is list:
        return "array"
    if origin is dict:
        return "object"

    return "string"
