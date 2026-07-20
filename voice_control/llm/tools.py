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
            if not isinstance(data, dict):
                raise TypeError("Parameter definition must be an object.")
            json_type = data.get("type")
            if json_type not in {
                "array", "boolean", "integer", "number", "object", "string"
            }:
                raise ValueError("Parameter definition has an unsupported type.")
            properties = data.get("properties", {})
            if not isinstance(properties, dict):
                raise TypeError("Parameter properties must be an object.")
            required = data.get("required")
            if required is not None and (
                not isinstance(required, list)
                or not all(isinstance(item, str) for item in required)
            ):
                raise TypeError("Required parameters must be a string array.")
            if required and not set(required).issubset(properties):
                raise ValueError("Required parameters must exist in properties.")
            return cls(
                type=json_type,
                description=data.get("description"),
                properties=(
                    {
                        k: cls.from_dict(v)
                        for k, v in properties.items()
                    }
                    if json_type == "object" and properties
                    else None
                ),
                enum=data.get("enum"),
                required=(
                    required
                    if json_type == "object"
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
    instruction: Optional[str] = None

    def __call__(self, **kwargs) -> Any:
        return self.callback(**self._parse_args(**kwargs))

    def _parse_args(self, **kwargs) -> Dict[str, Any]:
        if not self.parameters or not hasattr(self.parameters, "properties"):
            return kwargs

        casted_args = {}
        properties = self.parameters.properties
        properties = properties or {}
        unexpected = set(kwargs) - set(properties)
        if unexpected:
            raise TypeError(
                f"Unexpected tool arguments: {', '.join(sorted(unexpected))}"
            )
        missing = set(self.parameters.required or ()) - set(kwargs)
        if missing:
            raise TypeError(
                f"Missing required tool arguments: {', '.join(sorted(missing))}"
            )

        for arg_name, arg_value in kwargs.items():
            casted_args[arg_name] = arg_value
            if arg_name in properties:
                json_type = properties[arg_name].type
                if json_type == "integer":
                    casted_args[arg_name] = int(arg_value)
                elif json_type == "number":
                    casted_args[arg_name] = float(arg_value)
                elif json_type == "boolean":
                    if isinstance(arg_value, bool):
                        casted_args[arg_name] = arg_value
                    elif isinstance(arg_value, str) and arg_value.lower() in {
                        "true", "false"
                    }:
                        casted_args[arg_name] = arg_value.lower() == "true"
                    else:
                        raise TypeError(
                            f"Tool argument '{arg_name}' must be a boolean."
                        )
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
        instruction = doc_info.get("instruction") or None

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
            instruction=instruction,
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
        if not isinstance(d, dict):
            raise TypeError("Tool definition must be an object.")

        # Accept the canonical OpenAI function schema and the legacy flat
        # generator schema. ASVS 1.5.2 / 2.2.1: deserialize only the expected
        # structure and validate identifiers before they become dispatch keys.
        if "type" in d and d["type"] != "function":
            raise ValueError("Tool type must be 'function'.")
        function_data = d.get("function") or d
        if not isinstance(function_data, dict):
            raise TypeError("Tool function definition must be an object.")
        name = function_data.get("name")
        if not isinstance(name, str) or not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]{0,63}", name
        ):
            raise ValueError("Tool name must be a valid 1-64 character identifier.")

        parameters_data = function_data.get("parameters")
        legacy_params = function_data.get("params")
        if parameters_data is None and isinstance(legacy_params, list):
            properties = {}
            for param_name in legacy_params:
                if not isinstance(param_name, str) or not re.fullmatch(
                    r"[A-Za-z_][A-Za-z0-9_]{0,63}", param_name
                ):
                    raise ValueError("Tool parameter names must be valid identifiers.")
                properties[param_name] = {"type": "string"}
            parameters_data = {
                "type": "object",
                "properties": properties,
                "required": list(properties),
            }
        elif parameters_data is None and legacy_params is not None:
            raise TypeError("Legacy tool params must be an array.")

        return cls(
            name=name,
            description=function_data.get("description") or "No description provided.",
            parameters=(
                cls.Parameter.from_dict(parameters_data)
                if parameters_data
                else None
            ),
            instruction=function_data.get("instruction"),
        )


def _parse_method_docstring(docstring: Optional[str]) -> Dict[str, Any]:
    """
    Parses a method's docstring to extract overall description, parameter descriptions,
    return description, and usage instruction. Supports reStructuredText-like
    ':param:', ':returns:', ':instruction:' and simple Google-style
    'Args:', 'Returns:', 'Instruction:'.
    """
    if not docstring:
        return {"description": "", "params": {}, "returns": "", "instruction": ""}

    lines = docstring.strip().split("\n")
    main_description_lines = []
    param_descriptions = {}
    returns_description = ""
    instruction = ""

    in_params_section = False
    in_returns_section = False
    in_instruction_section = False

    param_re_rst = re.compile(r"^\s*:param\s+([a-zA-Z0-9_]+):(.*)$")
    returns_re_rst = re.compile(r"^\s*:returns:\s*(.*)$")
    instruction_re_rst = re.compile(r"^\s*:instruction:\s*(.*)$")
    param_re_google = re.compile(r"^\s*(\w+)\s*(?:\([^\)]+\))?:\s*(.*)$")
    args_header_re = re.compile(r"^\s*Args:\s*$")
    returns_header_re = re.compile(r"^\s*Returns:\s*$")
    instruction_header_re = re.compile(r"^\s*Instruction:\s*$")

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if args_header_re.match(line):
            in_params_section = True
            in_returns_section = False
            in_instruction_section = False
            i += 1
            continue
        elif returns_header_re.match(line):
            in_returns_section = True
            in_params_section = False
            in_instruction_section = False
            i += 1
            continue
        elif instruction_header_re.match(line):
            in_instruction_section = True
            in_params_section = False
            in_returns_section = False
            i += 1
            continue

        match_rst_inst = instruction_re_rst.match(line)
        if match_rst_inst:
            instruction = match_rst_inst.group(1).strip()
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

        elif in_instruction_section:
            if line:
                instruction = (instruction + " " + line).strip()
            else:
                in_instruction_section = False

        else:
            main_description_lines.append(line)

        i += 1

    main_desc = "\n".join(main_description_lines).strip()
    main_desc = inspect.cleandoc(main_desc)

    return {
        "description": main_desc,
        "params": param_descriptions,
        "returns": returns_description,
        "instruction": instruction,
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
