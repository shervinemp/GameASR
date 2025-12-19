import re
import json
import os
import argparse
from abc import ABC, abstractmethod


class BaseParser(ABC):
    """Abstract base class for language-specific API parsers."""

    @abstractmethod
    def parse(self, content: str) -> list:
        """Parses file content and returns a list of function specs."""
        pass


class LuaParser(BaseParser):
    """Parses Lua files for functions exposed on a specific table using LDoc comments."""

    def parse(self, content: str) -> list:
        """
        Finds functions documented with LDoc style comments:
        --[[
            Description...
            @param name (type): description
        ]]
        function rpc_api.method_name(...)
        """
        # Pattern to match LDoc comment block followed by function definition
        # Group 1: Comment content
        # Group 2: Function name
        pattern = re.compile(
            r"--\[\[(.*?)\]\]\s*function\s+rpc_api\.(\w+)",
            re.DOTALL
        )

        functions = []
        matches = pattern.finditer(content)

        for match in matches:
            comment_block = match.group(1)
            func_name = match.group(2)

            # Parse parameters from the comment block
            params = []
            param_matches = re.finditer(r"@param\s+(\w+)", comment_block)
            for pm in param_matches:
                params.append(pm.group(1))

            functions.append({"name": func_name, "params": params})

        return functions


# --- Future parsers can be added here to support other languages ---
# class PythonParser(BaseParser): ...
# class GDScriptParser(BaseParser): ...

PARSERS = {
    "lua": LuaParser,
    # "python": PythonParser,
}


def generate_spec(source_path: str, output_path: str, lang: str):
    """
    Generates a JSON API spec file from a source file using the specified language parser.
    """
    if lang not in PARSERS:
        print(
            f"Error: Unsupported language '{lang}'. Available: {list(PARSERS.keys())}"
        )
        return

    if not os.path.exists(source_path):
        print(f"Error: Source file not found at '{source_path}'")
        return

    print(f"Parsing '{source_path}' using '{lang}' parser...")
    with open(source_path, "r") as f:
        content = f.read()

    parser = PARSERS[lang]()
    functions = parser.parse(content)
    api_spec = {"functions": functions}

    with open(output_path, "w") as f:
        json.dump(api_spec, f, indent=4)

    print(f"API specification generated successfully at '{output_path}'")
    print(f"Found {len(functions)} functions.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate an API spec file from a client's source code.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--source",
        type=str,
        default=os.path.join("lua_client_example", "rpc_api.lua"),
        help="Path to the source file containing API definitions.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="api_spec.json",
        help="Path to save the output JSON API spec. Should be at the project root.",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="lua",
        choices=PARSERS.keys(),
        help="The source language to parse.",
    )
    args = parser.parse_args()
    generate_spec(args.source, args.output, args.lang)
