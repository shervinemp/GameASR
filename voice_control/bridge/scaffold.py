import argparse
import os
import shutil
import sys
from enum import Enum

from voice_control.common.utils import get_logger, setup_logging


class BridgeLanguage(Enum):
    LUA = "lua"
    PYTHON = "python"
    GDSCRIPT = "gdscript"
    CSHARP = "csharp"
    CPP = "cpp"
    JAVASCRIPT = "javascript"


def get_bridge_source_path(language: BridgeLanguage) -> str:
    """
    Returns the absolute path to the source directory of the bridge files
    for a given language using importlib.resources.
    """
    import importlib.resources as pkg_resources

    try:
        # Get the path to voice_control.bridge.clients.[language]
        package_name = f"voice_control.bridge.clients.{language.value}"

        # We need an actual file to resolve the directory path
        # Assuming there is an __init__.py or a specific file we can anchor to,
        # but files() returns a Traversable which has a path if it's a local file.

        if sys.version_info >= (3, 9):
            path = pkg_resources.files(package_name)
            return str(path)
        else:
            with pkg_resources.path(package_name, "__init__.py") as p:
                return str(p.parent)
    except Exception as e:
        # Fallback to standard os.path navigation if importlib fails
        import voice_control
        return os.path.join(
            os.path.dirname(voice_control.__file__),
            "bridge",
            "clients",
            language.value,
        )


# This dictionary defines extra files needed for a complete example for each language.
# The key is the language, the value is a list of paths relative to the project root.
EXAMPLE_DEPENDENCIES = {BridgeLanguage.LUA: ["lua_client_example/rpc_api.lua"]}


def scaffold_bridge(
    language: BridgeLanguage, destination: str, only_files: list = None
):
    """
    Copies the bridge files and their dependencies for the specified language.
    """
    logger = get_logger(__name__)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Determine which files to copy
    files_to_copy = []
    if only_files:
        # User specified exact files
        bridge_source_path = get_bridge_source_path(language)
        for file_name in only_files:
            files_to_copy.append(os.path.join(bridge_source_path, file_name))
    else:
        # Copy all bridge files for the language
        bridge_source_path = get_bridge_source_path(language)
        if os.path.isdir(bridge_source_path):
            for file_name in os.listdir(bridge_source_path):
                files_to_copy.append(
                    os.path.join(bridge_source_path, file_name)
                )

        # Also copy example dependencies
        if language in EXAMPLE_DEPENDENCIES:
            for dep_path in EXAMPLE_DEPENDENCIES[language]:
                files_to_copy.append(os.path.join(project_root, dep_path))

    if not files_to_copy:
        logger.warning("No files selected for scaffolding.")
        return

    logger.info(f"Scaffolding files for '{language.value}' to: {destination}")

    for source_path in files_to_copy:
        if not os.path.isfile(source_path):
            logger.warning(f"Source file not found: {source_path}. Skipping.")
            continue

        file_name = os.path.basename(source_path)
        destination_item = os.path.join(destination, file_name)

        if os.path.exists(destination_item):
            logger.warning(
                f"File '{file_name}' already exists in the destination. Skipping."
            )
            continue

        shutil.copy2(source_path, destination_item)
        logger.info(f"  - Copied: {file_name}")

    logger.info("Bridge scaffolding complete.")


def main():
    """
    Main entry point for the command-line tool.
    """
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Scaffold bridge files for a target language into the current directory."
    )
    parser.add_argument(
        "--lang",
        type=str,
        required=True,
        choices=[lang.value for lang in BridgeLanguage],
        help="The target language to scaffold the bridge for.",
    )
    parser.add_argument(
        "--dest",
        type=str,
        default=os.getcwd(),
        help="The destination directory to copy the files to. Defaults to the current working directory.",
    )
    parser.add_argument(
        "--only",
        type=str,
        nargs="+",
        help="A list of specific files to copy (e.g., --only llm_client.lua tool_server.lua).",
    )

    args = parser.parse_args()

    try:
        language = BridgeLanguage(args.lang)
    except ValueError:
        # This should theoretically not be reached due to `choices` in argparse
        get_logger(__name__).error(f"Invalid language choice: {args.lang}")
        sys.exit(1)

    scaffold_bridge(language, args.dest, only_files=args.only)


if __name__ == "__main__":
    main()
