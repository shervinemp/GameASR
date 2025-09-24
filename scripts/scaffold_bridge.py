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

def get_bridge_source_path(language: BridgeLanguage) -> str:
    """
    Returns the absolute path to the source directory of the bridge files
    for a given language.
    """
    try:
        current_file_path = os.path.abspath(__file__)
        # Navigate from scripts -> project root -> voice_control -> bridge -> clients -> [language]
        scripts_dir = os.path.dirname(current_file_path)
        project_root = os.path.dirname(scripts_dir)
        bridge_path = os.path.join(project_root, "voice_control", "bridge", "clients", language.value)
        return os.path.normpath(bridge_path)
    except Exception as e:
        # A fallback in case the file structure changes, though less robust.
        import voice_control
        return os.path.join(os.path.dirname(voice_control.__file__), "bridge", "clients", language.value)

def scaffold_bridge(language: BridgeLanguage, destination: str):
    """
    Copies the bridge files for the specified language to the destination directory.
    """
    logger = get_logger(__name__)
    source_path = get_bridge_source_path(language)

    if not os.path.isdir(source_path):
        logger.error(f"Source path for language '{language.value}' not found at: {source_path}")
        sys.exit(1)

    logger.info(f"Copying bridge files for '{language.value}' from {source_path} to {destination}")

    for item_name in os.listdir(source_path):
        source_item = os.path.join(source_path, item_name)
        destination_item = os.path.join(destination, item_name)

        if os.path.exists(destination_item):
            logger.warning(f"File '{item_name}' already exists in the destination. Skipping.")
            continue

        if os.path.isfile(source_item):
            shutil.copy2(source_item, destination)
            logger.info(f"  - Copied file: {item_name}")

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
        help="The target language to scaffold the bridge for."
    )
    parser.add_argument(
        "--dest",
        type=str,
        default=os.getcwd(),
        help="The destination directory to copy the files to. Defaults to the current working directory."
    )

    args = parser.parse_args()

    try:
        language = BridgeLanguage(args.lang)
    except ValueError:
        # This should theoretically not be reached due to `choices` in argparse
        get_logger(__name__).error(f"Invalid language choice: {args.lang}")
        sys.exit(1)

    scaffold_bridge(language, args.dest)

if __name__ == "__main__":
    main()