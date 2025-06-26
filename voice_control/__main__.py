# voice_control_backend_main.py
import argparse
import sys

from .pipeline import Pipeline

from .bridge.rpc_tool_client import ToolCaller, get_client_class
from .bridge.rpc_server import LLMService, RpcServer

from .common.utils import setup_logging, get_logger, parse_api_spec


def main():
    parser = argparse.ArgumentParser(description="Run the voice-control pipeline.")
    parser.add_argument(
        "specs-path",
        type=str,
        help="Path to the tool specification JSON file.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port used by the RPC server.",
    )
    parser.add_argument(
        "--protocol",
        type=str,
        default="tcp",
        choices=["tcp", "ipc"],
        help="Protocol used by the RPC server.",
    )
    parser.add_argument(
        "--tools-host",
        type=str,
        default="127.0.0.1",
        help="Host address of the tools server.",
    )
    parser.add_argument(
        "--tools-port",
        type=int,
        default=8080,
        help="Port used by the tools server.",
    )
    parser.add_argument(
        "--tools-protocol",
        type=str,
        default="tcp",
        choices=["tcp", "ipc"],
        help="Protocol used by the tools server.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level.",
    )
    args = parser.parse_args()

    setup_logging(log_level=args.log_level)
    logger = get_logger(__name__)

    try:
        tools_spec = parse_api_spec(args.specs_path)
        logger.info(f"Successfully parsed tool spec from '{args.specs_path}'.")
    except Exception as e:
        logger.critical(
            f"Error parsing game API spec '{args.specs_path}': {e}. Exiting.",
            exc_info=True,
        )
        sys.exit(1)

    tool_caller = ToolCaller()

    try:
        pipe = Pipeline()
        logger.info("Successfully created voice control pipeline.")
    except Exception as e:
        logger.critical(
            f"Error creating voice control pipeline: {e}. Exiting.",
            exc_info=True,
        )
        sys.exit(1)

    try:
        ToolClient = get_client_class(tools_spec)
        endpoint = (
            args.tools_protocol + "://" + args.tools_host + ":" + str(args.tools_port)
        )
        tool_client = ToolClient(endpoint, protocol=args.tools_protocol)
    except Exception as e:
        logger.critical(
            f"Error creating tools client: {e}. Exiting.",
            exc_info=True,
        )
        sys.exit(1)

    tool_caller = ToolCaller(tool_client)

    try:
        pipe = Pipeline(callback=tool_caller)
        logger.info("Pipeline instance created.")
    except Exception as e:
        logger.critical(f"Failed to initialize Pipeline: {e}. Exiting.", exc_info=True)
        sys.exit(1)

    try:
        service_api = LLMService(pipe.llm)
        endpoint = args.protocol + "://" + args.host + ":" + str(args.port)
        rpc_server = RpcServer(service_api, endpoint, protocol=args.protocol)
    except Exception as e:
        logger.critical(
            f"Failed to initialize RPC Server: {e}. Exiting.", exc_info=True
        )
        sys.exit(1)

    pipe.start()
    rpc_server.start()


if __name__ == "__main__":
    main()
