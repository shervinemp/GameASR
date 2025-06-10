#!/usr/bin/env python3
"""
ASR Module Main Entry Point

This script serves as the entry point for running the ASR system directly.
It demonstrates how to use the RealtimeASRSystem class and is useful for testing.

Usage:
    python -m asr.main --vad-threshold=0.25 --end-silence-duration=0.75
"""

import logging
import sys
import time

from core import RealtimeASRSystem, parse_asr_args

# Configure root-level logging (for main.py)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def console_transcription_callback(transcriptions):
    """
    Example callback for direct module testing.
    Prints transcriptions to the console.
    """
    print("\n--- Final Transcription from Module (DEBUG) ---")
    for i, trans in enumerate(transcriptions):
        print(f'DEBUG_TRANSCRIPTION {i+1}: "{trans}"')


def main():
    """Main entry point for running ASR system directly."""
    args = parse_asr_args()

    # Set default audio device if specified
    import sounddevice as sd  # Import here to avoid affecting core module

    if args.device is not None:
        try:
            sd.default.device = args.device
            logging.info(f"Using audio device ID: {args.device}")
        except Exception as e:
            logging.error(f"Could not set audio device {args.device}: {e}")
            print(
                "Please check your device ID using `python -m sounddevice`.",
                file=sys.stderr,
            )
            sys.exit(1)

    print("\n--- Direct Module Test Mode ---")
    print("This runs the ASR system directly from the module for testing purposes.")
    print("Press Enter to start (or Ctrl+C to stop this test).")
    input()

    asr_system = RealtimeASRSystem(args, console_transcription_callback)

    try:
        asr_system.start()
        # Keep the main thread alive while the ASR system runs in background threads
        # In a real app, your UI loop or other main logic would go here.
        while True:
            time.sleep(0.1)  # Small sleep to prevent busy-waiting
    except KeyboardInterrupt:
        pass  # Caught by finally block

    finally:
        asr_system.stop()
        logging.info("Module test application exiting.")


if __name__ == "__main__":
    main()
