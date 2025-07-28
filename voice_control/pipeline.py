#!/usr/bin/env python3
"""
Voice Control Pipeline Module

This module provides the integration between ASR (Automatic Speech Recognition),
LLM (Language Model), and TTS (Text-to-Speech) components to create a seamless
voice control pipeline.
"""
import re
import sys
from typing import Optional

from .common.utils import setup_logging, get_logger

from .asr import ParakeetV2
from .llm import Session
from .tts import TTS
from .bridge.rpc_server import RpcServer


class Pipeline:
    """
    Orchestrates the integration between ASR, LLM, and TTS components.

    This class initializes all three components and sets up a pipeline where:
    - ASR captures audio and transcribes it to text
    - The transcribed text is passed to LLM for processing
    - The LLM response is sent to TTS for audio output
    """

    def __init__(self, session: Optional[Session] = None, rpc_server: bool = False):
        """
        Initialize the voice control pipeline with ASR, LLM, and TTS components,
        and dynamically set up tool execution based on the game API spec.
        """
        self.logger = get_logger(__name__)

        self.asr = ParakeetV2()
        self.session = session or Session()
        self.tts = TTS()
        self.rpc_server = RpcServer(rpc_handler=self.session) if rpc_server else None

    def _callback(self, transcription: str):
        """
        Process transcribed text through LLM, handle tool calls, and then send to TTS.
        This is the main callback for ASR output.
        """
        if not transcription:
            return
        boundary = re.compile(r"[^.][.!?]\s+")
        buffer = ""
        interrupt = True
        for chunk in self.session(transcription):
            buffer += chunk
            matches = boundary.finditer(buffer)
            if matches:
                for match in matches:
                    sentence = buffer[: match.start() + 2]
                    buffer = buffer[match.end() :]
                    self.tts(sentence.strip(), interrupt=interrupt)
            interrupt = False
        else:
            if s := buffer.strip():
                self.tts(s, interrupt=interrupt)

    def run(self):
        """Start the voice control pipeline."""
        self.asr.start()
        self.tts.start()
        if self.rpc_server:
            self.rpc_server.start()
        try:
            for transcript in self.asr:
                self.logger.debug(f"{transcript=}")
                self._callback(transcript)
        finally:
            if self.rpc_server:
                self.rpc_server.stop()
            self.asr.stop()
            self.tts.stop()


def main():
    """
    Main function to run the pipeline.
    """
    setup_logging(log_level="DEBUG")
    logger = get_logger(__name__)

    try:
        pipe = Pipeline()
        logger.info("Starting voice control pipeline...")
        pipe.run()
    except Exception as e:
        logger.error(f"Error in main(): {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
