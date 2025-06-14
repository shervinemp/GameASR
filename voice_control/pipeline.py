#!/usr/bin/env python3
"""
Voice Control Pipeline Module

This module provides the integration between ASR (Automatic Speech Recognition),
LLM (Language Model), and TTS (Text-to-Speech) components to create a seamless
voice control pipeline.
"""

import sys
from .asr.core import ASRCore
from .llm.core import LLMCore, Parameter, Tool
from .tts.core import TTSCore

from .common.logging_utils import get_logger, setup_logging


class CallbackList(list):
    def __call__(self, *args, **kwargs):
        return "\n".join(filter(lambda x: x, (cb(*args, **kwargs) for cb in self)))


class Pipeline:
    """
    Orchestrates the integration between ASR, LLM, and TTS components.

    This class initializes all three components and sets up a pipeline where:
    - ASR captures audio and transcribes it to text
    - The transcribed text is passed to LLM for processing
    - The LLM response is sent to TTS for audio output
    """

    def __init__(
        self,
        callback: callable = None,
        system_prompt: str = "",
        tools: list[Tool] = tuple(),
        contexts: list[str] = tuple(),
    ):
        """
        Initialize the voice control pipeline with ASR, LLM, and TTS components.
        """
        # device = torch.cuda.current_device() if torch.cuda.is_available() else None
        self.asr = ASRCore(
            transcription_callback=self._asr_callback,
        )
        self.llm = LLMCore(tool_use=True)
        self.tts = TTSCore()
        self.callback = callback
        self.llm.system_prompt = system_prompt
        self.llm.tools = tools
        self.llm.contexts = contexts

    # Create a wrapper function that will be used as the ASR callback
    def _asr_callback(self, transcription):
        """Process transcribed text through LLM and then send to TTS."""
        messages = self.llm.create_messages(query=transcription)
        response = self.llm.generate_response(messages)
        response, tool_calls = self.llm.parse_response(response)
        if self.callback:
            response = self.callback(response, tool_calls)

        if response.strip():
            self.tts.speak(response)

    def start(self):
        """Start the voice control pipeline."""
        self.asr.process_audio()


def main():
    """
    Main function to run the pipeline.
    """
    # Configure logging for this session
    setup_logging(log_level="DEBUG")

    # Get a logger for this module
    logger = get_logger(__name__)

    cb = CallbackList([])

    tools = [
        Tool(
            name="jump_over",
            description="jump over an object/obstacle.",
            parameters=Parameter(
                type="object",
                properties={
                    "object": Parameter(
                        type="str",
                        description="object to jump over",
                    ),
                },
                required=["object"],
            ),
        ),
        Tool(
            name="go",
            description="go places.",
            parameters=Parameter(
                type="object",
                properties={
                    "speed": Parameter(
                        type="int", description="The speed at which to go in km/h."
                    ),
                    "method": Parameter(
                        type="str",
                        description="The method for going, e.g. Bus, Foot, Drive, etc.",
                    ),
                },
                required=["method"],
            ),
        ),
    ]

    contexts = ["You are a playable character in an open-world game."]

    try:
        # Create an instance of Pipeline
        pipe = Pipeline(cb, tools=tools, contexts=contexts)

        # Start processing audio from the microphone
        logger.info("Starting voice control pipeline...")
        pipe.start()
    except Exception as e:
        logger.error(f"Error in main(): {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Run the main function
    main()
