#!/usr/bin/env python3
"""
Voice Control Pipeline Module

This module provides the integration between ASR (Automatic Speech Recognition),
LLM (Language Model), and TTS (Text-to-Speech) components to create a seamless
voice control pipeline.
"""
import sys

from .common.utils import setup_logging, get_logger

from .asr.core import ASRCore
from .llm.core import LLMCore
from .tts.core import TTSCore


class CallbackList(list):
    def __call__(self, response: str, tool_calls: list[dict]) -> str:
        return "/n".join(cb(response, tool_calls) for cb in self)


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
        llm: LLMCore | None = None,
        callback: callable | None = None,
    ):
        """
        Initialize the voice control pipeline with ASR, LLM, and TTS components,
        and dynamically set up tool execution based on the game API spec.
        """
        self.asr = ASRCore(
            transcription_callback=self._asr_callback,
        )
        self.llm = llm or LLMCore(tool_use=isinstance(callback, callable))

        self.tts = TTSCore()
        self.callback = callback or CallbackList([])

    def register_callback(self, callback: callable) -> None:
        """
        Registers a callback receiving the llm response and tool calls.
        Can only be used if initialized without callback.
        """
        self.callback.append(callback)

    def _asr_callback(self, transcription: str):
        """
        Process transcribed text through LLM, handle tool calls, and then send to TTS.
        This is the main callback for ASR output.
        """
        messages = self.llm.create_messages(query=transcription)
        response_llm_raw = self.llm.generate_response(messages)
        response, tool_calls = self.llm.parse_response(response_llm_raw)

        if self.callback:
            response = self.callback(response, tool_calls)

        if r := response.strip():
            self.tts.speak(r)

    def start(self):
        """Start the voice control pipeline."""
        self.asr.process_audio()


def main():
    """
    Main function to run the pipeline.
    """
    setup_logging(log_level="DEBUG")
    logger = get_logger(__name__)

    try:
        pipe = Pipeline()
        l_ = pipe.llm
        l_.system_prompt = "Your goal is to assist the user in navigating and interacting with the game world through voice commands. Be helpful and responsive."
        l_.contexts.append("You are a playable character in an open-world game.")

        logger.info("Starting voice control pipeline...")
        pipe.start()
    except Exception as e:
        logger.error(f"Error in main(): {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
