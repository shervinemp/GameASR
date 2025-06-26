#!/usr/bin/env python3
"""
Module containing the AudioPlayer class.
Part of the TTS components package.

This module provides functionality for playing audio files and raw data.
"""

import numpy as np
import simpleaudio as sa

from ...common.utils import get_logger


class AudioPlayer:
    """
    Plays audio using the simpleaudio library.

    This class can play raw audio data or WAV files. It provides both
    blocking (synchronous) and non-blocking (asynchronous) playback options.
    """

    def __init__(self, device=None, buffer_size=2048):
        """
        Initializes the AudioPlayer.

        Args:
            device (str, optional): The output device to use. Defaults to None (system default).
            buffer_size (int, optional): The buffer size. Defaults to 2048. Not directly used
                                       by simpleaudio's play functions but kept for compatibility.
        """
        self.logger = get_logger(__name__)
        self.device = device  # Note: simpleaudio doesn't directly support device selection in play_*
        self.buffer_size = buffer_size
        self.logger.info("AudioPlayer initialized.")

    def play_audio(
        self,
        audio_data: np.ndarray | bytes,
        sample_rate: int = 24000,
        wait_done: bool = False,
    ) -> sa.PlayObject | None:
        """
        Plays raw audio data from a NumPy array or bytes.

        Args:
            audio_data (np.ndarray | bytes): A NumPy array or bytes object of audio samples.
            sample_rate (int): The sample rate in Hz (default: 24000).
            wait_done (bool): If True, this function will block until playback is finished.
                              If False (default), it returns a PlayObject immediately.

        Returns:
            simpleaudio.PlayObject or None: If wait_done is False, returns the PlayObject for
                                           manual control (e.g., stop(), is_playing()).
                                           If wait_done is True, returns None.
        """
        try:
            # If the input is a NumPy array, convert it to 16-bit PCM bytes
            if isinstance(audio_data, np.ndarray):
                # Ensure the data type is float for normalization
                if audio_data.dtype != np.float32 and audio_data.dtype != np.float64:
                    audio_data = audio_data.astype(np.float32)

                # Normalize to the range [-1.0, 1.0] if it isn't already
                if np.max(np.abs(audio_data)) > 1.0:
                    audio_data /= np.max(np.abs(audio_data))

                # Scale to 16-bit integer range
                audio_data = (audio_data * 32767).astype(np.int16)
                audio_bytes = audio_data.tobytes()
            else:
                audio_bytes = audio_data

            self.logger.debug("Starting audio playback...")
            play_obj = sa.play_buffer(
                audio_bytes,
                num_channels=1,
                bytes_per_sample=2,  # 16-bit PCM = 2 bytes
                sample_rate=sample_rate,
            )

            if wait_done:
                self.logger.debug("Waiting for playback to complete...")
                play_obj.wait_done()
                self.logger.debug("Playback finished.")
                return None
            else:
                return play_obj

        except Exception as e:
            self.logger.error(f"Error during audio playback: {e}")
            raise

    def play_file(self, file_path: str, wait_done: bool = True):
        """
        Plays a WAV audio file.

        Args:
            file_path (str): The path to the WAV audio file.
            wait_done (bool): If True, this function will block until playback is finished.
                              If False (default), it returns a PlayObject immediately.

        Returns:
            simpleaudio.PlayObject or None: If wait_done is False, returns the PlayObject.
                                           If wait_done is True, returns None.
        """
        try:
            self.logger.debug(f"Loading audio file: {file_path}")
            wave_obj = sa.WaveObject.from_wave_file(file_path)

            self.logger.debug("Starting file playback...")
            play_obj = wave_obj.play()

            if wait_done:
                self.logger.debug("Waiting for file playback to complete...")
                play_obj.wait_done()
                self.logger.debug("File playback finished.")
                return None
            else:
                return play_obj

        except Exception as e:
            self.logger.error(f"Error playing audio file '{file_path}': {e}")
            raise


if __name__ == "__main__":
    # This is an example of how to use the AudioPlayer class
    # To run this, you'll need a WAV file named 'test.wav' in the same directory.

    logger = get_logger("AudioPlayerExample")
    player = AudioPlayer()

    # --- Example 1: Playing a sine wave (blocking) ---
    logger.info("--- Testing blocking playback with a generated sine wave ---")
    sample_rate = 44100
    frequency = 440  # A4 note
    duration = 2.0  # seconds
    t = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False)
    sine_wave = 0.5 * np.sin(2 * np.pi * frequency * t)

    player.play_audio(sine_wave, sample_rate=sample_rate, wait_done=True)
    logger.info("Sine wave playback complete.\n")

    # --- Example 2: Playing a file (non-blocking) ---
    logger.info("--- Testing non-blocking playback with a WAV file ---")
    try:
        play_object = player.play_file("test.wav", wait_done=False)
        if play_object:
            logger.info("File playback started. The script can do other things now.")
            # For example, we can check if it's still playing
            while play_object.is_playing():
                logger.info("Still playing...")
                sa.time.sleep(0.5)
            logger.info("File playback complete.")
        else:
            logger.warning(
                "Could not start file playback (maybe wait_done=True was used or an error occurred)."
            )

    except FileNotFoundError:
        logger.error(
            "Could not find 'test.wav'. Please place a WAV file with this name in the directory to run the file example."
        )
    except Exception as e:
        logger.error(f"An error occurred during file playback test: {e}")
