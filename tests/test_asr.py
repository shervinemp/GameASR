import unittest
import os
import soundfile as sf
from voice_control.asr.models import KyutaiSTT

class TestASR(unittest.TestCase):
    def test_transcribe_audio(self):
        """
        Test that the ASR can transcribe audio from a file.

        Note: This test uses a placeholder implementation of the KyutaiSTT model.
        The test_audio.wav file was generated using the kokoro TTS model with the
        following text: "This is a test sentence for the voice detection system."
        """
        # Initialize the ASR model
        asr = KyutaiSTT(callback=lambda x: None)

        # Load the audio file
        audio_path = os.path.join(os.path.dirname(__file__), "test_audio.wav")
        samples, sample_rate = sf.read(audio_path)

        # Transcribe the audio
        transcript = asr(samples, sample_rate)

        # Check that the transcript is not empty
        self.assertIsNotNone(transcript)
        self.assertGreater(len(transcript), 0)

        # Check that the transcript is close to the original text
        original_text = "this is a test sentence for the voice detection system"
        # A simple similarity check
        self.assertGreater(len(set(transcript.lower().split()) & set(original_text.split())) / len(set(original_text.split())), 0.8)

    def test_empty_audio(self):
        """
        Test that the ASR returns an empty string for an empty audio stream.
        """
        # Initialize the ASR model
        asr = KyutaiSTT(callback=lambda x: None)

        import numpy as np
        # Transcribe an empty audio stream
        transcript = asr(np.array([]), 16000)

        # Check that the transcript is empty
        self.assertEqual(transcript, "")
