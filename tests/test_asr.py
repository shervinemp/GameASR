import unittest
import os
import soundfile as sf
from unittest.mock import patch, MagicMock
from onnx_asr import load_model


class TestASR(unittest.TestCase):
    def test_transcribe_audio_file(self):
        """
        Test that the ASR model can transcribe an audio file accurately.
        """
        # Load the ASR model
        model = load_model("nemo-parakeet-tdt-0.6b-v2", quantization="int8")

        # Load the test audio
        original_text = (
            "this is a test sentence for the voice detection system"
        )
        audio_path = os.path.join(os.path.dirname(__file__), "test_audio.wav")
        samples, sample_rate = sf.read(audio_path)

        # Transcribe the audio
        transcript = model.recognize(samples, sample_rate=sample_rate)

        # Check that the transcript is not empty
        self.assertIsNotNone(transcript)
        self.assertGreater(len(transcript), 0)

        # Check that the transcript is close to the original text
        # A simple similarity check using Jaccard similarity
        original_words = set(original_text.split())
        transcript_words = set(transcript.lower().split())
        similarity = len(original_words.intersection(transcript_words)) / len(original_words.union(transcript_words))

        self.assertGreater(similarity, 0.8, f"Transcription similarity is too low: {similarity}")