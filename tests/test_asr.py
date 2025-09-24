import unittest
import os
import soundfile as sf
from unittest.mock import patch, MagicMock
from voice_control.tts.model import TTS
from onnx_asr import load_model


def generate_test_audio(text, output_path):
    """
    Generate a test audio file using the kokoro TTS model.
    """
    with patch("voice_control.tts.model.AudioPlayer"):
        tts = TTS()
        phonemes = tts.tokenizer.phonemize(text, lang="en-us")
        samples, sample_rate = tts.kokoro.create(
            phonemes,
            voice="af_heart",
            speed=1.0,
            is_phonemes=True,
        )
        sf.write(output_path, samples, sample_rate)
        return samples, sample_rate


class TestASR(unittest.TestCase):
    def test_transcribe_audio_file(self):
        """
        Test that the ASR model can transcribe an audio file accurately.
        """
        # Load the ASR model
        model = load_model("nemo-parakeet-tdt-0.6b-v2", quantization="int8")

        # Generate the test audio
        original_text = (
            "this is a test sentence for the voice detection system"
        )
        audio_path = os.path.join(os.path.dirname(__file__), "test_audio.wav")
        samples, sample_rate = generate_test_audio(original_text, audio_path)

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
