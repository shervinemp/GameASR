import unittest
import os
import soundfile as sf
from unittest.mock import patch, MagicMock
from voice_control.asr.models import ParakeetV2
from voice_control.tts.model import TTS

def generate_test_audio(text, output_path):
    """
    Generate a test audio file using the kokoro TTS model.
    """
    with patch('voice_control.tts.model.AudioPlayer'):
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
    @patch('voice_control.asr.models.parakeetv2.sd.InputStream')
    @patch('voice_control.asr.models.parakeetv2.Silero')
    def test_parakeet_v2_transcribe(self, mock_silero, mock_input_stream):
        """
        Test that the ParakeetV2 model can transcribe audio from a file.
        """
        # Mock the Silero VAD model
        mock_vad = MagicMock()
        mock_silero.return_value = mock_vad

        # Mock the sounddevice InputStream
        mock_input_stream.return_value = MagicMock()

        # Initialize the ParakeetV2 model
        asr = ParakeetV2()

        # Generate the test audio
        original_text = "this is a test sentence for the voice detection system"
        audio_path = os.path.join(os.path.dirname(__file__), "test_audio.wav")
        samples, sample_rate = generate_test_audio(original_text, audio_path)

        # Transcribe the audio
        transcript = asr._model.recognize(samples, sample_rate=sample_rate)

        # Check that the transcript is not empty
        self.assertIsNotNone(transcript)
        self.assertGreater(len(transcript), 0)

        # Check that the transcript is close to the original text
        original_text = "this is a test sentence for the voice detection system"
        # A simple similarity check
        self.assertGreater(len(set(transcript.lower().split()) & set(original_text.split())) / len(set(original_text.split())), 0.8)
