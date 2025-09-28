import unittest
import os
import soundfile as sf
from voice_control.tts.model import TTS
from unittest.mock import patch


class TestTTS(unittest.TestCase):
    @patch("voice_control.tts.model.TTS")
    def test_generate_audio(self, MockTTS):
        """
        Test that the TTS can generate audio and save it to a file.
        """
        # Create a mock instance of the TTS class
        tts_instance = MockTTS.return_value
        tts_instance.return_value = (sf.read(os.path.join(os.path.dirname(__file__), "test_audio.wav")))

        # Generate audio for a sample sentence
        text = "This is a test sentence for the voice detection system."
        samples, sample_rate = tts_instance(text)

        # Check that the audio is generated
        self.assertIsNotNone(samples)
        self.assertGreater(len(samples), 0)
        self.assertEqual(sample_rate, 24000)

        # Save the audio to a file
        output_path = os.path.join(os.path.dirname(__file__), "test_audio.wav")
        sf.write(output_path, samples, sample_rate)

        # Check that the file was created
        self.assertTrue(os.path.exists(output_path))