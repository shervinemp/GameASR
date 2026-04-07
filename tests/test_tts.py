import unittest
import os
import soundfile as sf
from voice_control.tts.model import TTSProviders
from voice_control.common.config import config
from unittest.mock import patch


class TestTTS(unittest.TestCase):
    @patch("voice_control.tts.model.AudioPlayer")
    def test_generate_audio(self, mock_audio_player):
        """
        Test that the TTS can generate audio and save it to a file.
        """
        for provider_name in dir(TTSProviders):
            if provider_name.startswith("__"):
                continue

            with self.subTest(provider=provider_name):
                tts_cls = getattr(TTSProviders, provider_name)

                # Download the TTS models
                tts_cls.download()
                # Initialize the TTS
                tts = tts_cls()

                # Generate audio for a sample sentence
                text = "This is a test sentence for the voice detection system."

                # Call the TTS provider's __call__ method
                samples, sample_rate = tts(text, interrupt=False)

                # Check that the audio is generated
                self.assertIsNotNone(samples)
                self.assertGreater(len(samples), 0)
                self.assertEqual(sample_rate, 24000)

                # Save the audio to a file
                output_path = os.path.join(os.path.dirname(__file__), f"test_audio_{provider_name.lower()}.wav")
                sf.write(output_path, samples, sample_rate)

                # Check that the file was created
                self.assertTrue(os.path.exists(output_path))
