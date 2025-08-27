import unittest
import os
import soundfile as sf
from voice_control.tts.model import TTS
from unittest.mock import patch

class TestTTS(unittest.TestCase):
    @patch('voice_control.tts.model.AudioPlayer')
    def test_generate_audio(self, mock_audio_player):
        """
        Test that the TTS can generate audio and save it to a file.
        """
        # Download the TTS models
        TTS.download()

        # Initialize the TTS
        tts = TTS()

        # Generate audio for a sample sentence
        text = "This is a test sentence for the voice detection system."

        # We need to manually call the __call__ method to get the samples
        # because the tts object itself is a callable that plays the audio
        phonemes = tts.tokenizer.phonemize(text, lang="en-us")
        samples, sample_rate = tts.kokoro.create(
            phonemes,
            voice="af_heart",
            speed=1.0,
            is_phonemes=True,
        )

        # Check that the audio is generated
        self.assertIsNotNone(samples)
        self.assertGreater(len(samples), 0)
        self.assertEqual(sample_rate, 24000)

        # Save the audio to a file
        output_path = os.path.join(os.path.dirname(__file__), "test_audio.wav")
        sf.write(output_path, samples, sample_rate)

        # Check that the file was created
        self.assertTrue(os.path.exists(output_path))
