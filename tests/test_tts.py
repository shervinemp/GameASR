import unittest
from voice_control.tts.model import TTS
from unittest.mock import patch

class TestTTS(unittest.TestCase):
    @patch('voice_control.tts.model.AudioPlayer')
    def test_tts_initialization(self, mock_audio_player):
        """
        Test that the TTS class can be initialized.
        """
        # Download the TTS models
        TTS.download()

        # Initialize the TTS
        tts = TTS()

        # Check that the TTS object is created
        self.assertIsNotNone(tts)
