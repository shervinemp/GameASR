import unittest
import numpy as np
from unittest.mock import patch, MagicMock
from voice_control.asr.models import ParakeetV2
from voice_control.tts.model import TTSProviders
from voice_control.common.config import config


class TestASR(unittest.TestCase):
    @patch("voice_control.asr.models.parakeetv2.ParakeetV2._inputstream")
    @patch("voice_control.asr.models.parakeetv2.Silero")
    def test_parakeet_v2_transcribe(self, mock_silero, mock_input_stream):
        """Generate TTS audio in memory, transcribe with ASR, check similarity."""
        mock_vad = MagicMock()
        mock_silero.return_value = mock_vad
        mock_input_stream.return_value = MagicMock()
        asr = ParakeetV2()

        original = "this is a test sentence for the voice detection system"

        with patch("voice_control.tts.model.AudioPlayer"):
            tts_cls = getattr(TTSProviders, config.get("tts.provider"))
            tts_cls.download()
            tts = tts_cls()
            samples, sr = tts(original, interrupt=False)

        # ONNX ASR expects mono
        if samples.ndim > 1 and samples.shape[1] > 1:
            samples = np.mean(samples, axis=1, keepdims=False).astype(samples.dtype)

        transcript = asr._model.recognize(samples, sample_rate=sr)

        self.assertIsNotNone(transcript)
        self.assertGreater(len(transcript), 0)

        similarity = len(
            set(transcript.lower().split()) & set(original.split())
        ) / len(set(original.split()))
        self.assertGreater(similarity, 0.8)
