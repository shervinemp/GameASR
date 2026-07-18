#!/usr/bin/env python3
"""
Module containing the TTSProcessor class.
Part of the TTS components package.

This module provides text-to-speech processing functionality using ONNX models.
"""

import os
import re
import shutil
import numpy as np

# Dynamically locate espeak-ng for phonemization
_espeak_lib = os.environ.get("PHONEMIZER_ESPEAK_LIBRARY")
if not _espeak_lib:
    candidates = []
    if os.name == "nt":
        candidates = [
            r"C:\Program Files\eSpeak NG\libespeak-ng.dll",
            r"C:\Program Files (x86)\eSpeak NG\libespeak-ng.dll",
        ]
    espeak_exe = shutil.which("espeak-ng")
    if espeak_exe:
        candidates.append(espeak_exe)
    for path in candidates:
        if os.path.exists(path):
            _espeak_lib = path
            os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", _espeak_lib)
            if os.name == "nt":
                parent = os.path.dirname(path)
                os.environ.setdefault("PATH", "")
                if parent not in os.environ["PATH"]:
                    os.environ["PATH"] = parent + os.pathsep + os.environ["PATH"]
            break

from kokoro_onnx import Kokoro as KokoroONNX  # noqa: E402
from kokoro_onnx.tokenizer import Tokenizer  # noqa: E402

from .audio import AudioPlayer  # noqa: E402

from ..common.utils import download_file, get_logger  # noqa: E402


from ..common.config import config  # noqa: E402


class Kokoro:
    sample_rate: int = 24_000

    def __init__(self):
        self.logger = get_logger(__name__)

        weights_dir = config.get("tts.weights_dir", "model_files/tts")

        self.kokoro = KokoroONNX(
            model_path=os.path.join(weights_dir, "kokoro-v1.0.onnx"),
            voices_path=os.path.join(weights_dir, "voices-v1.0.bin"),
        )
        self.tokenizer = Tokenizer()
        self.audio_player = AudioPlayer()
        self._executor = None

    @classmethod
    def download(cls):
        weights_dir = config.get("tts.weights_dir", "model_files/tts")
        os.makedirs(weights_dir, exist_ok=True)

        required_files = {
            "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx": (
                "7d5df8ecf7d4b1878015a32686053fd0eebe2bc377234608764cc0ef3636a6c5",
                400_000_000,
            ),
            "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin": (
                "bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d",
                100_000_000,
            ),
        }

        allowed_hosts = {
            "github.com",
            "objects.githubusercontent.com",
            "release-assets.githubusercontent.com",
        }
        for url, (expected_sha256, max_bytes) in required_files.items():
            filename = url.split("/")[-1]
            destination = os.path.join(weights_dir, filename)
            if os.path.exists(destination):
                from ..common.utils import verify_file_sha256

                verify_file_sha256(destination, expected_sha256)
            else:
                download_file(
                    url,
                    destination,
                    expected_sha256=expected_sha256,
                    allowed_hosts=allowed_hosts,
                    max_bytes=max_bytes,
                )

    def _synthesize(self, text: str, voice: str, language: str, speed: float, interrupt: bool):
        text = re.sub(r'[*_~`´<>]', '', text)
        import emoji
        text = emoji.demojize(text).strip()
        if not text:
            self.logger.warning("Empty text after sanitization. Skipping TTS.")
            return np.array([], dtype=np.float32), 0

        phonemes = self.tokenizer.phonemize(text, lang=language)
        if not phonemes.strip():
            self.logger.warning("Empty phonemes. Skipping TTS.")
            return np.array([], dtype=np.float32), 0

        samples, sample_rate = self.kokoro.create(phonemes, voice=voice, speed=speed, is_phonemes=True)
        self.audio_player(samples, sample_rate, interrupt)
        return samples, sample_rate

    def __call__(
        self,
        text: str,
        voice: str = "af_heart",
        language: str = "en-us",
        speed: float = 1.0,
        interrupt: bool = False,
    ):
        if self._executor is None:
            return self._synthesize(text, voice, language, speed, interrupt)
        self._executor.submit(self._synthesize, text, voice, language, speed, interrupt)

    def start(self):
        from concurrent.futures import ThreadPoolExecutor
        self._executor = ThreadPoolExecutor(max_workers=1)
        self.audio_player.start()

    def stop(self):
        if self._executor:
            self._executor.shutdown(wait=False)
        self.audio_player.stop()

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


# ----------------------------------------------------------------------


class TTSProviders:
    Kokoro: type = Kokoro


# ----------------------------------------------------------------------
