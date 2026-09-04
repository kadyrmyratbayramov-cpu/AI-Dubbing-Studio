"""Audio processing and manipulation utilities."""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.config.settings import Config


class AudioProcessor:
    """Handles audio loading and utility transforms."""

    def __init__(self, config: Config):
        self.config = config
        self.sample_rate = config.sample_rate

    def load_audio(self, file_path: str, sr: Optional[int] = None) -> np.ndarray:
        import librosa

        sample_rate = sr or self.sample_rate
        audio, _ = librosa.load(file_path, sr=sample_rate, mono=False)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)
        return audio

    def save_audio(self, audio: np.ndarray, file_path: str, sr: Optional[int] = None) -> None:
        import soundfile as sf

        sf.write(file_path, audio, sr or self.sample_rate)

    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        max_val = float(np.max(np.abs(audio))) if audio.size else 0.0
        return audio / max_val if max_val > 0 else audio

    def apply_fade(self, audio: np.ndarray, fade_in: int = 0, fade_out: int = 0) -> np.ndarray:
        output = audio.astype(np.float32).copy()
        fade_in = max(0, min(fade_in, len(output)))
        fade_out = max(0, min(fade_out, len(output)))

        if fade_in > 0:
            output[:fade_in] *= np.linspace(0.0, 1.0, fade_in)
        if fade_out > 0:
            output[-fade_out:] *= np.linspace(1.0, 0.0, fade_out)
        return output
