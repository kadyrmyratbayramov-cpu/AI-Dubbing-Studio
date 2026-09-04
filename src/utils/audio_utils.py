"""Audio utility functions."""

import numpy as np
from typing import Tuple, Optional


class AudioUtils:
    """Audio utility functions and helpers."""

    @staticmethod
    def get_duration(audio: np.ndarray, sr: int) -> float:
        """Get duration of audio in seconds.

        Args:
            audio: Audio data
            sr: Sample rate

        Returns:
            Duration in seconds
        """
        return len(audio) / sr

    @staticmethod
    def trim_silence(audio: np.ndarray, sr: int, threshold: float = 0.01) -> np.ndarray:
        """Trim silence from audio.

        Args:
            audio: Audio data
            sr: Sample rate
            threshold: Silence threshold

        Returns:
            Trimmed audio
        """
        energy = np.abs(audio)
        mask = energy > threshold
        if np.any(mask):
            first = np.argmax(mask)
            last = len(mask) - np.argmax(mask[::-1])
            return audio[first:last]
        return audio

    @staticmethod
    def resample(audio: np.ndarray, sr_orig: int, sr_target: int) -> np.ndarray:
        """Resample audio to target sample rate.

        Args:
            audio: Audio data
            sr_orig: Original sample rate
            sr_target: Target sample rate

        Returns:
            Resampled audio
        """
        if sr_orig == sr_target:
            return audio
        try:
            import librosa
            return librosa.resample(audio, orig_sr=sr_orig, target_sr=sr_target)
        except ImportError:
            raise ImportError("librosa is required for resampling")

    @staticmethod
    def apply_volume(audio: np.ndarray, factor: float) -> np.ndarray:
        """Apply volume adjustment.

        Args:
            audio: Audio data
            factor: Volume factor (1.0 = no change)

        Returns:
            Volume-adjusted audio
        """
        return np.clip(audio * factor, -1.0, 1.0)
