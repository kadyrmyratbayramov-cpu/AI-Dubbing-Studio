"""Audio processing and manipulation utilities."""

import numpy as np
from typing import Tuple, Optional
from src.config.settings import Config


class AudioProcessor:
    """Handles audio loading, processing, and saving."""

    def __init__(self, config: Config):
        """Initialize audio processor.

        Args:
            config: Configuration object
        """
        self.config = config
        self.sample_rate = config.sample_rate

    def load_audio(
        self,
        file_path: str,
        sr: Optional[int] = None
    ) -> np.ndarray:
        """Load audio file.

        Args:
            file_path: Path to audio file
            sr: Sample rate (uses config default if None)

        Returns:
            Audio data as numpy array
        """
        try:
            import librosa
            sr = sr or self.sample_rate
            audio, _ = librosa.load(file_path, sr=sr)
            return audio
        except ImportError:
            raise ImportError("librosa is required for audio loading")
        except Exception as e:
            raise RuntimeError(f"Failed to load audio file: {str(e)}")

    def save_audio(
        self,
        audio: np.ndarray,
        file_path: str,
        sr: Optional[int] = None
    ) -> None:
        """Save audio to file.

        Args:
            audio: Audio data as numpy array
            file_path: Output file path
            sr: Sample rate (uses config default if None)
        """
        try:
            import soundfile as sf
            sr = sr or self.sample_rate
            sf.write(file_path, audio, sr)
        except ImportError:
            raise ImportError("soundfile is required for audio saving")
        except Exception as e:
            raise RuntimeError(f"Failed to save audio file: {str(e)}")

    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to [-1, 1] range.

        Args:
            audio: Audio data

        Returns:
            Normalized audio
        """
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio / max_val
        return audio

    def apply_fade(
        self,
        audio: np.ndarray,
        fade_in: int = 0,
        fade_out: int = 0
    ) -> np.ndarray:
        """Apply fade in/out effects.

        Args:
            audio: Audio data
            fade_in: Fade in duration in samples
            fade_out: Fade out duration in samples

        Returns:
            Audio with fade effects
        """
        if fade_in > 0:
            audio[:fade_in] *= np.linspace(0, 1, fade_in)
        if fade_out > 0:
            audio[-fade_out:] *= np.linspace(1, 0, fade_out)
        return audio
