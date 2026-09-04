"""Tests for utility modules."""

import pytest
import numpy as np
from src.utils.audio_utils import AudioUtils
from src.utils.text_utils import TextUtils
from src.utils.validators import validate_output_path, validate_audio_format


class TestAudioUtils:
    """Test audio utility functions."""

    def test_get_duration(self):
        """Test duration calculation."""
        audio = np.zeros(22050)  # 1 second at 22050 Hz
        duration = AudioUtils.get_duration(audio, 22050)
        assert duration == 1.0

    def test_apply_volume(self):
        """Test volume adjustment."""
        audio = np.ones(1000) * 0.5
        adjusted = AudioUtils.apply_volume(audio, 2.0)
        assert np.max(adjusted) == 1.0  # Clipped at 1.0


class TestTextUtils:
    """Test text utility functions."""

    def test_clean_text(self):
        """Test text cleaning."""
        text = "Hello  world   test"
        cleaned = TextUtils.clean_text(text)
        assert cleaned == "Hello world test"

    def test_split_sentences(self):
        """Test sentence splitting."""
        text = "Hello world. This is a test. Another sentence!"
        sentences = TextUtils.split_sentences(text)
        assert len(sentences) >= 2

    def test_tokenize(self):
        """Test tokenization."""
        text = "Hello world test"
        tokens = TextUtils.tokenize(text)
        assert tokens == ["Hello", "world", "test"]


class TestValidators:
    """Test validation functions."""

    def test_validate_audio_format(self):
        """Test audio format validation."""
        with pytest.raises(ValueError):
            validate_audio_format("file.txt")

        # Should not raise for valid formats
        assert validate_audio_format("file.wav") is True
        assert validate_audio_format("file.mp3") is True
