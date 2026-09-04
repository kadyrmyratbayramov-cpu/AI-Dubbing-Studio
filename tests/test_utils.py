"""Tests for utility modules."""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.utils.audio_utils import AudioUtils
from src.utils.text_utils import TextUtils
from src.utils.validators import (
    validate_audio_format,
    validate_input_file,
    validate_output_path,
)


class TestAudioUtils:
    """Test audio utility functions."""

    def test_get_duration(self):
        """Test duration calculation."""
        audio = np.zeros(22050)  # 1 second at 22050 Hz
        duration = AudioUtils.get_duration(audio, 22050)
        assert duration == 1.0

    def test_trim_silence(self):
        """Test silence trimming."""
        audio = np.array([0.0, 0.0, 0.2, 0.3, 0.0])
        trimmed = AudioUtils.trim_silence(audio, sr=22050, threshold=0.1)
        assert np.array_equal(trimmed, np.array([0.2, 0.3]))

    def test_resample_same_rate_returns_input(self):
        """Test no-op resample when sample rates are equal."""
        audio = np.array([0.1, 0.2])
        result = AudioUtils.resample(audio, sr_orig=22050, sr_target=22050)
        assert np.array_equal(result, audio)

    def test_resample_calls_librosa(self, monkeypatch):
        """Test resample delegates to librosa."""

        def fake_resample(audio, orig_sr, target_sr):
            assert orig_sr == 22050
            assert target_sr == 16000
            return np.array([1.0, -1.0])

        monkeypatch.setitem(sys.modules, "librosa", SimpleNamespace(resample=fake_resample))

        output = AudioUtils.resample(np.array([0.1, 0.2]), sr_orig=22050, sr_target=16000)
        assert np.array_equal(output, np.array([1.0, -1.0]))

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

    def test_normalize_punctuation(self):
        """Test punctuation normalization."""
        text = "Hello!!! What??"
        normalized = TextUtils.normalize_punctuation(text)
        assert normalized == "Hello! What?"


class TestValidators:
    """Test validation functions."""

    def test_validate_audio_format(self):
        """Test audio format validation."""
        with pytest.raises(ValueError):
            validate_audio_format("file.txt")

        assert validate_audio_format("file.wav") is True
        assert validate_audio_format("file.mp3") is True

    def test_validate_input_file_success(self, tmp_path: Path):
        """Test input file validation success."""
        file_path = tmp_path / "input.wav"
        file_path.write_text("ok")
        assert validate_input_file(str(file_path)) is True

    def test_validate_input_file_not_found(self):
        """Test input file validation not found."""
        with pytest.raises(FileNotFoundError):
            validate_input_file("/nonexistent/file.wav")

    def test_validate_input_file_not_file(self, tmp_path: Path):
        """Test input path is directory."""
        with pytest.raises(ValueError):
            validate_input_file(str(tmp_path))

    def test_validate_input_file_not_readable(self, monkeypatch, tmp_path: Path):
        """Test input file validation for unreadable file."""
        file_path = tmp_path / "input.wav"
        file_path.write_text("ok")
        monkeypatch.setattr(os, "access", lambda *_: False)

        with pytest.raises(PermissionError):
            validate_input_file(str(file_path))

    def test_validate_output_path_success(self, tmp_path: Path):
        """Test output path validation success."""
        output_file = tmp_path / "output.wav"
        assert validate_output_path(str(output_file)) is True

    def test_validate_output_path_missing_directory(self, tmp_path: Path):
        """Test missing output directory."""
        output_file = tmp_path / "missing" / "output.wav"
        with pytest.raises(FileNotFoundError):
            validate_output_path(str(output_file))

    def test_validate_output_path_not_writable(self, monkeypatch, tmp_path: Path):
        """Test output path validation for unwritable directory."""
        output_file = tmp_path / "output.wav"
        monkeypatch.setattr(os, "access", lambda *_: False)

        with pytest.raises(PermissionError):
            validate_output_path(str(output_file))
