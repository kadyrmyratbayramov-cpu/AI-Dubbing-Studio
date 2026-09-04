"""Tests for utility modules."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.config.settings import Config
from src.core.video_metadata import VideoMetadataReader
from src.utils.audio_utils import AudioUtils
from src.utils.text_utils import TextUtils
from src.utils.validators import validate_audio_format


class TestAudioUtils:
    def test_get_duration(self):
        audio = np.zeros(22050)
        duration = AudioUtils.get_duration(audio, 22050)
        assert duration == 1.0

    def test_apply_volume(self):
        audio = np.ones(1000) * 0.5
        adjusted = AudioUtils.apply_volume(audio, 2.0)
        assert np.max(adjusted) == 1.0


class TestTextUtils:
    def test_clean_text(self):
        text = "Hello  world   test"
        cleaned = TextUtils.clean_text(text)
        assert cleaned == "Hello world test"

    def test_split_sentences(self):
        text = "Hello world. This is a test. Another sentence!"
        sentences = TextUtils.split_sentences(text)
        assert len(sentences) >= 2

    def test_tokenize(self):
        text = "Hello world test"
        tokens = TextUtils.tokenize(text)
        assert tokens == ["Hello", "world", "test"]


class TestValidators:
    def test_validate_audio_format(self):
        with pytest.raises(ValueError):
            validate_audio_format("file.txt")
        assert validate_audio_format("file.wav") is True
        assert validate_audio_format("file.mp3") is True


class TestVideoMetadataReader:
    def test_parse_metadata(self, tmp_path: Path):
        config = Config()
        reader = VideoMetadataReader(config)
        video_path = tmp_path / "video.mp4"
        video_path.write_bytes(b"video")
        payload = {
            "format": {"duration": "12.5"},
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 1280,
                    "height": 720,
                    "avg_frame_rate": "30000/1001",
                },
                {
                    "codec_type": "audio",
                    "codec_name": "aac",
                    "sample_rate": "48000",
                },
            ],
        }
        parsed = reader._parse_metadata(str(video_path), json.loads(json.dumps(payload)))
        assert parsed.duration_seconds == 12.5
        assert parsed.width == 1280
        assert parsed.audio_sample_rate == 48000
