"""Tests for core modules."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from src.config.settings import Config
from src.core.audio_processor import AudioProcessor
from src.core.dubbing_pipeline import DubbingPipeline


class TestConfig:
    """Test configuration module."""

    def test_config_initialization(self):
        """Test Config class initialization."""
        config = Config()
        assert config.sample_rate == 22050
        assert config.channels == 1
        assert config.bit_depth == 16

    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = Config()
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert "sample_rate" in config_dict

    def test_load_from_file_yaml_nested(self, tmp_path: Path):
        """Test nested YAML configuration loading."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            yaml.safe_dump(
                {
                    "audio": {"sample_rate": 44100, "channels": 2},
                    "model": {"default_voice": "en_GB", "cache_models": False},
                    "processing": {"remove_silence": True},
                    "output": {"debug": True},
                    "dubbing": {"voice_speed": 1.25},
                }
            )
        )

        config = Config()
        config.load_from_file(str(config_file))

        assert config.sample_rate == 44100
        assert config.channels == 2
        assert config.default_voice == "en_GB"
        assert config.cache_models is False
        assert config.remove_silence is True
        assert config.debug is True
        assert config.voice_speed == 1.25

    def test_load_from_file_json_nested(self, tmp_path: Path):
        """Test nested JSON configuration loading."""
        config_file = tmp_path / "config.json"
        config_file.write_text(
            json.dumps(
                {
                    "audio": {"sample_rate": 48000},
                    "dubbing": {"target_language": "es"},
                }
            )
        )

        config = Config()
        config.load_from_file(str(config_file))

        assert config.sample_rate == 48000
        assert config.target_language == "es"

    def test_load_from_file_unsupported_extension(self, tmp_path: Path):
        """Test unsupported config extension raises error."""
        config_file = tmp_path / "config.txt"
        config_file.write_text("invalid")

        with pytest.raises(ValueError):
            Config().load_from_file(str(config_file))

    def test_load_yaml_wraps_errors(self):
        """Test YAML load wraps runtime errors."""
        with pytest.raises(RuntimeError):
            Config()._load_yaml("/nonexistent/config.yaml")

    def test_load_json_wraps_errors(self):
        """Test JSON load wraps runtime errors."""
        with pytest.raises(RuntimeError):
            Config()._load_json("/nonexistent/config.json")


class TestAudioProcessor:
    """Test audio processing module."""

    def test_audio_processor_initialization(self):
        """Test AudioProcessor initialization."""
        config = Config()
        processor = AudioProcessor(config)
        assert processor.sample_rate == config.sample_rate

    def test_load_audio(self, monkeypatch):
        """Test loading audio delegates to librosa."""
        called = {}

        def fake_load(path, sr):
            called["path"] = path
            called["sr"] = sr
            return np.array([0.1, -0.1]), sr

        monkeypatch.setitem(sys.modules, "librosa", SimpleNamespace(load=fake_load))

        processor = AudioProcessor(Config())
        audio = processor.load_audio("sample.wav", sr=16000)

        assert np.array_equal(audio, np.array([0.1, -0.1]))
        assert called == {"path": "sample.wav", "sr": 16000}

    def test_save_audio(self, monkeypatch, tmp_path: Path):
        """Test saving audio delegates to soundfile."""
        called = {}

        def fake_write(path, audio, sr):
            called["path"] = path
            called["audio"] = audio
            called["sr"] = sr

        monkeypatch.setitem(sys.modules, "soundfile", SimpleNamespace(write=fake_write))

        processor = AudioProcessor(Config())
        audio = np.array([0.2, -0.2])
        output_file = tmp_path / "out.wav"
        processor.save_audio(audio, str(output_file), sr=24000)

        assert called["path"] == str(output_file)
        assert np.array_equal(called["audio"], audio)
        assert called["sr"] == 24000

    def test_normalize_audio(self):
        """Test audio normalization."""
        processor = AudioProcessor(Config())
        audio = np.array([0.5, -2.0, 1.0])

        normalized = processor.normalize_audio(audio)

        assert np.allclose(normalized, np.array([0.25, -1.0, 0.5]))

    def test_apply_fade(self):
        """Test fade in/out."""
        processor = AudioProcessor(Config())
        audio = np.ones(10)

        faded = processor.apply_fade(audio.copy(), fade_in=2, fade_out=2)

        assert faded[0] == 0.0
        assert faded[1] == 1.0
        assert faded[-1] == 0.0
        assert faded[-2] == 1.0


class TestDubbingPipeline:
    """Test dubbing pipeline module."""

    def test_pipeline_initialization(self):
        """Test DubbingPipeline initialization."""
        config = Config()
        pipeline = DubbingPipeline(config)
        assert pipeline.config == config
        assert pipeline.audio_processor is not None
        assert pipeline.voice_synthesis is not None

    def test_process(self, monkeypatch, tmp_path: Path):
        """Test full pipeline process flow."""
        input_file = tmp_path / "input.wav"
        output_file = tmp_path / "output.wav"
        input_file.write_text("stub")

        validated = {"called": False}

        def fake_validate(path):
            validated["called"] = True
            assert path == str(input_file)
            return True

        monkeypatch.setattr("src.core.dubbing_pipeline.validate_input_file", fake_validate)

        pipeline = DubbingPipeline(Config())
        source_audio = np.array([0.1, 0.2, 0.3])
        dubbed_audio = np.array([0.3, 0.2, 0.1])
        saved = {}

        monkeypatch.setattr(pipeline.audio_processor, "load_audio", lambda _: source_audio)
        monkeypatch.setattr(
            pipeline.voice_synthesis,
            "synthesize",
            lambda audio, **kwargs: dubbed_audio,
        )

        def fake_save(audio, path):
            saved["audio"] = audio
            saved["path"] = path

        monkeypatch.setattr(pipeline.audio_processor, "save_audio", fake_save)

        result = pipeline.process(str(input_file), str(output_file), language="en")

        assert validated["called"] is True
        assert np.array_equal(saved["audio"], dubbed_audio)
        assert saved["path"] == str(output_file)
        assert result["status"] == "success"
        assert result["input"] == str(input_file)
        assert result["output"] == str(output_file)
        assert result["duration"] == len(dubbed_audio) / pipeline.config.sample_rate
