"""Tests for core modules."""

import pytest
from src.config.settings import Config
from src.core.dubbing_pipeline import DubbingPipeline
from src.core.audio_processor import AudioProcessor


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


class TestAudioProcessor:
    """Test audio processing module."""

    def test_audio_processor_initialization(self):
        """Test AudioProcessor initialization."""
        config = Config()
        processor = AudioProcessor(config)
        assert processor.sample_rate == config.sample_rate


class TestDubbingPipeline:
    """Test dubbing pipeline module."""

    def test_pipeline_initialization(self):
        """Test DubbingPipeline initialization."""
        config = Config()
        pipeline = DubbingPipeline(config)
        assert pipeline.config == config
        assert pipeline.audio_processor is not None
        assert pipeline.voice_synthesis is not None
