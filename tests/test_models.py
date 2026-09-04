"""Tests for model modules."""

import pytest
from src.config.settings import Config
from src.models.voice_synthesis import VoiceSynthesis
from src.models.model_loader import ModelLoader


class TestModelLoader:
    """Test model loading module."""

    def test_model_loader_initialization(self):
        """Test ModelLoader initialization."""
        config = Config()
        loader = ModelLoader(config)
        assert loader.config == config
        assert isinstance(loader.model_cache, dict)

    def test_model_cache_clear(self):
        """Test model cache clearing."""
        config = Config()
        loader = ModelLoader(config)
        loader.model_cache["test"] = "model"
        loader.clear_cache()
        assert len(loader.model_cache) == 0


class TestVoiceSynthesis:
    """Test voice synthesis module."""

    def test_voice_synthesis_initialization(self):
        """Test VoiceSynthesis initialization."""
        config = Config()
        synthesis = VoiceSynthesis(config)
        assert synthesis.config == config
        assert synthesis.model_loader is not None

    def test_get_available_voices(self):
        """Test getting available voices."""
        config = Config()
        synthesis = VoiceSynthesis(config)
        voices = synthesis.get_available_voices()
        assert isinstance(voices, dict)
        assert len(voices) > 0
