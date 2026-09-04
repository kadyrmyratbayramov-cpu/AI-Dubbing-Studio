"""Tests for model modules."""

import numpy as np

from src.config.settings import Config
from src.models.model_loader import ModelLoader
from src.models.voice_synthesis import VoiceSynthesis


class TestModelLoader:
    """Test model loading module."""

    def test_model_loader_initialization(self):
        """Test ModelLoader initialization."""
        config = Config()
        loader = ModelLoader(config)
        assert loader.config == config
        assert isinstance(loader.model_cache, dict)

    def test_load_uses_cache(self, monkeypatch):
        """Test model loading caches values."""
        loader = ModelLoader(Config())
        calls = {"count": 0}

        def fake_load(name):
            calls["count"] += 1
            return {"name": name}

        monkeypatch.setattr(loader, "_load_model", fake_load)

        first = loader.load("tts")
        second = loader.load("tts")

        assert first == {"name": "tts"}
        assert second == first
        assert calls["count"] == 1

    def test_load_model_default(self):
        """Test default internal model load implementation."""
        loader = ModelLoader(Config())
        assert loader._load_model("unknown") is None

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

    def test_synthesize_returns_audio(self):
        """Test synthesize scaffold behavior."""
        synthesis = VoiceSynthesis(Config())
        audio = np.array([0.1, -0.2])
        output = synthesis.synthesize(audio, language="en", speaker="default")
        assert np.array_equal(output, audio)

    def test_load_model_uses_loader(self, monkeypatch):
        """Test model loading through synthesis wrapper."""
        synthesis = VoiceSynthesis(Config())
        calls = {"count": 0}

        def fake_load(name):
            calls["count"] += 1
            return f"model:{name}"

        monkeypatch.setattr(synthesis.model_loader, "load", fake_load)

        synthesis.load_model("voice_a")
        synthesis.load_model("voice_a")

        assert synthesis.models["voice_a"] == "model:voice_a"
        assert calls["count"] == 1

    def test_get_available_voices(self):
        """Test getting available voices."""
        config = Config()
        synthesis = VoiceSynthesis(config)
        voices = synthesis.get_available_voices()
        assert isinstance(voices, dict)
        assert len(voices) > 0
