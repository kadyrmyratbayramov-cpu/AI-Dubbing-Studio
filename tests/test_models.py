"""Tests for model modules."""

from __future__ import annotations

from src.config.settings import Config
from src.models.model_loader import ModelLoader
from src.models.speech_to_text import SpeechToTextEngine
from src.models.voice_synthesis import VoiceSynthesis


class TestModelLoader:
    def test_model_loader_initialization(self):
        config = Config()
        loader = ModelLoader(config)
        assert loader.config == config
        assert isinstance(loader.model_cache, dict)

    def test_model_cache_clear(self):
        config = Config()
        loader = ModelLoader(config)
        loader.model_cache["test"] = "model"
        loader.clear_cache()
        assert len(loader.model_cache) == 0


class TestVoiceSynthesis:
    def test_voice_synthesis_initialization(self):
        config = Config()
        synthesis = VoiceSynthesis(config)
        assert synthesis.config == config
        assert synthesis.model_loader is not None

    def test_get_available_voices(self):
        config = Config()
        synthesis = VoiceSynthesis(config)
        voices = synthesis.get_available_voices()
        assert isinstance(voices, dict)
        assert len(voices) > 0


class TestSpeechToText:
    def test_engine_info(self):
        config = Config()
        engine = SpeechToTextEngine(config)
        assert engine.info()["engine"] == "faster-whisper"
