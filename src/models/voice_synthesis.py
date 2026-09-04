"""Compatibility wrapper for voice synthesis features."""

from __future__ import annotations

import numpy as np
from typing import Any, Dict

from src.config.settings import Config
from src.models.model_loader import ModelLoader
from src.models.text_to_speech import TextToSpeechEngine


class VoiceSynthesis:
    def __init__(self, config: Config):
        self.config = config
        self.model_loader = ModelLoader(config)
        self.models = {}
        self.tts_engine = TextToSpeechEngine(config)

    def synthesize(self, audio_data: np.ndarray, **kwargs) -> np.ndarray:
        return audio_data

    def load_model(self, model_name: str) -> None:
        if model_name not in self.models:
            self.models[model_name] = self.model_loader.load(model_name)

    def get_available_voices(self) -> Dict[str, Any]:
        return {
            "en_US": {"languages": ["English"], "speakers": ["default"]},
            "tr_TR": {"languages": ["Turkish"], "speakers": ["default"]},
            "es_ES": {"languages": ["Spanish"], "speakers": ["default"]},
        }
