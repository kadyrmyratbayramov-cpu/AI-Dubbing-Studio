"""Voice synthesis service wrapping Coqui XTTS."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from src.config.settings import Config
from src.core.gpu_manager import GPUManager
from src.models.model_loader import ModelLoader


class VoiceSynthesis:
    """Handles voice synthesis and TTS generation."""

    def __init__(self, config: Config):
        self.config = config
        self.model_loader = ModelLoader(config)
        self.gpu = GPUManager(force_cpu=config.force_cpu)

    def synthesize(
        self,
        audio_data: np.ndarray,
        language: str = "en",
        speaker: str = "neutral",
        **kwargs: Any,
    ) -> np.ndarray:
        text = kwargs.get("text")
        if not text:
            return audio_data

        output_path = kwargs.get("output_path")
        if not output_path:
            raise ValueError("output_path is required when synthesizing from text")

        model_name = kwargs.get("model_name", self.config.tts_model)
        speaker_wav = kwargs.get("speaker_wav")
        loaded = self.model_loader.load(model_name, kind="tts")
        tts = loaded.model

        tts.tts_to_file(text=text, file_path=output_path, language=language, speaker_wav=speaker_wav)
        import soundfile as sf

        generated, _ = sf.read(output_path)
        if generated.ndim > 1:
            generated = np.mean(generated, axis=1)
        return generated.astype(np.float32)

    def load_model(self, model_name: str) -> None:
        self.model_loader.load(model_name, kind="tts")

    def get_available_voices(self) -> Dict[str, Any]:
        return {
            "xtts_multilingual": {
                "languages": ["en", "es", "fr", "de", "pt", "it", "nl", "tr", "ru", "pl"],
                "speakers": ["neutral"],
                "voice_clone_requires_reference": True,
            }
        }
