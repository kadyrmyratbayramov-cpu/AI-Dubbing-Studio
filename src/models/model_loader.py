"""Model loading and lifecycle management utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from src.config.settings import Config
from src.core.gpu_manager import GPUManager


@dataclass
class LoadedModel:
    name: str
    kind: str
    model: Any


class ModelLoader:
    """Model loader with cache and controlled unload support."""

    def __init__(self, config: Config):
        self.config = config
        self.gpu = GPUManager(force_cpu=config.force_cpu)
        self.model_cache: Dict[str, LoadedModel] = {}

    def load(self, model_name: str, kind: str = "generic") -> LoadedModel:
        cache_key = f"{kind}:{model_name}"
        if cache_key in self.model_cache:
            return self.model_cache[cache_key]

        model = self._load_model(model_name, kind)
        loaded = LoadedModel(name=model_name, kind=kind, model=model)
        self.model_cache[cache_key] = loaded
        return loaded

    def _load_model(self, model_name: str, kind: str) -> Any:
        if kind == "whisper":
            import whisper

            return whisper.load_model(model_name, device=self.gpu.preferred_device())
        if kind == "translation":
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            return {
                "tokenizer": AutoTokenizer.from_pretrained(model_name),
                "model": AutoModelForSeq2SeqLM.from_pretrained(model_name),
            }
        if kind == "tts":
            from TTS.api import TTS

            return TTS(model_name, gpu=self.gpu.cuda_available())

        raise RuntimeError(f"Unknown model kind: {kind}")

    def unload(self, model_name: str, kind: str = "generic") -> None:
        cache_key = f"{kind}:{model_name}"
        item = self.model_cache.pop(cache_key, None)
        if item is not None:
            self.gpu.unload_model(item.model)

    def clear_cache(self) -> None:
        for model in list(self.model_cache.values()):
            if isinstance(model, LoadedModel):
                self.gpu.unload_model(model.model)
        self.model_cache.clear()
