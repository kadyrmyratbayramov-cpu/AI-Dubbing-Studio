"""Model loading and management utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from src.config.settings import Config


class ModelLoader:
    """Handles loading and caching of AI models."""

    def __init__(self, config: Config):
        self.config = config
        self.model_cache = {}
        Path(self.config.model_dir).mkdir(parents=True, exist_ok=True)

    def load(self, model_name: str) -> Optional[Any]:
        if model_name in self.model_cache:
            return self.model_cache[model_name]

        model = self._load_model(model_name)
        self.model_cache[model_name] = model
        return model

    def _load_model(self, model_name: str) -> Optional[Any]:
        return None

    def clear_cache(self) -> None:
        self.model_cache.clear()
