"""Model loading and caching utilities."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from src.config.settings import Config


class ModelLoader:
    def __init__(self, config: Config):
        self.config = config
        self.model_cache: Dict[str, Any] = {}
        self._factories: Dict[str, Callable[[], Any]] = {}

    def register(self, model_name: str, factory: Callable[[], Any]) -> None:
        self._factories[model_name] = factory

    def load(self, model_name: str) -> Optional[Any]:
        if model_name in self.model_cache:
            return self.model_cache[model_name]
        factory = self._factories.get(model_name)
        model = factory() if factory else self._load_model(model_name)
        self.model_cache[model_name] = model
        return model

    def _load_model(self, model_name: str) -> Optional[Any]:
        return None

    def clear_cache(self) -> None:
        self.model_cache.clear()
