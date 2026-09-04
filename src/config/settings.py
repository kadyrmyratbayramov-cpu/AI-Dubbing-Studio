"""Application configuration settings."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Config:
    """Runtime configuration for AI Dubbing Studio."""

    # audio/video processing
    sample_rate: int = 22050
    extraction_sample_rate: int = 16000
    channels: int = 1
    bit_depth: int = 16
    segment_seconds: int = 45

    # directories
    output_dir: str = "output"
    jobs_dir: str = "jobs"
    temp_dir: str = "tmp"
    log_dir: str = "logs"

    # model settings
    whisper_model: str = "base"
    tts_model: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    translation_model_map: Dict[str, str] = field(
        default_factory=lambda: {
            "en-es": "Helsinki-NLP/opus-mt-en-es",
            "es-en": "Helsinki-NLP/opus-mt-es-en",
            "en-fr": "Helsinki-NLP/opus-mt-en-fr",
            "fr-en": "Helsinki-NLP/opus-mt-fr-en",
            "en-de": "Helsinki-NLP/opus-mt-en-de",
            "de-en": "Helsinki-NLP/opus-mt-de-en",
            "en-pt": "Helsinki-NLP/opus-mt-en-ROMANCE",
            "pt-en": "Helsinki-NLP/opus-mt-ROMANCE-en",
        }
    )
    default_language_pairs: List[str] = field(default_factory=lambda: ["en-es", "es-en"])

    # resource settings
    force_cpu: bool = False
    max_vram_gb: float = 6.0
    logging_level: str = "INFO"

    # external auth
    huggingface_token: str = ""

    # optional user preference
    tts_voice: str = "neutral"

    def __post_init__(self) -> None:
        self.normalize_paths()
        self.ensure_runtime_dirs()

    @classmethod
    def load(cls, config_file: Optional[str] = None) -> "Config":
        """Load configuration from optional file or default path."""
        default_path = Path("config/config.yaml")
        path = Path(config_file) if config_file else default_path
        config = cls()
        if path.exists():
            config.load_from_file(str(path))
            config.normalize_paths()
            config.ensure_runtime_dirs()
        env_token = os.getenv("HUGGINGFACE_TOKEN")
        if env_token:
            config.huggingface_token = env_token
        return config

    def normalize_paths(self) -> None:
        self.output_dir = str(Path(self.output_dir).expanduser())
        self.jobs_dir = str(Path(self.jobs_dir).expanduser())
        self.temp_dir = str(Path(self.temp_dir).expanduser())
        self.log_dir = str(Path(self.log_dir).expanduser())

    def ensure_runtime_dirs(self) -> None:
        for path in (self.output_dir, self.jobs_dir, self.temp_dir, self.log_dir):
            Path(path).mkdir(parents=True, exist_ok=True)

    def load_from_file(self, config_file: str) -> None:
        ext = Path(config_file).suffix.lower()
        if ext in {".yaml", ".yml"}:
            self._load_yaml(config_file)
        elif ext == ".json":
            self._load_json(config_file)
        else:
            raise ValueError(f"Unsupported config format: {ext}")

    def _load_yaml(self, file_path: str) -> None:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("PyYAML is required to load YAML configuration") from exc

        with open(file_path, "r", encoding="utf-8") as file:
            raw = yaml.safe_load(file) or {}
        flat = self._flatten_legacy_config(raw)
        self._update_from_dict(flat)

    def _load_json(self, file_path: str) -> None:
        with open(file_path, "r", encoding="utf-8") as file:
            raw = json.load(file) or {}
        flat = self._flatten_legacy_config(raw)
        self._update_from_dict(flat)

    def _flatten_legacy_config(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if not any(k in data for k in ("audio", "model", "processing", "output", "dubbing")):
            return data

        flattened: Dict[str, Any] = {}
        audio = data.get("audio", {})
        output = data.get("output", {})
        dubbing = data.get("dubbing", {})
        model = data.get("model", {})

        flattened.update(
            {
                "sample_rate": audio.get("sample_rate", self.sample_rate),
                "channels": audio.get("channels", self.channels),
                "bit_depth": audio.get("bit_depth", self.bit_depth),
                "output_dir": output.get("output_dir", self.output_dir),
                "log_dir": output.get("log_dir", self.log_dir),
                "tts_voice": model.get("default_voice", self.tts_voice),
            }
        )
        src = dubbing.get("language", "en")
        tgt = dubbing.get("target_language", "es")
        flattened["default_language_pairs"] = [f"{src}-{tgt}"]
        return flattened

    def _update_from_dict(self, config_dict: Dict[str, Any]) -> None:
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as file:
            if target.suffix.lower() in {".yaml", ".yml"}:
                try:
                    import yaml
                except ImportError as exc:
                    raise ImportError("PyYAML is required to save YAML configuration") from exc
                yaml.safe_dump(self.to_dict(), file, sort_keys=False)
            else:
                json.dump(self.to_dict(), file, indent=2)

    def language_pairs(self) -> List[Tuple[str, str]]:
        pairs: List[Tuple[str, str]] = []
        for pair in self.default_language_pairs:
            if "-" not in pair:
                continue
            src, tgt = pair.split("-", 1)
            pairs.append((src, tgt))
        return pairs
