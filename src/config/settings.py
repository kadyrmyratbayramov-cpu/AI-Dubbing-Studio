"""Application configuration settings."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


class Config:
    """Central configuration for AI Dubbing Studio."""

    def __init__(self, config_file: Optional[str] = None):
        self.repo_root = Path(__file__).resolve().parents[2]
        self.config_file = Path(config_file) if config_file else self.repo_root / "config" / "config.yaml"

        # Legacy-compatible defaults
        self.sample_rate = 22050
        self.channels = 1
        self.bit_depth = 16
        self.frame_length = 2048
        self.hop_length = 512
        self.model_dir = "models"
        self.default_model = "default_tts"
        self.default_voice = "en_US"
        self.default_speaker = "default"
        self.normalize_audio = True
        self.remove_silence = False
        self.silence_threshold = 0.01
        self.fade_in_duration = 0.1
        self.fade_out_duration = 0.1
        self.output_dir = "output"
        self.log_dir = "logs"
        self.debug = False

        # Production pipeline defaults
        self.jobs_dir = "jobs"
        self.workspace_dir = "workspace"
        self.models_cache_dir = "models"
        self.log_level = "INFO"
        self.ffmpeg_bin = "ffmpeg"
        self.ffprobe_bin = "ffprobe"
        self.segment_seconds = 45
        self.retry_attempts = 3
        self.cleanup_intermediates = False

        self.device = "auto"
        self.max_vram_mb = 7600

        self.whisper_model = "small"
        self.pyannote_model = "pyannote/speaker-diarization"
        self.xtts_model = "tts_models/multilingual/multi-dataset/xtts_v2"
        self.stt_sample_rate = 16000
        self.hf_token = os.getenv("HUGGINGFACE_TOKEN")

        if self.config_file.exists():
            self.load_from_file(str(self.config_file))

        self._normalize_and_create_paths()

    def resolve_path(self, path_value: str) -> Path:
        path = Path(path_value)
        return path if path.is_absolute() else (self.repo_root / path).resolve()

    def _normalize_and_create_paths(self) -> None:
        self.model_dir = str(self.resolve_path(str(self.model_dir)))
        self.models_cache_dir = str(self.resolve_path(str(self.models_cache_dir)))
        self.output_dir = str(self.resolve_path(str(self.output_dir)))
        self.log_dir = str(self.resolve_path(str(self.log_dir)))
        self.jobs_dir = str(self.resolve_path(str(self.jobs_dir)))
        self.workspace_dir = str(self.resolve_path(str(self.workspace_dir)))

        for path in [self.model_dir, self.models_cache_dir, self.output_dir, self.log_dir, self.jobs_dir, self.workspace_dir]:
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

        with open(file_path, "r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}

        if any(key in payload for key in ["audio", "model", "processing", "output", "pipeline", "engines", "system"]):
            payload = self._flatten_structured_config(payload)
        self._update_from_dict(payload)

    def _flatten_structured_config(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        flat: Dict[str, Any] = {}
        for section in ["audio", "model", "processing", "output", "pipeline", "engines", "system", "dubbing"]:
            values = payload.get(section, {})
            if isinstance(values, dict):
                flat.update(values)

        # Backwards-compatible nested keys
        if "cache" in payload and isinstance(payload["cache"], dict):
            flat.update(payload["cache"])
        return flat

    def _load_json(self, file_path: str) -> None:
        with open(file_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self._update_from_dict(payload)

    def _update_from_dict(self, config_dict: Dict[str, Any]) -> None:
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        return {key: value for key, value in self.__dict__.items() if not key.startswith("_")}

    def __repr__(self) -> str:
        return f"Config({self.to_dict()})"
