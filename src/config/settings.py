"""Application configuration settings."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


class Config:
    """Main configuration class for AI Dubbing Studio."""

    def __init__(self, config_file: Optional[str] = None):
        repo_root = Path(__file__).resolve().parents[2]
        self.repo_root = str(repo_root)
        self.sample_rate = 16000
        self.channels = 1
        self.bit_depth = 16
        self.frame_length = 2048
        self.hop_length = 512

        self.model_dir = str(repo_root / "models")
        self.default_model = "whisper-small"
        self.default_voice = "en_US"
        self.default_speaker = "default"
        self.stt_model_name = "small"
        self.stt_device = "auto"
        self.stt_compute_type = "auto"
        self.translation_model_name = "Helsinki-NLP/opus-mt-en-tr"
        self.tts_model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
        self.diarization_model_name = "pyannote/speaker-diarization-3.1"
        self.lipsync_model_name = "yolov8n-face.pt"

        self.normalize_audio = True
        self.remove_silence = False
        self.silence_threshold = 0.01
        self.fade_in_duration = 0.1
        self.fade_out_duration = 0.1
        self.segment_duration_seconds = 30
        self.max_transcription_segments = 1
        self.max_stage_retries = 2

        self.output_dir = str(repo_root / "output")
        self.log_dir = str(repo_root / "logs")
        self.checkpoint_dir = str(repo_root / "output" / "checkpoints")
        self.workspace_dir = str(repo_root / "output" / "workspace")
        self.debug = False

        self.ffmpeg_binary = "ffmpeg"
        self.ffprobe_binary = "ffprobe"
        self.ffmpeg_audio_codec = "pcm_s16le"
        self.video_probe_timeout_seconds = 30
        self.stage_retry_backoff_seconds = 1

        self.available_languages = [
            {"code": "auto", "label": "Auto Detect"},
            {"code": "en", "label": "English"},
            {"code": "tr", "label": "Turkish"},
            {"code": "es", "label": "Spanish"},
            {"code": "de", "label": "German"},
            {"code": "fr", "label": "French"},
            {"code": "it", "label": "Italian"},
            {"code": "pt", "label": "Portuguese"},
        ]

        self.auto_load_default_config = True
        default_config = repo_root / "config" / "config.yaml"
        if self.auto_load_default_config and default_config.exists():
            self.load_from_file(str(default_config))
        if config_file and os.path.exists(config_file):
            self.load_from_file(config_file)

        self.ensure_runtime_directories()

    def ensure_runtime_directories(self) -> None:
        for directory in [self.output_dir, self.log_dir, self.checkpoint_dir, self.workspace_dir, self.model_dir]:
            Path(directory).mkdir(parents=True, exist_ok=True)

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

            with open(file_path, "r", encoding="utf-8") as file:
                config_dict = yaml.safe_load(file) or {}
        except ImportError as exc:
            raise ImportError("PyYAML is required to load YAML configuration") from exc
        except Exception as exc:
            raise RuntimeError(f"Failed to load YAML config: {exc}") from exc
        self._update_from_dict(config_dict)

    def _load_json(self, file_path: str) -> None:
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                config_dict = json.load(file)
        except Exception as exc:
            raise RuntimeError(f"Failed to load JSON config: {exc}") from exc
        self._update_from_dict(config_dict)

    def _update_from_dict(self, config_dict: Dict[str, Any]) -> None:
        nested_key_map = {
            "audio": {"sample_rate", "channels", "bit_depth", "frame_length", "hop_length"},
            "model": {
                "model_dir",
                "default_model",
                "default_voice",
                "default_speaker",
                "stt_model_name",
                "translation_model_name",
                "tts_model_name",
                "diarization_model_name",
                "lipsync_model_name",
            },
            "processing": {
                "normalize_audio",
                "remove_silence",
                "silence_threshold",
                "fade_in_duration",
                "fade_out_duration",
                "segment_duration_seconds",
                "max_transcription_segments",
                "max_stage_retries",
            },
            "output": {"output_dir", "log_dir", "checkpoint_dir", "workspace_dir", "debug"},
            "media": {"ffmpeg_binary", "ffprobe_binary", "ffmpeg_audio_codec", "video_probe_timeout_seconds"},
            "dubbing": {"source_language", "target_language"},
        }
        for key, value in config_dict.items():
            if isinstance(value, dict) and key in nested_key_map:
                for nested_key, nested_value in value.items():
                    if nested_key in nested_key_map[key] and hasattr(self, nested_key):
                        setattr(self, nested_key, nested_value)
                continue
            if hasattr(self, key):
                setattr(self, key, value)
        self.ensure_runtime_directories()

    def to_dict(self) -> Dict[str, Any]:
        return {key: value for key, value in self.__dict__.items() if not key.startswith("_")}

    def __repr__(self) -> str:
        return f"Config({self.to_dict()})"
