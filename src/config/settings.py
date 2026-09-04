"""Application configuration settings."""

import os
import json
from typing import Optional, Dict, Any
from pathlib import Path


class Config:
    """Main configuration class for AI Dubbing Studio."""

    def __init__(self, config_file: Optional[str] = None):
        """Initialize configuration.

        Args:
            config_file: Path to configuration file (optional)
        """
        # Default settings
        self.sample_rate = 22050
        self.channels = 1
        self.bit_depth = 16
        self.frame_length = 2048
        self.hop_length = 512

        # Model settings
        self.model_dir = "models"
        self.default_model = "default_tts"
        self.default_voice = "en_US"
        self.default_speaker = "default"
        self.cache_models = True

        # Processing settings
        self.normalize_audio = True
        self.remove_silence = False
        self.silence_threshold = 0.01
        self.fade_in_duration = 0.1
        self.fade_out_duration = 0.1

        # Output settings
        self.output_dir = "output"
        self.log_dir = "logs"
        self.debug = False

        # Dubbing settings
        self.language = "en"
        self.target_language = "en"
        self.voice_speed = 1.0
        self.pitch_shift = 0.0

        # Load from file if provided
        if config_file and os.path.exists(config_file):
            self.load_from_file(config_file)

    def load_from_file(self, config_file: str) -> None:
        """Load configuration from file.

        Args:
            config_file: Path to configuration file (YAML or JSON)
        """
        ext = Path(config_file).suffix.lower()

        if ext == ".yaml" or ext == ".yml":
            self._load_yaml(config_file)
        elif ext == ".json":
            self._load_json(config_file)
        else:
            raise ValueError(f"Unsupported config format: {ext}")

    def _load_yaml(self, file_path: str) -> None:
        """Load YAML configuration.

        Args:
            file_path: Path to YAML file
        """
        try:
            import yaml
            with open(file_path, 'r') as f:
                config_dict = yaml.safe_load(f) or {}
                self._update_from_dict(config_dict)
        except ImportError:
            raise ImportError("PyYAML is required to load YAML configuration")
        except Exception as e:
            raise RuntimeError(f"Failed to load YAML config: {str(e)}")

    def _load_json(self, file_path: str) -> None:
        """Load JSON configuration.

        Args:
            file_path: Path to JSON file
        """
        try:
            with open(file_path, 'r') as f:
                config_dict = json.load(f)
                self._update_from_dict(config_dict)
        except Exception as e:
            raise RuntimeError(f"Failed to load JSON config: {str(e)}")

    def _update_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration from dictionary.

        Args:
            config_dict: Configuration dictionary
        """
        sections = ("audio", "model", "processing", "output", "dubbing")

        for section in sections:
            section_values = config_dict.get(section)
            if isinstance(section_values, dict):
                self._update_flat_values(section_values)

        self._update_flat_values(config_dict)

    def _update_flat_values(self, values: Dict[str, Any]) -> None:
        """Update only known top-level config attributes from flat key-value pairs."""
        for key, value in values.items():
            if isinstance(value, dict):
                continue
            if hasattr(self, key):
                setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Configuration as dictionary
        """
        return {
            key: value for key, value in self.__dict__.items()
            if not key.startswith('_')
        }

    def __repr__(self) -> str:
        """String representation of configuration."""
        config_dict = self.to_dict()
        return f"Config({config_dict})"
