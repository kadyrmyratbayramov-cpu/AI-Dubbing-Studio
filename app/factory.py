"""Application factory for the runnable AI Dubbing Studio foundation."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from src import __version__
from src.config.settings import Config

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
DEFAULT_DOTENV_PATH = PROJECT_ROOT / ".env"
DEFAULT_ENVIRONMENT = "development"
ENV_PREFIX = "AI_DUBBING_"
CONFIG_SECTIONS = {
    "audio",
    "model",
    "processing",
    "output",
    "dubbing",
    "runtime",
}


@dataclass
class Application:
    """Runtime application container."""

    config: Config
    environment: str
    root_path: Path
    config_path: Optional[Path]
    runtime_settings: Dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return "AI Dubbing Studio"

    @property
    def version(self) -> str:
        return __version__

    @property
    def static_path(self) -> Path:
        return self.root_path / "web" / "static"

    @property
    def templates_path(self) -> Path:
        return self.root_path / "web" / "templates"

    @property
    def output_path(self) -> Path:
        return (self.root_path / self.config.output_dir).resolve()

    @property
    def log_path(self) -> Path:
        return (self.root_path / self.config.log_dir).resolve()

    def to_metadata(self) -> Dict[str, Any]:
        """Return serializable runtime metadata."""
        return {
            "name": self.name,
            "version": self.version,
            "environment": self.environment,
            "config_path": str(self.config_path) if self.config_path else None,
            "paths": {
                "root": str(self.root_path),
                "static": str(self.static_path),
                "templates": str(self.templates_path),
                "output": str(self.output_path),
                "logs": str(self.log_path),
            },
            "config": self.config.to_dict(),
            "runtime": dict(self.runtime_settings),
        }


def create_application(
    config_path: Optional[str] = None,
    environment: Optional[str] = None,
) -> Application:
    """Create an initialized application instance."""
    _load_dotenv(DEFAULT_DOTENV_PATH)
    env_name = f"{ENV_PREFIX}ENV"
    runtime_environment = environment or os.getenv(
        env_name,
        DEFAULT_ENVIRONMENT,
    )
    resolved_config_path = _resolve_config_path(config_path)

    config = Config()
    runtime_settings: Dict[str, Any] = {}

    if resolved_config_path:
        runtime_settings.update(
            _apply_config_file(config, resolved_config_path)
        )

    environment_path = PROJECT_ROOT / "config" / f"{runtime_environment}.yaml"
    if environment_path.exists():
        runtime_settings.update(_apply_config_file(config, environment_path))

    runtime_settings.update(_apply_environment_overrides(config))

    application = Application(
        config=config,
        environment=runtime_environment,
        root_path=PROJECT_ROOT,
        config_path=resolved_config_path,
        runtime_settings=runtime_settings,
    )
    _ensure_runtime_directories(application)
    return application


def _load_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), _strip_quotes(value.strip()))


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _resolve_config_path(config_path: Optional[str]) -> Optional[Path]:
    candidate = (
        Path(config_path).expanduser()
        if config_path
        else DEFAULT_CONFIG_PATH
    )
    if not candidate.exists():
        if config_path:
            raise FileNotFoundError(
                f"Configuration file not found: {candidate}"
            )
        return None
    return candidate.resolve()


def _apply_config_file(config: Config, config_path: Path) -> Dict[str, Any]:
    payload = _read_config_file(config_path)
    section_payload = _extract_section_values(payload)
    return _apply_mapping(config, section_payload)


def _read_config_file(config_path: Path) -> Dict[str, Any]:
    suffix = config_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "PyYAML is required to read YAML runtime configuration"
            ) from exc
        with config_path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    if suffix == ".json":
        with config_path.open("r", encoding="utf-8") as handle:
            return json.load(handle) or {}
    raise ValueError(f"Unsupported configuration format: {config_path.suffix}")


def _extract_section_values(data: Dict[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for key, value in data.items():
        if key in CONFIG_SECTIONS and isinstance(value, dict):
            payload.update(value)
        else:
            payload[key] = value
    return payload


def _apply_mapping(config: Config, payload: Dict[str, Any]) -> Dict[str, Any]:
    runtime: Dict[str, Any] = {}
    for key, value in payload.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            runtime[key] = value
    return runtime


def _apply_environment_overrides(config: Config) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    mapping = {
        "debug": f"{ENV_PREFIX}DEBUG",
        "output_dir": f"{ENV_PREFIX}OUTPUT_DIR",
        "log_dir": f"{ENV_PREFIX}LOG_DIR",
        "host": f"{ENV_PREFIX}HOST",
        "api_port": f"{ENV_PREFIX}API_PORT",
        "web_port": f"{ENV_PREFIX}WEB_PORT",
    }

    for key, env_name in mapping.items():
        raw_value = os.getenv(env_name)
        if raw_value is None:
            continue
        value = _coerce_value(raw_value)
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            overrides[key] = value
    return overrides


def _coerce_value(value: str) -> Any:
    lowered = value.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered.isdigit():
        return int(lowered)
    try:
        return float(lowered)
    except ValueError:
        return value


def _ensure_runtime_directories(application: Application) -> None:
    application.output_path.mkdir(parents=True, exist_ok=True)
    application.log_path.mkdir(parents=True, exist_ok=True)


__all__ = ["Application", "DEFAULT_CONFIG_PATH", "create_application"]
