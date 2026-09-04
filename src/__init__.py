"""AI Dubbing Studio package."""

__version__ = "1.1.0"
__author__ = "AI Dubbing Studio Team"

from src.core.dubbing_pipeline import DubbingPipeline
from src.config.settings import Config

__all__ = ["DubbingPipeline", "Config"]
