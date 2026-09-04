"""AI Dubbing Studio - desktop-first local dubbing pipeline."""

__version__ = "1.1.0"
__author__ = "AI Dubbing Studio Team"

from src.config.settings import Config
from src.core.dubbing_pipeline import DubbingPipeline

__all__ = ["Config", "DubbingPipeline"]
