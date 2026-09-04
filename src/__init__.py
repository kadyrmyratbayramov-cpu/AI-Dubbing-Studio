"""AI Dubbing Studio package."""

__version__ = "2.0.0"
__author__ = "AI Dubbing Studio Team"

from src.config.settings import Config
from src.core.dubbing_pipeline import DubbingPipeline

__all__ = ["Config", "DubbingPipeline"]
