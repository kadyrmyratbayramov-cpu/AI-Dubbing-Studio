"""Centralized logging helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union


def setup_logger(name: str, log_dir: Union[Path, str], level: str = "INFO") -> logging.Logger:
    """Create or return configured logger."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)

    file_handler = logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")
    file_handler.setFormatter(fmt)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get logger without side effects."""
    return logging.getLogger(name)
