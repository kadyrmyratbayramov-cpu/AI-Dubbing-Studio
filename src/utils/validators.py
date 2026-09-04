"""Input/output validation utilities."""

from __future__ import annotations

import os
from pathlib import Path


def validate_input_file(file_path: str) -> bool:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")
    if not os.access(path, os.R_OK):
        raise PermissionError(f"File is not readable: {file_path}")
    return True


def validate_output_path(file_path: str) -> bool:
    path = Path(file_path)
    directory = path.parent
    if directory and not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)
    if directory and not os.access(directory, os.W_OK):
        raise PermissionError(f"Output directory is not writable: {directory}")
    return True


def validate_audio_format(file_path: str) -> bool:
    supported = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac")
    if Path(file_path).suffix.lower() not in supported:
        raise ValueError(f"Unsupported audio format. Supported: {supported}")
    return True


def validate_video_format(file_path: str) -> bool:
    supported = (".mp4", ".mkv", ".mov", ".avi", ".m4v", ".webm")
    if Path(file_path).suffix.lower() not in supported:
        raise ValueError(f"Unsupported video format. Supported: {supported}")
    return True
