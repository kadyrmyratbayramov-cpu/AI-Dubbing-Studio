"""Video metadata utilities backed by ffprobe."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from src.config.settings import Config
from src.utils.validators import validate_input_file


@dataclass
class VideoMetadata:
    file_path: str
    file_size_bytes: int
    duration_seconds: float
    width: int
    height: int
    frame_rate: float
    video_codec: str
    audio_codec: str
    audio_sample_rate: Optional[int]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class VideoMetadataReader:
    def __init__(self, config: Config):
        self.config = config

    def probe(self, file_path: str) -> VideoMetadata:
        validate_input_file(file_path)
        command = [
            self.config.ffprobe_binary,
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=index,codec_type,codec_name,width,height,avg_frame_rate,sample_rate",
            "-of",
            "json",
            file_path,
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
                timeout=self.config.video_probe_timeout_seconds,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("ffprobe was not found on PATH") from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(exc.stderr.strip() or "ffprobe failed") from exc
        data = json.loads(completed.stdout or "{}")
        return self._parse_metadata(file_path, data)

    def _parse_metadata(self, file_path: str, data: Dict[str, Any]) -> VideoMetadata:
        video_stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), {})
        audio_stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), {})
        duration_seconds = float(data.get("format", {}).get("duration") or 0.0)
        frame_rate = self._parse_frame_rate(video_stream.get("avg_frame_rate"))
        audio_sample_rate = audio_stream.get("sample_rate")
        return VideoMetadata(
            file_path=file_path,
            file_size_bytes=Path(file_path).stat().st_size,
            duration_seconds=duration_seconds,
            width=int(video_stream.get("width") or 0),
            height=int(video_stream.get("height") or 0),
            frame_rate=frame_rate,
            video_codec=video_stream.get("codec_name") or "unknown",
            audio_codec=audio_stream.get("codec_name") or "unknown",
            audio_sample_rate=int(audio_sample_rate) if audio_sample_rate else None,
        )

    @staticmethod
    def _parse_frame_rate(value: Optional[str]) -> float:
        if not value or value in {"0/0", "N/A"}:
            return 0.0
        if "/" in value:
            numerator, denominator = value.split("/", 1)
            if float(denominator) == 0:
                return 0.0
            return float(numerator) / float(denominator)
        return float(value)
