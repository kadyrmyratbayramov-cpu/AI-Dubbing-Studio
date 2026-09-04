"""Chunked FFmpeg media processing helpers."""

from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from src.config.settings import Config
from src.core.video_metadata import VideoMetadata, VideoMetadataReader


@dataclass
class AudioSegment:
    index: int
    start_seconds: float
    duration_seconds: float
    path: str


class FFmpegMediaPipeline:
    def __init__(self, config: Config):
        self.config = config
        self.metadata_reader = VideoMetadataReader(config)

    def probe(self, file_path: str) -> VideoMetadata:
        return self.metadata_reader.probe(file_path)

    def iter_audio_segments(
        self,
        file_path: str,
        workspace_dir: Optional[str] = None,
    ) -> Iterator[AudioSegment]:
        metadata = self.probe(file_path)
        workspace = Path(workspace_dir or self.config.workspace_dir)
        workspace.mkdir(parents=True, exist_ok=True)
        segment_length = max(1, int(self.config.segment_duration_seconds))
        if metadata.duration_seconds <= 0:
            raise RuntimeError("Unable to determine media duration")
        segment_count = max(1, math.ceil(metadata.duration_seconds / segment_length))
        for index in range(segment_count):
            start_seconds = index * segment_length
            duration_seconds = min(
                segment_length,
                max(0.1, metadata.duration_seconds - start_seconds),
            )
            output_path = workspace / f"segment_{index:04d}.wav"
            self.extract_audio_segment(
                file_path,
                str(output_path),
                start_seconds,
                duration_seconds,
            )
            yield AudioSegment(
                index=index,
                start_seconds=float(start_seconds),
                duration_seconds=float(duration_seconds),
                path=str(output_path),
            )

    def extract_audio_segment(
        self,
        input_file: str,
        output_file: str,
        start_seconds: float,
        duration_seconds: float,
    ) -> str:
        command = [
            self.config.ffmpeg_binary,
            "-y",
            "-ss",
            str(start_seconds),
            "-t",
            str(duration_seconds),
            "-i",
            input_file,
            "-vn",
            "-ac",
            str(self.config.channels),
            "-ar",
            str(self.config.sample_rate),
            "-acodec",
            self.config.ffmpeg_audio_codec,
            output_file,
        ]
        try:
            subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("ffmpeg was not found on PATH") from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                exc.stderr.strip() or "ffmpeg extraction failed"
            ) from exc
        return output_file
