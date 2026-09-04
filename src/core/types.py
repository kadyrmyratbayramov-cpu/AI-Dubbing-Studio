"""Shared domain types for dubbing pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class AudioStreamInfo:
    index: int
    codec: str
    channels: int
    sample_rate: int
    language: str = "und"


@dataclass
class VideoMetadata:
    path: str
    duration: float
    width: int
    height: int
    fps: float
    video_codec: str
    audio_codec: str
    bitrate: int
    audio_streams: List[AudioStreamInfo] = field(default_factory=list)
    subtitle_streams: int = 0


@dataclass
class Segment:
    id: str
    index: int
    start: float
    end: float
    path: str

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class TranscriptChunk:
    start: float
    end: float
    text: str
    confidence: float
    speaker: str = "SPEAKER_00"
    words: List[Dict[str, float | str]] = field(default_factory=list)


@dataclass
class SegmentResult:
    segment: Segment
    transcript: List[TranscriptChunk]
    translated_text: str
    speaker: str
    synthesized_path: str


@dataclass
class JobPaths:
    job_id: str
    root: Path
    segments_dir: Path
    synthesized_dir: Path
    artifacts_dir: Path
    checkpoint_file: Path
    manifest_file: Path


@dataclass
class PipelineStatus:
    stage: str
    progress: float
    eta_seconds: Optional[float] = None
    message: str = ""
