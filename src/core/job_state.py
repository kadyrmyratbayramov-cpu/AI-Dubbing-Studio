"""Pipeline job state models."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from threading import Event
from typing import Any, Dict, List, Optional


class JobStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class PipelineStage(str, Enum):
    IDLE = "idle"
    PROBE = "probe"
    EXTRACT = "extract"
    TRANSCRIBE = "transcribe"
    DIARIZE = "diarize"
    TRANSLATE = "translate"
    SYNTHESIZE = "synthesize"
    TIMING = "timing"
    MIX = "mix"
    QUALITY_CONTROL = "quality_control"
    COMPLETE = "complete"


@dataclass
class PipelineRequest:
    input_file: str
    source_language: str = "auto"
    target_language: str = "en"
    output_dir: Optional[str] = None


@dataclass
class PipelineEvent:
    stage: str
    status: str
    message: str
    progress: float = 0.0
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineJobState:
    input_file: str
    source_language: str
    target_language: str
    output_dir: Optional[str] = None
    status: str = JobStatus.IDLE.value
    stage: str = PipelineStage.IDLE.value
    progress: float = 0.0
    attempts: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    transcript_preview: List[Dict[str, Any]] = field(default_factory=list)
    completed_segments: List[int] = field(default_factory=list)
    last_error: Optional[str] = None

    def checkpoint_path(self, checkpoint_dir: str) -> Path:
        stem = Path(self.input_file).stem.replace(" ", "_") or "job"
        safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", stem).strip("._") or "job"
        normalized = str(Path(self.input_file).resolve())
        suffix = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]
        return Path(checkpoint_dir) / f"{safe_name}-{suffix}.json"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_request(cls, request: PipelineRequest) -> "PipelineJobState":
        return cls(
            input_file=request.input_file,
            source_language=request.source_language,
            target_language=request.target_language,
            output_dir=request.output_dir,
        )


class JobController:
    def __init__(self) -> None:
        self._pause_requested = Event()
        self._stop_requested = Event()

    def pause(self) -> None:
        self._pause_requested.set()

    def resume(self) -> None:
        self._pause_requested.clear()

    def stop(self) -> None:
        self._stop_requested.set()
        self._pause_requested.clear()

    def should_stop(self) -> bool:
        return self._stop_requested.is_set()

    def is_paused(self) -> bool:
        return self._pause_requested.is_set()
