"""Job metadata and lifecycle management."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any
from uuid import uuid4


@dataclass
class Job:
    id: str
    input_video: str
    source_language: str
    target_language: str
    status: str = "created"
    stage: str = "init"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


class JobManager:
    def __init__(self, jobs_dir: str):
        self.jobs_dir = Path(jobs_dir)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

    def create(self, input_video: str, source_language: str, target_language: str) -> Job:
        job = Job(id=uuid4().hex, input_video=input_video, source_language=source_language, target_language=target_language)
        job_dir = self.jobs_dir / job.id
        job_dir.mkdir(parents=True, exist_ok=True)
        return job
