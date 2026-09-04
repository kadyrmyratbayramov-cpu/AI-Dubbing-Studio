"""Disk-backed job/checkpoint persistence."""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from src.core.types import JobPaths, Segment


class JobStore:
    def __init__(self, jobs_dir: str) -> None:
        self.jobs_dir = Path(jobs_dir)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = self.jobs_dir / "history.json"

    def new_job(self) -> JobPaths:
        job_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        root = self.jobs_dir / job_id
        paths = JobPaths(
            job_id=job_id,
            root=root,
            segments_dir=root / "segments",
            synthesized_dir=root / "synthesized",
            artifacts_dir=root / "artifacts",
            checkpoint_file=root / "checkpoint.json",
            manifest_file=root / "segment_manifest.json",
        )
        for path in (paths.root, paths.segments_dir, paths.synthesized_dir, paths.artifacts_dir):
            path.mkdir(parents=True, exist_ok=True)
        return paths

    def save_manifest(self, manifest_path: Path, segments: List[Segment]) -> None:
        payload = [
            {
                "id": s.id,
                "index": s.index,
                "start": s.start,
                "end": s.end,
                "duration": s.duration,
                "path": s.path,
            }
            for s in segments
        ]
        manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def save_checkpoint(self, checkpoint_path: Path, state: Dict[str, Any]) -> None:
        checkpoint_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def load_checkpoint(self, checkpoint_path: Path) -> Dict[str, Any]:
        if not checkpoint_path.exists():
            return {}
        return json.loads(checkpoint_path.read_text(encoding="utf-8"))

    def append_history(self, entry: Dict[str, Any]) -> None:
        history = self.load_history()
        history.insert(0, entry)
        self.history_path.write_text(json.dumps(history[:200], indent=2), encoding="utf-8")

    def load_history(self) -> List[Dict[str, Any]]:
        if not self.history_path.exists():
            return []
        return json.loads(self.history_path.read_text(encoding="utf-8"))

    def cleanup(self, job_root: Path) -> None:
        if job_root.exists():
            shutil.rmtree(job_root, ignore_errors=True)
