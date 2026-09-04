"""Checkpoint persistence for job state."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional


class StateManager:
    def __init__(self, jobs_dir: str):
        self.jobs_dir = Path(jobs_dir)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

    def state_path(self, job_id: str) -> Path:
        return self.jobs_dir / job_id / "state.json"

    def save(self, job_id: str, state: Dict[str, Any]) -> None:
        path = self.state_path(job_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {**state, "updated_at": datetime.now(timezone.utc).isoformat()}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load(self, job_id: str) -> Optional[Dict[str, Any]]:
        path = self.state_path(job_id)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
