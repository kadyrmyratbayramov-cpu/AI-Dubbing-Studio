"""Lightweight lip-sync marker engine."""

from __future__ import annotations

from typing import Dict, List


class LipSyncEngine:
    def build_sync_markers(self, transcript_segments: List[Dict[str, object]]) -> List[Dict[str, float]]:
        return [
            {
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
            }
            for seg in transcript_segments
        ]
