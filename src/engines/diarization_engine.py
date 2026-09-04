"""pyannote-based speaker diarization engine."""

from __future__ import annotations

from typing import Dict, List, Optional


class PyannoteDiarizationEngine:
    def __init__(self, model_id: str = "pyannote/speaker-diarization", auth_token: Optional[str] = None):
        self.model_id = model_id
        self.auth_token = auth_token
        self._pipeline = None

    def load(self) -> None:
        if self._pipeline is not None:
            return
        from pyannote.audio import Pipeline

        self._pipeline = Pipeline.from_pretrained(self.model_id, use_auth_token=self.auth_token)

    def diarize(self, audio_path: str) -> List[Dict[str, object]]:
        self.load()
        diarization = self._pipeline(audio_path)
        timeline: List[Dict[str, object]] = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            timeline.append({
                "speaker": speaker,
                "start": float(turn.start),
                "end": float(turn.end),
            })
        return timeline
