"""Whisper-based speech-to-text engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class STTSegment:
    start: float
    end: float
    text: str
    confidence: float


class WhisperSTTEngine:
    def __init__(self, model_name: str = "small", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None

    def load(self) -> None:
        if self._model is not None:
            return
        import whisper

        self._model = whisper.load_model(self.model_name, device=self.device)

    def transcribe(self, audio_path: str, language: Optional[str] = None) -> Dict[str, Any]:
        self.load()
        result = self._model.transcribe(audio_path, language=language, fp16=self.device == "cuda")
        segments: List[STTSegment] = []
        for seg in result.get("segments", []):
            confidence = float(seg.get("avg_logprob", -1.0))
            segments.append(
                STTSegment(
                    start=float(seg.get("start", 0.0)),
                    end=float(seg.get("end", 0.0)),
                    text=seg.get("text", "").strip(),
                    confidence=confidence,
                )
            )
        return {
            "language": result.get("language", language),
            "text": result.get("text", "").strip(),
            "segments": [segment.__dict__ for segment in segments],
        }
