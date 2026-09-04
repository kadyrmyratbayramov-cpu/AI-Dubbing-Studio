"""Quality control checks for generated dubbing audio."""

from __future__ import annotations

from typing import Dict, List


class QualityControlEngine:
    def run(self, audio_path: str, expected_duration: float) -> Dict[str, object]:
        import librosa
        import numpy as np

        audio, sr = librosa.load(audio_path, sr=None)
        duration = len(audio) / sr if sr else 0.0
        peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
        silence_ratio = float(np.mean(np.abs(audio) < 0.002)) if len(audio) else 1.0

        issues: List[str] = []
        if peak >= 0.999:
            issues.append("possible_clipping")
        if abs(duration - expected_duration) > 1.5:
            issues.append("duration_mismatch")
        if silence_ratio > 0.95:
            issues.append("excessive_silence")

        return {
            "duration": duration,
            "expected_duration": expected_duration,
            "peak": peak,
            "silence_ratio": silence_ratio,
            "issues": issues,
            "passed": len(issues) == 0,
        }
