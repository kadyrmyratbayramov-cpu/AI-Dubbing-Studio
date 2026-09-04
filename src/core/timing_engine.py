"""Timing synchronization for synthesized segments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import librosa
import numpy as np
import soundfile as sf

from src.core.types import Segment


@dataclass
class TimingRecord:
    segment_id: str
    start: float
    end: float
    duration: float
    speaker: str


class TimingEngine:
    def estimate_wpm(self, text: str, duration: float) -> float:
        words = len([word for word in text.split() if word.strip()])
        if duration <= 0:
            return 0.0
        return words / duration * 60.0

    def stretch_to_duration(self, input_wav: str, output_wav: str, target_duration: float) -> None:
        audio, sr = sf.read(input_wav)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        current_duration = len(audio) / sr if sr else 0.0
        if current_duration <= 0 or target_duration <= 0:
            sf.write(output_wav, audio, sr)
            return

        rate = current_duration / target_duration
        rate = min(max(rate, 0.5), 2.0)
        stretched = librosa.effects.time_stretch(audio.astype(np.float32), rate=rate)

        expected_samples = max(1, int(target_duration * sr))
        if len(stretched) < expected_samples:
            stretched = np.pad(stretched, (0, expected_samples - len(stretched)))
        elif len(stretched) > expected_samples:
            stretched = stretched[:expected_samples]

        sf.write(output_wav, stretched, sr)

    def build_manifest(
        self,
        segments: List[Segment],
        speakers: Dict[str, str],
    ) -> List[TimingRecord]:
        records: List[TimingRecord] = []
        for segment in segments:
            records.append(
                TimingRecord(
                    segment_id=segment.id,
                    start=segment.start,
                    end=segment.end,
                    duration=segment.duration,
                    speaker=speakers.get(segment.id, "SPEAKER_00"),
                )
            )
        return records
