"""Segment-based audio processing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import soundfile as sf
import numpy as np


@dataclass
class AudioSegmentRef:
    index: int
    path: str
    start_sec: float
    end_sec: float


class SegmentProcessor:
    def __init__(self, segment_seconds: int = 45):
        self.segment_seconds = segment_seconds

    def create_segments(self, audio_path: str, output_dir: str) -> List[AudioSegmentRef]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        info = sf.info(audio_path)
        sr = info.samplerate
        channels = info.channels
        samples_per_segment = int(self.segment_seconds * sr)
        segments: List[AudioSegmentRef] = []
        idx = 0
        start_sample = 0
        buffer: np.ndarray = np.empty((0, channels)) if channels > 1 else np.empty((0,))

        for block in sf.blocks(audio_path, blocksize=max(samples_per_segment // 4, 1024), always_2d=channels > 1):
            if buffer.size == 0:
                buffer = block
            else:
                buffer = np.concatenate([buffer, block], axis=0)

            while len(buffer) >= samples_per_segment:
                chunk = buffer[:samples_per_segment]
                buffer = buffer[samples_per_segment:]
                segment_path = output / f"segment_{idx:04d}.wav"
                sf.write(segment_path, chunk, sr)
                start_sec = start_sample / sr
                end_sec = (start_sample + len(chunk)) / sr
                segments.append(AudioSegmentRef(idx, str(segment_path), start_sec, end_sec))
                idx += 1
                start_sample += len(chunk)

        if len(buffer):  # tail segment
            segment_path = output / f"segment_{idx:04d}.wav"
            sf.write(segment_path, buffer, sr)
            start_sec = start_sample / sr
            end_sec = (start_sample + len(buffer)) / sr
            segments.append(AudioSegmentRef(idx, str(segment_path), start_sec, end_sec))
        return segments

    def iter_segments(self, audio_path: str, output_dir: str) -> Iterable[AudioSegmentRef]:
        for segment in self.create_segments(audio_path, output_dir):
            yield segment
