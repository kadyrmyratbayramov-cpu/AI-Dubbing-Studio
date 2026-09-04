"""Audio mixing utilities for dubbed output."""

from __future__ import annotations

from typing import Any, List

import numpy as np


class AudioMixingEngine:
    def mix_with_ducking(
        self,
        dubbed_segments: List[Any],
        background_wav: str,
        output_wav: str,
        ducking_db: float = -10.0,
    ) -> None:
        import soundfile as sf

        bg_audio, sr = sf.read(background_wav)
        if bg_audio.ndim == 1:
            bg_audio = np.stack([bg_audio, bg_audio], axis=1)

        mixed = bg_audio.astype(np.float32).copy()
        speech_track = np.zeros_like(mixed)

        cursor = 0
        for item in dubbed_segments:
            if isinstance(item, dict):
                segment_path = str(item.get("path"))
                start_seconds = float(item.get("start", 0.0))
                start_idx = int(start_seconds * sr)
            else:
                segment_path = str(item)
                start_idx = cursor

            seg_audio, seg_sr = sf.read(segment_path)
            if seg_sr != sr:
                raise RuntimeError(f"Sample rate mismatch between background ({sr}) and segment ({seg_sr})")

            if seg_audio.ndim == 1:
                seg_audio = np.stack([seg_audio, seg_audio], axis=1)

            end = min(start_idx + seg_audio.shape[0], speech_track.shape[0])
            if end <= start_idx:
                continue
            speech_track[start_idx:end] += seg_audio[: end - start_idx]

            energy = np.mean(np.abs(seg_audio[: end - start_idx]), axis=1)
            voice_active = energy > 0.01
            if np.any(voice_active):
                reduction = 10 ** (ducking_db / 20.0)
                for i, active in enumerate(voice_active):
                    if active and start_idx + i < mixed.shape[0]:
                        mixed[start_idx + i] *= reduction

            cursor = end

        out = mixed + speech_track
        peak = np.max(np.abs(out)) if out.size else 0.0
        if peak > 0.99:
            out = out / peak * 0.99

        sf.write(output_wav, out, sr)
