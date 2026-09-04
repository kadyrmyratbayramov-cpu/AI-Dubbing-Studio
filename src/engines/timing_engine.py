"""Timing and alignment engine for synthesized audio."""

from __future__ import annotations

from pathlib import Path


class TimingEngine:
    def stretch_to_duration(self, input_audio: str, target_duration_sec: float, output_audio: str) -> str:
        import librosa
        import soundfile as sf

        audio, sr = librosa.load(input_audio, sr=None)
        current_duration = len(audio) / sr if sr else 0.0
        if current_duration <= 0:
            raise ValueError("Input audio has invalid duration")

        rate = current_duration / max(target_duration_sec, 1e-6)
        stretched = librosa.effects.time_stretch(audio, rate=rate)

        out = Path(output_audio)
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out, stretched, sr)
        return str(out)
