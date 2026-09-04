"""Audio mixing engine for dubbing output."""

from __future__ import annotations

from pathlib import Path


class MixingEngine:
    def mix_voice_with_background(
        self,
        voice_audio: str,
        background_audio: str,
        output_audio: str,
        ducking_db: float = 10.0,
    ) -> str:
        from pydub import AudioSegment
        from pydub.effects import normalize

        voice = AudioSegment.from_file(voice_audio)
        background = AudioSegment.from_file(background_audio) - ducking_db
        mixed = normalize(background.overlay(voice))
        out = Path(output_audio)
        out.parent.mkdir(parents=True, exist_ok=True)
        mixed.export(out, format="wav")
        return str(out)
