"""Coqui XTTS-v2 text-to-speech engine."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


class XTTSv2Engine:
    def __init__(self, model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2", gpu: bool = False):
        self.model_name = model_name
        self.gpu = gpu
        self._tts = None

    def load(self) -> None:
        if self._tts is not None:
            return
        from TTS.api import TTS

        self._tts = TTS(self.model_name)
        if self.gpu:
            self._tts.to("cuda")

    def synthesize_to_file(
        self,
        text: str,
        output_path: str,
        language: str,
        speaker_wav: Optional[str] = None,
        speaker: Optional[str] = None,
    ) -> str:
        self.load()
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        kwargs = {"text": text, "file_path": str(out), "language": language}
        if speaker_wav:
            kwargs["speaker_wav"] = speaker_wav
        elif speaker:
            kwargs["speaker"] = speaker

        self._tts.tts_to_file(**kwargs)
        return str(out)
