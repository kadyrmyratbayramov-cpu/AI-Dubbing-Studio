"""Model-backed STT/diarization/translation/TTS engines."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.core.gpu_manager import GPUManager
from src.core.types import TranscriptChunk


LOGGER = logging.getLogger(__name__)


class WhisperEngine:
    def __init__(self, model_name: str, gpu: GPUManager) -> None:
        self.model_name = model_name
        self.gpu = gpu
        self.model = None

    def load(self) -> None:
        if self.model is not None:
            return
        try:
            import whisper
        except ImportError as exc:
            raise RuntimeError("openai-whisper is not installed") from exc

        device = self.gpu.preferred_device()
        self.model = whisper.load_model(self.model_name, device=device)
        LOGGER.info("Whisper model loaded: %s on %s", self.model_name, device)

    def unload(self) -> None:
        if self.model is not None:
            self.gpu.unload_model(self.model)
            self.model = None

    def transcribe_segment(self, wav_path: str) -> List[TranscriptChunk]:
        self.load()
        assert self.model is not None

        try:
            result = self.model.transcribe(wav_path, word_timestamps=True)
        except RuntimeError as exc:
            if "CUDA" in str(exc).upper():
                LOGGER.warning("Whisper CUDA failure, retrying on CPU")
                self.unload()
                self.gpu.force_cpu = True
                self.load()
                result = self.model.transcribe(wav_path, word_timestamps=True)
            else:
                raise

        chunks: List[TranscriptChunk] = []
        for seg in result.get("segments", []):
            words = []
            for word in seg.get("words", []) or []:
                words.append(
                    {
                        "word": word.get("word", "").strip(),
                        "start": float(word.get("start", seg.get("start", 0.0))),
                        "end": float(word.get("end", seg.get("end", 0.0))),
                        "confidence": float(word.get("probability", 0.0)),
                    }
                )
            conf_values = [float(w.get("confidence", 0.0)) for w in words]
            confidence = sum(conf_values) / len(conf_values) if conf_values else 0.0
            chunks.append(
                TranscriptChunk(
                    start=float(seg.get("start", 0.0)),
                    end=float(seg.get("end", 0.0)),
                    text=str(seg.get("text", "")).strip(),
                    confidence=confidence,
                    words=words,
                )
            )
        return chunks


class DiarizationEngine:
    def __init__(self, token: str, gpu: GPUManager) -> None:
        self.token = token
        self.gpu = gpu
        self.pipeline = None
        self.available = bool(token)

    def load(self) -> None:
        if not self.available or self.pipeline is not None:
            return
        try:
            from pyannote.audio import Pipeline
        except ImportError as exc:
            raise RuntimeError("pyannote.audio is not installed") from exc

        self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=self.token)
        if self.gpu.cuda_available():
            import torch

            self.pipeline.to(torch.device("cuda"))

    def unload(self) -> None:
        if self.pipeline is not None:
            self.gpu.unload_model(self.pipeline)
            self.pipeline = None

    def assign_speakers(self, wav_path: str, chunks: List[TranscriptChunk]) -> List[TranscriptChunk]:
        if not self.available:
            return chunks

        self.load()
        assert self.pipeline is not None

        try:
            diarization = self.pipeline(wav_path)
        except RuntimeError as exc:
            if "CUDA" in str(exc).upper():
                LOGGER.warning("Diarization CUDA failure, retrying on CPU")
                self.unload()
                self.gpu.force_cpu = True
                self.load()
                diarization = self.pipeline(wav_path)
            else:
                raise

        turns: List[tuple[float, float, str]] = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            turns.append((float(turn.start), float(turn.end), str(speaker)))

        if not turns:
            return chunks

        for chunk in chunks:
            matched = [t for t in turns if not (t[1] < chunk.start or t[0] > chunk.end)]
            if not matched:
                chunk.speaker = "SPEAKER_00"
                continue
            speaker = max(matched, key=lambda t: min(chunk.end, t[1]) - max(chunk.start, t[0]))[2]
            chunk.speaker = speaker
        return chunks


class TranslationEngine:
    def __init__(self, model_map: Dict[str, str], gpu: GPUManager) -> None:
        self.model_map = model_map
        self.gpu = gpu
        self.loaded_pair: Optional[str] = None
        self.tokenizer = None
        self.model = None

    def _load_pair(self, source_lang: str, target_lang: str) -> str:
        pair = f"{source_lang}-{target_lang}"
        model_name = self.model_map.get(pair)
        if not model_name:
            raise RuntimeError(f"Translation pair not supported: {pair}")

        if self.loaded_pair == pair and self.tokenizer is not None and self.model is not None:
            return pair

        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("transformers is not installed") from exc

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        if self.gpu.cuda_available():
            self.model.to("cuda")
        self.loaded_pair = pair
        return pair

    def unload(self) -> None:
        if self.model is not None:
            self.gpu.unload_model(self.model)
            self.model = None
            self.tokenizer = None
            self.loaded_pair = None

    def translate_texts(self, texts: List[str], source_lang: str, target_lang: str) -> List[str]:
        if not texts:
            return []

        pair = self._load_pair(source_lang, target_lang)
        assert self.model is not None and self.tokenizer is not None
        try:
            import torch

            target_device = "cuda" if self.gpu.cuda_available() else "cpu"
            model_device = next(self.model.parameters()).device.type
            if model_device != target_device:
                self.model.to(target_device)

            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                output_tokens = self.model.generate(**inputs, max_new_tokens=512)
            return self.tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
        except RuntimeError as exc:
            if "CUDA" in str(exc).upper():
                LOGGER.warning("Translation CUDA failure for %s, retrying on CPU", pair)
                self.unload()
                self.gpu.force_cpu = True
                return self.translate_texts(texts, source_lang, target_lang)
            raise


class TtsEngine:
    def __init__(self, model_name: str, gpu: GPUManager, default_voice: str = "neutral") -> None:
        self.model_name = model_name
        self.gpu = gpu
        self.default_voice = default_voice
        self.tts = None

    def load(self) -> None:
        if self.tts is not None:
            return
        try:
            from TTS.api import TTS
        except ImportError as exc:
            raise RuntimeError("Coqui TTS package is not installed") from exc

        use_gpu = self.gpu.cuda_available()
        self.tts = TTS(self.model_name, gpu=use_gpu)

    def unload(self) -> None:
        if self.tts is not None:
            self.gpu.unload_model(self.tts)
            self.tts = None

    def synthesize_to_file(
        self,
        text: str,
        output_path: str,
        language: str,
        speaker_wav: Optional[str] = None,
    ) -> str:
        self.load()
        assert self.tts is not None

        try:
            self.tts.tts_to_file(
                text=text,
                file_path=output_path,
                language=language,
                speaker_wav=speaker_wav,
            )
            return output_path
        except RuntimeError as exc:
            if "CUDA" in str(exc).upper():
                LOGGER.warning("TTS CUDA failure, retrying on CPU")
                self.unload()
                self.gpu.force_cpu = True
                self.load()
                self.tts.tts_to_file(
                    text=text,
                    file_path=output_path,
                    language=language,
                    speaker_wav=speaker_wav,
                )
                return output_path
            raise
