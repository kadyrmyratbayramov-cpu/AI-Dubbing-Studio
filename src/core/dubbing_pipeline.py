"""Production dubbing pipeline orchestration."""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.config.settings import Config
from src.core.ffmpeg_engine import FFmpegEngine, FFmpegError
from src.core.gpu_manager import GPUManager
from src.core.job_store import JobStore
from src.core.lipsync_engine import LipSyncEngine
from src.core.mixing_engine import AudioMixingEngine
from src.core.audio_processor import AudioProcessor
from src.core.model_engines import DiarizationEngine, TranslationEngine, TtsEngine, WhisperEngine
from src.core.timing_engine import TimingEngine
from src.core.types import PipelineStatus, Segment, SegmentResult
from src.models.voice_synthesis import VoiceSynthesis
from src.utils.validators import validate_input_file


LOGGER = logging.getLogger(__name__)


class DubbingPipeline:
    """Complete disk-based dubbing pipeline."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config.load()
        self.gpu = GPUManager(force_cpu=self.config.force_cpu)
        self.ffmpeg: Optional[FFmpegEngine] = None
        self.ffmpeg_error: Optional[str] = None
        self.job_store = JobStore(self.config.jobs_dir)
        self.whisper = WhisperEngine(self.config.whisper_model, self.gpu)
        self.diarization = DiarizationEngine(self.config.huggingface_token, self.gpu)
        self.translation = TranslationEngine(self.config.translation_model_map, self.gpu)
        self.tts = TtsEngine(self.config.tts_model, self.gpu, default_voice=self.config.tts_voice)
        self.timing = TimingEngine()
        self.mixer = AudioMixingEngine()
        self.lipsync = LipSyncEngine()
        self.audio_processor = AudioProcessor(self.config)
        self.voice_synthesis = VoiceSynthesis(self.config)

        self._pause_event = threading.Event()
        self._pause_event.set()
        self._stop_requested = False
        self._cancel_requested = False

    def get_ffmpeg(self) -> FFmpegEngine:
        if self.ffmpeg is not None:
            return self.ffmpeg
        try:
            self.ffmpeg = FFmpegEngine()
            self.ffmpeg_error = None
            return self.ffmpeg
        except FFmpegError as exc:
            self.ffmpeg_error = str(exc)
            raise

    def pause(self) -> None:
        self._pause_event.clear()

    def resume(self) -> None:
        self._pause_event.set()

    def stop(self) -> None:
        self._stop_requested = True
        self._pause_event.set()

    def cancel(self) -> None:
        self._cancel_requested = True
        self._pause_event.set()

    def process(
        self,
        input_file: str,
        output_file: Optional[str] = None,
        source_lang: str = "en",
        target_lang: str = "es",
        speaker_reference_wav: Optional[str] = None,
        progress_callback: Optional[Callable[[PipelineStatus], None]] = None,
        log_callback: Optional[Callable[[str, str], None]] = None,
    ) -> Dict[str, Any]:
        validate_input_file(input_file)
        output_path = output_file or str(Path(self.config.output_dir) / (Path(input_file).stem + "_dubbed.mp4"))

        job = self.job_store.new_job()
        start_ts = time.time()
        ffmpeg = self.get_ffmpeg()
        metadata = ffmpeg.probe_video(input_file)

        state: Dict[str, Any] = {
            "job_id": job.job_id,
            "input_file": input_file,
            "output_file": output_path,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "status": "running",
            "stage": "metadata",
            "processed_segment_indices": [],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": metadata.duration,
        }
        self.job_store.save_checkpoint(job.checkpoint_file, state)

        def emit_progress(progress: float, stage: str, message: str = "") -> None:
            elapsed = max(time.time() - start_ts, 0.01)
            eta = None
            if progress > 0:
                eta = elapsed * (100.0 - progress) / progress
            if progress_callback:
                progress_callback(PipelineStatus(stage=stage, progress=progress, eta_seconds=eta, message=message))

        def emit_log(level: str, message: str) -> None:
            LOGGER.log(getattr(logging, level.upper(), logging.INFO), message)
            if log_callback:
                log_callback(level.upper(), message)

        try:
            emit_progress(1.0, "metadata", "Video metadata extracted")
            emit_log("info", f"Audio streams: {len(metadata.audio_streams)} | subtitles: {metadata.subtitle_streams}")

            extracted_wav = str(job.artifacts_dir / "extracted.wav")
            state["stage"] = "audio_extraction"
            self.job_store.save_checkpoint(job.checkpoint_file, state)
            ffmpeg.extract_audio(
                input_file,
                extracted_wav,
                sample_rate=self.config.extraction_sample_rate,
                channels=1,
                duration_hint=metadata.duration,
                progress=lambda p, s: emit_progress(5.0 + p * 0.10, s),
            )

            self._maybe_wait_or_interrupt(job.root)

            state["stage"] = "segmentation"
            self.job_store.save_checkpoint(job.checkpoint_file, state)
            segments = ffmpeg.split_audio_segments(
                extracted_wav,
                str(job.segments_dir),
                self.config.segment_seconds,
                metadata.duration,
            )
            self.job_store.save_manifest(job.manifest_file, segments)
            emit_progress(20.0, "segmentation", f"Created {len(segments)} segments")

            self._maybe_wait_or_interrupt(job.root)

            segment_results: List[SegmentResult] = []
            speakers: Dict[str, str] = {}

            for idx, segment in enumerate(segments):
                self._maybe_wait_or_interrupt(job.root)
                emit_log("info", f"Processing segment {segment.index + 1}/{len(segments)}")

                transcripts = self.whisper.transcribe_segment(segment.path)
                if self.diarization.available:
                    try:
                        transcripts = self.diarization.assign_speakers(segment.path, transcripts)
                    except Exception as exc:
                        emit_log("warning", f"Diarization skipped for segment {segment.id}: {exc}")
                else:
                    for chunk in transcripts:
                        chunk.speaker = "SPEAKER_00"

                original_text = " ".join(chunk.text for chunk in transcripts).strip()
                translated_list = self.translation.translate_texts([original_text], source_lang, target_lang)
                translated_text = translated_list[0] if translated_list else ""

                synthesized_raw = str(job.synthesized_dir / f"{segment.id}_raw.wav")
                self.tts.synthesize_to_file(
                    text=translated_text,
                    output_path=synthesized_raw,
                    language=target_lang,
                    speaker_wav=speaker_reference_wav,
                )

                synthesized_aligned = str(job.synthesized_dir / f"{segment.id}.wav")
                self.timing.stretch_to_duration(synthesized_raw, synthesized_aligned, segment.duration)

                dominant_speaker = transcripts[0].speaker if transcripts else "SPEAKER_00"
                speakers[segment.id] = dominant_speaker

                segment_results.append(
                    SegmentResult(
                        segment=segment,
                        transcript=transcripts,
                        translated_text=translated_text,
                        speaker=dominant_speaker,
                        synthesized_path=synthesized_aligned,
                    )
                )

                state["processed_segment_indices"].append(segment.index)
                state["stage"] = "segment_processing"
                self.job_store.save_checkpoint(job.checkpoint_file, state)

                base = 20.0
                portion = (idx + 1) / max(1, len(segments))
                emit_progress(base + portion * 60.0, "segment_processing")

            self._maybe_wait_or_interrupt(job.root)

            timing_manifest = [record.__dict__ for record in self.timing.build_manifest(segments, speakers)]
            timing_manifest_path = job.artifacts_dir / "timing_manifest.json"
            timing_manifest_path.write_text(json.dumps(timing_manifest, indent=2), encoding="utf-8")

            transcript_payload = []
            for result in segment_results:
                transcript_payload.append(
                    {
                        "segment_id": result.segment.id,
                        "start": result.segment.start,
                        "end": result.segment.end,
                        "speaker": result.speaker,
                        "translated_text": result.translated_text,
                        "chunks": [chunk.__dict__ for chunk in result.transcript],
                    }
                )
            transcript_path = job.artifacts_dir / "transcript.json"
            transcript_path.write_text(json.dumps(transcript_payload, indent=2), encoding="utf-8")

            emit_progress(85.0, "mixing")
            merged_speech = str(job.artifacts_dir / "speech_merged.wav")
            ffmpeg.merge_audio_segments([x.synthesized_path for x in segment_results], merged_speech)

            background_wav = str(job.artifacts_dir / "background.wav")
            ffmpeg.extract_audio(
                input_file,
                background_wav,
                sample_rate=self.config.extraction_sample_rate,
                channels=2,
                duration_hint=metadata.duration,
                progress=None,
            )

            mixed_wav = str(job.artifacts_dir / "dubbed_mix.wav")
            self.mixer.mix_with_ducking([merged_speech], background_wav, mixed_wav)

            emit_progress(95.0, "final_mux")
            ffmpeg.mux_audio_with_video(
                input_file,
                mixed_wav,
                output_path,
                duration_hint=metadata.duration,
                progress=lambda p, s: emit_progress(95.0 + p * 0.05, s),
            )

            state["status"] = "completed"
            state["stage"] = "done"
            state["completed_at"] = datetime.now(timezone.utc).isoformat()
            self.job_store.save_checkpoint(job.checkpoint_file, state)

            history_entry = {
                "job_id": job.job_id,
                "input_file": input_file,
                "output_file": output_path,
                "created_at": state["created_at"],
                "completed_at": state["completed_at"],
                "source_lang": source_lang,
                "target_lang": target_lang,
                "status": "completed",
                "duration_seconds": metadata.duration,
            }
            self.job_store.append_history(history_entry)

            emit_progress(100.0, "done", "Dubbing finished")

            return {
                "status": "success",
                "job_id": job.job_id,
                "input": input_file,
                "output": output_path,
                "metadata": metadata.__dict__,
                "transcript_json": str(transcript_path),
                "timing_manifest": str(timing_manifest_path),
                "lip_sync": self.lipsync.get_status().__dict__,
            }

        except Exception as exc:
            state["status"] = "cancelled" if self._cancel_requested else "failed"
            state["stage"] = "error"
            state["error"] = str(exc)
            state["completed_at"] = datetime.now(timezone.utc).isoformat()
            self.job_store.save_checkpoint(job.checkpoint_file, state)

            self.job_store.append_history(
                {
                    "job_id": job.job_id,
                    "input_file": input_file,
                    "output_file": output_path,
                    "created_at": state["created_at"],
                    "completed_at": state["completed_at"],
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                    "status": state["status"],
                    "error": str(exc),
                }
            )
            if self._cancel_requested:
                self.job_store.cleanup(job.root)
            emit_log("error", f"Pipeline failed: {exc}")
            raise
        finally:
            self.whisper.unload()
            self.diarization.unload()
            self.translation.unload()
            self.tts.unload()
            self.gpu.clear_cuda_cache()
            self._stop_requested = False
            self._cancel_requested = False
            self._pause_event.set()

    def _maybe_wait_or_interrupt(self, job_root: Path) -> None:
        while not self._pause_event.is_set():
            time.sleep(0.25)
        if self._cancel_requested:
            self.job_store.cleanup(job_root)
            raise RuntimeError("Job cancelled by user")
        if self._stop_requested:
            raise RuntimeError("Job stopped by user")

    def get_job_history(self) -> List[Dict[str, Any]]:
        return self.job_store.load_history()

    def get_gpu_status(self) -> Dict[str, Any]:
        return self.gpu.as_dict()
