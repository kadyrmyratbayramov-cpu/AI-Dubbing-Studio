"""End-to-end dubbing orchestrator with retry/recovery and checkpoints."""

from __future__ import annotations

import json
import time
from pathlib import Path
from threading import Event
from typing import Any, Callable, Dict, Optional

from src.engines import (
    LipSyncEngine,
    MarianTranslationEngine,
    MixingEngine,
    PyannoteDiarizationEngine,
    QualityControlEngine,
    TimingEngine,
    WhisperSTTEngine,
    XTTSv2Engine,
)
from src.media.ffmpeg_wrapper import FFmpegWrapper
from src.media.segment_processor import SegmentProcessor
from src.pipeline.job_manager import JobManager
from src.pipeline.state_manager import StateManager
from src.utils.gpu_manager import GPUManager
from src.utils.logger import setup_logger

ProgressCallback = Callable[[str, float, str], None]


class JobCancelledError(RuntimeError):
    """Raised when a running job is cancelled by the user."""


class DubbingOrchestrator:
    def __init__(self, config):
        self.config = config
        self.ffmpeg = FFmpegWrapper(config.ffmpeg_bin, config.ffprobe_bin)
        self.segment_processor = SegmentProcessor(config.segment_seconds)
        self.job_manager = JobManager(str(config.jobs_dir))
        self.state_manager = StateManager(str(config.jobs_dir))
        self.gpu = GPUManager(config.device, config.max_vram_mb)
        self.logger = setup_logger("orchestrator", config.log_dir, config.log_level)

        self.pause_event = Event()
        self.cancel_event = Event()

    def pause(self) -> None:
        self.pause_event.set()

    def resume(self) -> None:
        self.pause_event.clear()

    def cancel(self) -> None:
        self.cancel_event.set()

    def _wait_if_paused(self) -> None:
        while self.pause_event.is_set() and not self.cancel_event.is_set():
            time.sleep(0.2)

    def _stage(self, name: str, pct: float, callback: Optional[ProgressCallback], message: str) -> None:
        self.logger.info("[%s] %s", name, message)
        if callback:
            callback(name, pct, message)

    def _run_with_retry(self, func: Callable[[], Any], stage: str, retries: int) -> Any:
        delay = 1.0
        last_error = None
        for attempt in range(retries + 1):
            try:
                return func()
            except Exception as exc:
                last_error = exc
                self.logger.exception("Stage %s failed on attempt %s", stage, attempt + 1)
                if attempt < retries:
                    time.sleep(delay)
                    delay *= 2
        raise RuntimeError(f"Stage {stage} failed after retries") from last_error

    def process(
        self,
        input_video: str,
        source_language: str,
        target_language: str,
        progress: Optional[ProgressCallback] = None,
        resume_job_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        self.pause_event.clear()
        self.cancel_event.clear()

        if resume_job_id:
            saved = self.state_manager.load(resume_job_id)
            if not saved:
                raise FileNotFoundError(f"Checkpoint not found for job_id={resume_job_id}")
            job_id = resume_job_id
            input_video = saved.get("input_video", input_video)
            source_language = saved.get("source_language", source_language)
            target_language = saved.get("target_language", target_language)
            stage = saved.get("stage", "init")
        else:
            saved = None
            stage = "init"
            job = self.job_manager.create(input_video, source_language, target_language)
            job_id = job.id
        job_dir = Path(self.config.jobs_dir) / job_id
        audio_path = job_dir / "input_audio.wav"
        segments_dir = job_dir / "segments"
        segment_manifest = job_dir / "segment_manifest.json"
        tts_dir = job_dir / "tts"
        output_audio = job_dir / "dubbed.wav"
        mixed_audio = job_dir / "mixed.wav"
        output_video = Path(self.config.output_dir) / f"{Path(input_video).stem}_{source_language}_{target_language}.mp4"
        job_dir.mkdir(parents=True, exist_ok=True)
        segments_dir.mkdir(parents=True, exist_ok=True)
        tts_dir.mkdir(parents=True, exist_ok=True)
        output_video.parent.mkdir(parents=True, exist_ok=True)

        state = saved or {
            "job_id": job_id,
            "input_video": input_video,
            "source_language": source_language,
            "target_language": target_language,
            "status": "running",
            "stage": stage,
            "gpu": self.gpu.snapshot(),
        }
        self.state_manager.save(job_id, state)

        def check_cancel() -> None:
            if self.cancel_event.is_set():
                state["status"] = "cancelled"
                self.state_manager.save(job_id, state)
                raise JobCancelledError("Job cancelled")

        try:
            if stage in {"init", "failed", "cancelled"}:
                self._stage("validation", 0.05, progress, "Validating input and reading metadata")
                metadata = self._run_with_retry(lambda: self.ffmpeg.probe(input_video), "probe", self.config.retry_attempts)
                state.update({"stage": "metadata", "metadata": metadata})
                self.state_manager.save(job_id, state)
            else:
                metadata = state.get("metadata", {})
            check_cancel()
            self._wait_if_paused()

            if not audio_path.exists():
                self._stage("audio_extraction", 0.12, progress, "Extracting normalized audio")
                self._run_with_retry(
                    lambda: self.ffmpeg.extract_audio(input_video, str(audio_path), self.config.stt_sample_rate, 1),
                    "audio_extraction",
                    self.config.retry_attempts,
                )
                state["stage"] = "audio_extracted"
                self.state_manager.save(job_id, state)
            check_cancel()
            self._wait_if_paused()

            if segments_dir.exists() and list(segments_dir.glob("segment_*.wav")):
                from src.media.segment_processor import AudioSegmentRef

                segments = []
                if segment_manifest.exists():
                    manifest = json.loads(segment_manifest.read_text(encoding="utf-8"))
                    for entry in manifest:
                        segments.append(
                            AudioSegmentRef(
                                index=int(entry["index"]),
                                path=str(entry["path"]),
                                start_sec=float(entry["start_sec"]),
                                end_sec=float(entry["end_sec"]),
                            )
                        )
                else:
                    import soundfile as sf

                    cursor = 0.0
                    for idx, path in enumerate(sorted(segments_dir.glob("segment_*.wav"))):
                        info = sf.info(str(path))
                        duration = float(info.frames) / float(info.samplerate) if info.samplerate else 0.0
                        segments.append(AudioSegmentRef(idx, str(path), cursor, cursor + duration))
                        cursor += duration
            else:
                self._stage("segmentation", 0.2, progress, "Splitting audio into segments")
                segments = self.segment_processor.create_segments(str(audio_path), str(segments_dir))
                manifest = [
                    {
                        "index": segment.index,
                        "path": segment.path,
                        "start_sec": segment.start_sec,
                        "end_sec": segment.end_sec,
                    }
                    for segment in segments
                ]
                segment_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            state["stage"] = "segmented"
            state["segment_count"] = len(segments)
            self.state_manager.save(job_id, state)
            check_cancel()

            device = self.gpu.select_device()
            stt = WhisperSTTEngine(self.config.whisper_model, device=device)
            diarization = PyannoteDiarizationEngine(self.config.pyannote_model, self.config.hf_token)
            translation = MarianTranslationEngine()
            tts = XTTSv2Engine(self.config.xtts_model, gpu=device == "cuda")
            timing = TimingEngine()
            mixing = MixingEngine()
            lipsync = LipSyncEngine()
            qc = QualityControlEngine()

            stt_segments = []
            translated_segments = []
            synthesized_segments = []

            for idx, segment in enumerate(segments):
                self._wait_if_paused()
                check_cancel()
                base_progress = 0.25 + (idx / max(len(segments), 1)) * 0.55

                self._stage("transcription", base_progress, progress, f"Transcribing segment {idx + 1}/{len(segments)}")
                stt_result = self._run_with_retry(
                    lambda seg_path=segment.path: stt.transcribe(seg_path, language=source_language),
                    "stt",
                    self.config.retry_attempts,
                )
                text = stt_result.get("text", "").strip()
                for stt_segment in stt_result.get("segments", []):
                    stt_segments.append(
                        {
                            **stt_segment,
                            "start": float(stt_segment.get("start", 0.0)) + segment.start_sec,
                            "end": float(stt_segment.get("end", 0.0)) + segment.start_sec,
                        }
                    )

                self._stage("diarization", base_progress + 0.04, progress, f"Diarizing segment {idx + 1}/{len(segments)}")
                diarization_segments = self._run_with_retry(
                    lambda seg_path=segment.path: diarization.diarize(seg_path),
                    "diarization",
                    self.config.retry_attempts,
                )

                self._stage("translation", base_progress + 0.08, progress, f"Translating segment {idx + 1}/{len(segments)}")
                if not text:
                    translated = ""
                else:
                    translated_batch = self._run_with_retry(
                        lambda txt=text: translation.translate([txt], source_language, target_language),
                        "translation",
                        self.config.retry_attempts,
                    )
                    translated = translated_batch[0] if translated_batch else ""
                translated_segments.append(
                    {
                        "index": idx,
                        "source_text": text,
                        "translated_text": translated,
                        "start": segment.start_sec,
                        "end": segment.end_sec,
                        "speakers": diarization_segments,
                    }
                )

                self._stage("synthesis", base_progress + 0.12, progress, f"Synthesizing segment {idx + 1}/{len(segments)}")
                tts_output = tts_dir / f"tts_{idx:04d}.wav"
                self._run_with_retry(
                    lambda out=str(tts_output), txt=translated: tts.synthesize_to_file(
                        text=txt,
                        output_path=out,
                        language=target_language,
                        speaker=self.config.default_speaker,
                    ),
                    "tts",
                    self.config.retry_attempts,
                )

                target_duration = max(segment.end_sec - segment.start_sec, 0.1)
                timed_output = tts_dir / f"tts_timed_{idx:04d}.wav"
                self._run_with_retry(
                    lambda src=str(tts_output), dur=target_duration, out=str(timed_output): timing.stretch_to_duration(
                        src, dur, out
                    ),
                    "timing",
                    self.config.retry_attempts,
                )
                synthesized_segments.append((str(timed_output), segment.start_sec))

            from pydub import AudioSegment
            import soundfile as sf

            self._stage("audio_merge", 0.84, progress, "Merging synthesized segments")
            total_duration_ms = int(float(metadata.get("duration", 0.0)) * 1000)
            source_audio_info = sf.info(str(audio_path))
            mix_sample_rate = int(source_audio_info.samplerate or self.config.stt_sample_rate)
            combined = AudioSegment.silent(
                duration=max(total_duration_ms, 1),
                frame_rate=mix_sample_rate,
            ).set_channels(1)
            for file_path, start_sec in synthesized_segments:
                combined = combined.overlay(AudioSegment.from_file(file_path), position=int(start_sec * 1000))
            combined.export(output_audio, format="wav")

            self._stage("audio_mixing", 0.9, progress, "Mixing dubbed voice with original background")
            self._run_with_retry(
                lambda: mixing.mix_voice_with_background(str(output_audio), str(audio_path), str(mixed_audio)),
                "mixing",
                self.config.retry_attempts,
            )

            self._stage("lip_sync", 0.94, progress, "Generating lip-sync markers")
            lip_markers = lipsync.build_sync_markers(stt_segments)

            self._stage("quality_control", 0.97, progress, "Running quality checks")
            qc_result = qc.run(str(mixed_audio), expected_duration=float(metadata.get("duration", 0.0)))

            self._stage("video_output", 0.99, progress, "Muxing final dubbed video")
            self._run_with_retry(
                lambda: self.ffmpeg.mux_video_with_audio(input_video, str(mixed_audio), str(output_video)),
                "video_output",
                self.config.retry_attempts,
            )

            state.update(
                {
                    "status": "completed",
                    "stage": "completed",
                    "output_video": str(output_video),
                    "translated_segments": translated_segments,
                    "qc": qc_result,
                    "lip_sync_markers": lip_markers,
                }
            )
            self.state_manager.save(job_id, state)
            self._stage("completed", 1.0, progress, "Job completed")
            return state
        except Exception as exc:
            if isinstance(exc, JobCancelledError):
                state["status"] = "cancelled"
                state["stage"] = "cancelled"
            else:
                state["status"] = "failed"
                state["stage"] = "failed"
            state["error"] = str(exc)
            self.state_manager.save(job_id, state)
            raise
        finally:
            if self.config.cleanup_intermediates and job_dir.exists() and state.get("status") == "completed":
                keep_files = {"state.json", "segment_manifest.json"}
                for path in job_dir.rglob("*"):
                    if path.is_file() and path.name not in keep_files:
                        path.unlink(missing_ok=True)
                for path in sorted(job_dir.rglob("*"), reverse=True):
                    if path.is_dir() and not any(path.iterdir()):
                        path.rmdir()
            self.gpu.cleanup()
