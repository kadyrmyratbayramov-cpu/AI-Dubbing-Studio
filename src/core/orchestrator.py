"""Processing orchestrator for the desktop pipeline."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

from src.config.settings import Config
from src.core.job_state import JobController, JobStatus, PipelineEvent, PipelineJobState, PipelineRequest, PipelineStage
from src.core.media_pipeline import FFmpegMediaPipeline
from src.models.diarization import DiarizationEngine
from src.models.quality_control import QualityControlEngine
from src.models.speech_to_text import SpeechToTextEngine
from src.models.text_to_speech import TextToSpeechEngine
from src.models.timing_engine import TimingEngine
from src.models.translation import TranslationEngine

StatusCallback = Optional[Callable[[PipelineEvent], None]]


class DubbingOrchestrator:
    def __init__(self, config: Config):
        self.config = config
        self.media_pipeline = FFmpegMediaPipeline(config)
        self.speech_to_text = SpeechToTextEngine(config)
        self.diarization = DiarizationEngine(config)
        self.translation = TranslationEngine(config)
        self.text_to_speech = TextToSpeechEngine(config)
        self.timing_engine = TimingEngine(config)
        self.quality_control = QualityControlEngine(config)

    def run(self, request: PipelineRequest, callback: StatusCallback = None, controller: Optional[JobController] = None) -> Dict[str, object]:
        controller = controller or JobController()
        job = self._load_or_create_job(request)
        job.status = JobStatus.RUNNING.value
        self._emit(callback, PipelineStage.PROBE, job.status, "Pipeline started", 0.0, {"resources": self._collect_resources()})

        metadata = self._run_stage(job, PipelineStage.PROBE, callback, controller, self.media_pipeline.probe, request.input_file)
        job.metadata = metadata.to_dict()
        self._save_checkpoint(job)

        qc_report = self._run_stage(job, PipelineStage.QUALITY_CONTROL, callback, controller, self.quality_control.inspect_metadata, metadata)
        self._save_checkpoint(job)

        transcript_preview: List[Dict[str, object]] = []
        segment_limit = max(1, int(self.config.max_transcription_segments))
        for segment in self.media_pipeline.iter_audio_segments(request.input_file, self.config.workspace_dir):
            self._wait_if_paused(job, callback, controller)
            if controller.should_stop():
                job.status = JobStatus.STOPPED.value
                job.stage = PipelineStage.TRANSCRIBE.value
                self._save_checkpoint(job)
                self._emit(callback, PipelineStage.TRANSCRIBE, job.status, "Pipeline stopped", job.progress)
                return job.to_dict()
            self._emit(
                callback,
                PipelineStage.EXTRACT,
                job.status,
                f"Prepared audio segment {segment.index + 1}",
                min(0.55, 0.15 + (segment.index * 0.05)),
                {"segment_path": segment.path, "start_seconds": segment.start_seconds},
            )
            transcription = self._run_stage(
                job,
                PipelineStage.TRANSCRIBE,
                callback,
                controller,
                self.speech_to_text.transcribe,
                segment.path,
                source_language=request.source_language,
            )
            transcript_preview.extend(transcription["segments"])
            job.transcript_preview = transcript_preview
            job.progress = min(0.85, 0.55 + ((segment.index + 1) / segment_limit) * 0.3)
            self._save_checkpoint(job)
            if segment.index + 1 >= segment_limit:
                break

        job.progress = 1.0
        job.stage = PipelineStage.COMPLETE.value
        job.status = JobStatus.COMPLETED.value
        result = {
            "status": job.status,
            "stage": job.stage,
            "metadata": job.metadata,
            "quality_control": qc_report,
            "transcript_preview": job.transcript_preview,
            "resources": self._collect_resources(),
        }
        self._save_checkpoint(job)
        self._emit(callback, PipelineStage.COMPLETE, job.status, "Analysis pipeline completed", 1.0, result)
        return result

    def _run_stage(self, job: PipelineJobState, stage: PipelineStage, callback: StatusCallback, controller: JobController, func, *args, **kwargs):
        attempts = 0
        while True:
            self._wait_if_paused(job, callback, controller)
            if controller.should_stop():
                raise RuntimeError("Processing stopped by user")
            try:
                attempts += 1
                job.stage = stage.value
                job.attempts[stage.value] = attempts
                progress = self._stage_progress(stage)
                self._emit(callback, stage, job.status, f"Running {stage.value} stage", progress, {"attempt": attempts})
                value = func(*args, **kwargs)
                job.progress = max(job.progress, progress)
                self._emit(callback, stage, job.status, f"Finished {stage.value} stage", max(job.progress, progress), {"attempt": attempts})
                return value
            except Exception as exc:
                job.last_error = str(exc)
                retryable = attempts <= self.config.max_stage_retries
                self._emit(
                    callback,
                    stage,
                    JobStatus.RUNNING.value if retryable else JobStatus.FAILED.value,
                    f"{stage.value} failed: {exc}",
                    job.progress,
                    {"attempt": attempts, "retrying": retryable},
                )
                self._save_checkpoint(job)
                if not retryable:
                    job.status = JobStatus.FAILED.value
                    raise
                time.sleep(self.config.stage_retry_backoff_seconds)

    def _load_or_create_job(self, request: PipelineRequest) -> PipelineJobState:
        job = PipelineJobState.from_request(request)
        checkpoint = job.checkpoint_path(self.config.checkpoint_dir)
        if checkpoint.exists():
            with checkpoint.open("r", encoding="utf-8") as file:
                stored = json.load(file)
            return PipelineJobState(**stored)
        return job

    def _save_checkpoint(self, job: PipelineJobState) -> None:
        checkpoint = job.checkpoint_path(self.config.checkpoint_dir)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        with checkpoint.open("w", encoding="utf-8") as file:
            json.dump(job.to_dict(), file, indent=2)

    def _emit(self, callback: StatusCallback, stage: PipelineStage, status: str, message: str, progress: float, payload: Optional[Dict[str, object]] = None) -> None:
        if callback:
            callback(PipelineEvent(stage=stage.value, status=status, message=message, progress=progress, payload=payload or {}))

    def _wait_if_paused(self, job: PipelineJobState, callback: StatusCallback, controller: JobController) -> None:
        while controller.is_paused() and not controller.should_stop():
            job.status = JobStatus.PAUSED.value
            self._emit(callback, PipelineStage(job.stage), job.status, "Pipeline paused", job.progress)
            time.sleep(0.2)
        if not controller.should_stop():
            job.status = JobStatus.RUNNING.value

    def _collect_resources(self) -> Dict[str, object]:
        memory = self._memory_info()
        gpu = self._gpu_info()
        return {"memory": memory, "gpu": gpu}

    @staticmethod
    def _memory_info() -> Dict[str, float]:
        try:
            import psutil

            stats = psutil.virtual_memory()
            return {"used_gb": round(stats.used / (1024**3), 2), "total_gb": round(stats.total / (1024**3), 2)}
        except Exception:
            return {"used_gb": 0.0, "total_gb": 0.0}

    @staticmethod
    def _gpu_info() -> Dict[str, Optional[float]]:
        try:
            completed = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=True,
            )
        except Exception:
            return {"used_mb": None, "total_mb": None}
        first_line = (completed.stdout or "").splitlines()[0]
        used_mb, total_mb = [value.strip() for value in first_line.split(",", 1)]
        return {"used_mb": float(used_mb), "total_mb": float(total_mb)}

    @staticmethod
    def _stage_progress(stage: PipelineStage) -> float:
        return {
            PipelineStage.PROBE: 0.1,
            PipelineStage.QUALITY_CONTROL: 0.2,
            PipelineStage.EXTRACT: 0.4,
            PipelineStage.TRANSCRIBE: 0.6,
            PipelineStage.COMPLETE: 1.0,
        }.get(stage, 0.0)
