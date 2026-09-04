"""Tests for core modules."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config.settings import Config
from src.core.audio_processor import AudioProcessor
from src.core.dubbing_pipeline import DubbingPipeline
from src.core.job_state import JobController, JobStatus, PipelineEvent, PipelineRequest
from src.core.orchestrator import DubbingOrchestrator
from src.core.video_metadata import VideoMetadata


class TestConfig:
    def test_config_initialization(self):
        config = Config()
        assert config.sample_rate == 16000
        assert config.channels == 1
        assert config.source_language == "auto"
        assert Path(config.output_dir).exists()

    def test_config_to_dict(self):
        config = Config()
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert "sample_rate" in config_dict

    def test_nested_config_loading(self, tmp_path: Path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "audio:\n  sample_rate: 44100\n"
            "output:\n  output_dir: custom-output\n"
            "processing:\n  max_stage_retries: 4\n  stage_retry_backoff_seconds: 2\n"
            "media:\n  ffprobe_binary: ffprobe-custom\n"
            "dubbing:\n  source_language: tr\n",
            encoding="utf-8",
        )
        config = Config(str(config_file))
        assert config.sample_rate == 44100
        assert config.output_dir.endswith("custom-output")
        assert config.max_stage_retries == 4
        assert config.stage_retry_backoff_seconds == 2
        assert config.ffprobe_binary == "ffprobe-custom"
        assert config.source_language == "tr"


class TestAudioProcessor:
    def test_audio_processor_initialization(self):
        config = Config()
        processor = AudioProcessor(config)
        assert processor.sample_rate == config.sample_rate


class TestDubbingPipeline:
    def test_pipeline_initialization(self):
        config = Config()
        pipeline = DubbingPipeline(config)
        assert pipeline.config == config
        assert pipeline.audio_processor is not None
        assert pipeline.voice_synthesis is not None
        assert pipeline.orchestrator is not None


class TestOrchestrator:
    def test_orchestrator_run_saves_checkpoint(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        config = Config()
        config.checkpoint_dir = str(tmp_path / "checkpoints")
        config.workspace_dir = str(tmp_path / "workspace")
        config.max_transcription_segments = 1
        orchestrator = DubbingOrchestrator(config)
        input_file = tmp_path / "video.mp4"
        input_file.write_bytes(b"video")

        metadata = VideoMetadata(
            file_path=str(input_file),
            file_size_bytes=5,
            duration_seconds=8.0,
            width=1920,
            height=1080,
            frame_rate=24.0,
            video_codec="h264",
            audio_codec="aac",
            audio_sample_rate=48000,
        )

        monkeypatch.setattr(orchestrator.media_pipeline, "probe", lambda _: metadata)
        monkeypatch.setattr(
            orchestrator.media_pipeline,
            "iter_audio_segments",
            lambda *_args, **_kwargs: iter(
                [
                    type(
                        "Segment",
                        (),
                        {
                            "index": 0,
                            "path": str(tmp_path / "segment.wav"),
                            "start_seconds": 0.0,
                            "duration_seconds": 8.0,
                        },
                    )()
                ]
            ),
        )
        monkeypatch.setattr(
            orchestrator.speech_to_text,
            "transcribe",
            lambda *_args, **_kwargs: {
                "segments": [
                    {
                        "start": 0.0,
                        "end": 1.0,
                        "text": "hello",
                        "language": "en",
                    }
                ]
            },
        )

        events: list[PipelineEvent] = []
        request = PipelineRequest(input_file=str(input_file))
        result = orchestrator.run(
            request,
            callback=events.append,
            controller=JobController(),
        )

        saved_files = list(Path(config.checkpoint_dir).glob("video-*.json"))
        assert result["status"] == JobStatus.COMPLETED.value
        assert len(saved_files) == 1
        stored = json.loads(saved_files[0].read_text(encoding="utf-8"))
        assert stored["stage"] == "complete"
        assert stored["completed_segments"] == [0]
        assert events[-1].stage == "complete"

    def test_orchestrator_failure_records_retry_state(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        config = Config()
        config.checkpoint_dir = str(tmp_path / "checkpoints")
        config.max_stage_retries = 0
        config.stage_retry_backoff_seconds = 0
        orchestrator = DubbingOrchestrator(config)
        input_file = tmp_path / "video.mp4"
        input_file.write_bytes(b"video")

        def fail_probe(_path: str):
            raise RuntimeError("probe failed")

        monkeypatch.setattr(orchestrator.media_pipeline, "probe", fail_probe)

        events: list[PipelineEvent] = []
        request = PipelineRequest(input_file=str(input_file))
        with pytest.raises(RuntimeError, match="probe failed"):
            orchestrator.run(
                request,
                callback=events.append,
                controller=JobController(),
            )

        saved_files = list(Path(config.checkpoint_dir).glob("video-*.json"))
        assert len(saved_files) == 1
        stored = json.loads(saved_files[0].read_text(encoding="utf-8"))
        assert stored["last_error"] == "probe failed"
        assert stored["attempts"]["probe"] == 1
        assert events[-1].status == JobStatus.FAILED.value
        assert events[-1].payload["retrying"] is False

    def test_orchestrator_reuses_saved_checkpoint_data(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        config = Config()
        config.checkpoint_dir = str(tmp_path / "checkpoints")
        config.workspace_dir = str(tmp_path / "workspace")
        config.max_transcription_segments = 1
        orchestrator = DubbingOrchestrator(config)
        input_file = tmp_path / "video.mp4"
        input_file.write_bytes(b"video")

        request = PipelineRequest(
            input_file=str(input_file),
            source_language="tr",
            target_language="en",
        )
        checkpoint_path = orchestrator._load_or_create_job(request).checkpoint_path(
            config.checkpoint_dir
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text(
            json.dumps(
                {
                    "input_file": str(input_file),
                    "source_language": "auto",
                    "target_language": "de",
                    "output_dir": None,
                    "status": "paused",
                    "stage": "transcribe",
                    "progress": 0.6,
                    "attempts": {"probe": 1},
                    "metadata": {"file_path": str(input_file)},
                    "transcript_preview": [
                        {
                            "start": 0.0,
                            "end": 0.8,
                            "text": "existing",
                            "language": "en",
                        }
                    ],
                    "completed_segments": [0],
                    "last_error": None,
                }
            ),
            encoding="utf-8",
        )

        metadata = VideoMetadata(
            file_path=str(input_file),
            file_size_bytes=5,
            duration_seconds=8.0,
            width=1920,
            height=1080,
            frame_rate=24.0,
            video_codec="h264",
            audio_codec="aac",
            audio_sample_rate=48000,
        )

        monkeypatch.setattr(orchestrator.media_pipeline, "probe", lambda _: metadata)
        monkeypatch.setattr(
            orchestrator.media_pipeline,
            "iter_audio_segments",
            lambda *_args, **_kwargs: iter(
                [
                    type(
                        "Segment",
                        (),
                        {
                            "index": 0,
                            "path": str(tmp_path / "segment.wav"),
                            "start_seconds": 0.0,
                            "duration_seconds": 8.0,
                        },
                    )()
                ]
            ),
        )

        result = orchestrator.run(request, controller=JobController())

        assert result["status"] == JobStatus.COMPLETED.value
        assert result["transcript_preview"][0]["text"] == "existing"
        resumed_job = orchestrator._load_or_create_job(request)
        assert resumed_job.source_language == "tr"
        assert resumed_job.target_language == "en"
