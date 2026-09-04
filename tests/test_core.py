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
        assert Path(config.output_dir).exists()

    def test_config_to_dict(self):
        config = Config()
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert "sample_rate" in config_dict

    def test_nested_config_loading(self, tmp_path: Path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "audio:\n  sample_rate: 44100\noutput:\n  output_dir: custom-output\n",
            encoding="utf-8",
        )
        config = Config(str(config_file))
        assert config.sample_rate == 44100
        assert config.output_dir.endswith("custom-output")


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
        result = orchestrator.run(
            PipelineRequest(input_file=str(input_file)),
            callback=events.append,
            controller=JobController(),
        )

        checkpoint_file = Path(config.checkpoint_dir) / "video.json"
        assert result["status"] == JobStatus.COMPLETED.value
        assert checkpoint_file.exists()
        stored = json.loads(checkpoint_file.read_text(encoding="utf-8"))
        assert stored["stage"] == "complete"
        assert events[-1].stage == "complete"
