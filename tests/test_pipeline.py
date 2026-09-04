"""Tests for pipeline and media orchestration helpers."""

from pathlib import Path

from src.config.settings import Config
from src.media.ffmpeg_wrapper import FFmpegWrapper
from src.pipeline.job_manager import JobManager
from src.pipeline.state_manager import StateManager


def test_config_creates_runtime_directories(tmp_path: Path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
output:
  output_dir: out
  log_dir: logs
pipeline:
  jobs_dir: jobs
  workspace_dir: workspace
""",
        encoding="utf-8",
    )

    config = Config(str(config_file))

    assert Path(config.output_dir).exists()
    assert Path(config.log_dir).exists()
    assert Path(config.jobs_dir).exists()
    assert Path(config.workspace_dir).exists()


def test_state_manager_roundtrip(tmp_path: Path):
    manager = StateManager(str(tmp_path / "jobs"))
    payload = {"status": "running", "stage": "transcription"}
    manager.save("abc", payload)

    loaded = manager.load("abc")
    assert loaded is not None
    assert loaded["status"] == "running"
    assert loaded["stage"] == "transcription"
    assert "updated_at" in loaded


def test_job_manager_create(tmp_path: Path):
    manager = JobManager(str(tmp_path / "jobs"))
    job = manager.create("input.mp4", "en", "es")

    assert job.input_video == "input.mp4"
    assert job.source_language == "en"
    assert job.target_language == "es"
    assert (tmp_path / "jobs" / job.id).exists()


def test_ffmpeg_probe_parsing(monkeypatch):
    wrapper = FFmpegWrapper()

    class _Result:
        returncode = 0
        stdout = (
            '{"streams":[{"codec_type":"video","width":1920,"height":1080,'
            '"codec_name":"h264","avg_frame_rate":"30000/1001"},'
            '{"codec_type":"audio","codec_name":"aac"}],'
            '"format":{"duration":"12.5","format_name":"mov,mp4,m4a","bit_rate":"256000"}}'
        )
        stderr = ""

    monkeypatch.setattr(wrapper, "_run", lambda cmd: _Result())

    metadata = wrapper.probe("video.mp4")
    assert metadata["duration"] == 12.5
    assert metadata["resolution"] == "1920x1080"
    assert metadata["video_codec"] == "h264"
    assert metadata["audio_codec"] == "aac"
