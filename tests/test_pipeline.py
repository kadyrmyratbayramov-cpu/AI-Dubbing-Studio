"""Tests for pipeline and media orchestration helpers."""

import threading
import time
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from src.config.settings import Config
from src.media.ffmpeg_wrapper import FFmpegWrapper
from src.pipeline.job_manager import JobManager
from src.pipeline.orchestrator import DubbingOrchestrator
from src.pipeline.state_manager import StateManager
from src.media.segment_processor import AudioSegmentRef


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


def test_orchestrator_cancel_persists_checkpoint(tmp_path: Path, monkeypatch):
    config = Config()
    config.jobs_dir = str(tmp_path / "jobs")
    config.output_dir = str(tmp_path / "out")
    config.log_dir = str(tmp_path / "logs")
    config.retry_attempts = 0
    config._normalize_and_create_paths()

    orchestrator = DubbingOrchestrator(config)
    audio_path = tmp_path / "audio.wav"
    sf.write(audio_path, np.zeros(16000, dtype=np.float32), 16000)

    monkeypatch.setattr(orchestrator.ffmpeg, "probe", lambda *_: {"duration": 1.0})

    def _extract(_input_video, output_wav, *_args):
        sf.write(output_wav, np.zeros(16000, dtype=np.float32), 16000)
        return output_wav

    monkeypatch.setattr(orchestrator.ffmpeg, "extract_audio", _extract)
    monkeypatch.setattr(orchestrator.segment_processor, "create_segments", lambda *_: [AudioSegmentRef(0, str(audio_path), 0.0, 1.0)])
    monkeypatch.setattr("src.pipeline.orchestrator.WhisperSTTEngine", lambda *args, **kwargs: None)

    def _cancel_on_first_progress(*_args, **_kwargs):
        orchestrator.cancel()

    try:
        orchestrator.process("video.mp4", "en", "es", progress=_cancel_on_first_progress)
    except RuntimeError:
        pass

    state_files = sorted((tmp_path / "jobs").rglob("state.json"))
    assert state_files
    state = state_files[-1].read_text(encoding="utf-8")
    assert "\"status\": \"cancelled\"" in state


def test_orchestrator_pause_waits_for_resume(tmp_path: Path, monkeypatch):
    config = Config()
    config.jobs_dir = str(tmp_path / "jobs")
    config.output_dir = str(tmp_path / "out")
    config.log_dir = str(tmp_path / "logs")
    config.retry_attempts = 0
    config._normalize_and_create_paths()

    orchestrator = DubbingOrchestrator(config)
    audio_path = tmp_path / "audio.wav"
    sf.write(audio_path, np.zeros(16000, dtype=np.float32), 16000)

    monkeypatch.setattr(orchestrator.ffmpeg, "probe", lambda *_: {"duration": 1.0})

    def _extract(_input_video, output_wav, *_args):
        sf.write(output_wav, np.zeros(16000, dtype=np.float32), 16000)
        return output_wav

    monkeypatch.setattr(orchestrator.ffmpeg, "extract_audio", _extract)
    monkeypatch.setattr(orchestrator.segment_processor, "create_segments", lambda *_: [AudioSegmentRef(0, str(audio_path), 0.0, 1.0)])

    class _STT:
        def __init__(self, *_, **__):
            pass

        def transcribe(self, *_args, **_kwargs):
            return {"text": "hello", "segments": [{"start": 0.0, "end": 1.0}]}

    class _Diar:
        def __init__(self, *_, **__):
            pass

        def diarize(self, *_args, **_kwargs):
            return []

    class _Trans:
        def translate(self, texts, *_args, **_kwargs):
            return list(texts)

    class _TTS:
        def __init__(self, *_, **__):
            pass

        def synthesize_to_file(self, text, output_path, language, **_kwargs):
            sf.write(output_path, np.zeros(16000, dtype=np.float32), 16000)
            return output_path

    class _Timing:
        def stretch_to_duration(self, input_audio, target_duration_sec, output_audio):
            sf.write(output_audio, np.zeros(16000, dtype=np.float32), 16000)
            return output_audio

    class _Mix:
        def mix_voice_with_background(self, voice_audio, background_audio, output_audio, ducking_db=10.0):
            sf.write(output_audio, np.zeros(16000, dtype=np.float32), 16000)
            return output_audio

    class _Lip:
        def build_sync_markers(self, transcript_segments):
            return transcript_segments

    class _QC:
        def run(self, *_args, **_kwargs):
            return {"passed": True, "issues": []}

    monkeypatch.setattr("src.pipeline.orchestrator.WhisperSTTEngine", _STT)
    monkeypatch.setattr("src.pipeline.orchestrator.PyannoteDiarizationEngine", _Diar)
    monkeypatch.setattr("src.pipeline.orchestrator.MarianTranslationEngine", _Trans)
    monkeypatch.setattr("src.pipeline.orchestrator.XTTSv2Engine", _TTS)
    monkeypatch.setattr("src.pipeline.orchestrator.TimingEngine", _Timing)
    monkeypatch.setattr("src.pipeline.orchestrator.MixingEngine", _Mix)
    monkeypatch.setattr("src.pipeline.orchestrator.LipSyncEngine", _Lip)
    monkeypatch.setattr("src.pipeline.orchestrator.QualityControlEngine", _QC)
    monkeypatch.setattr(orchestrator.ffmpeg, "mux_video_with_audio", lambda *_: str(tmp_path / "out.mp4"))

    outcome = {"done": False}
    reached_pause = threading.Event()

    def _run():
        orchestrator.process("video.mp4", "en", "es", progress=_progress_callback)
        outcome["done"] = True

    def _progress_callback(stage, *_args):
        if stage == "validation" and not reached_pause.is_set():
            orchestrator.pause()
            reached_pause.set()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    reached_pause.wait(timeout=1)
    time.sleep(0.3)
    assert not outcome["done"]

    orchestrator.resume()
    thread.join(timeout=2)
    assert outcome["done"]


def test_orchestrator_resume_from_checkpoint_reuses_saved_context(tmp_path: Path, monkeypatch):
    config = Config()
    config.jobs_dir = str(tmp_path / "jobs")
    config.output_dir = str(tmp_path / "out")
    config.log_dir = str(tmp_path / "logs")
    config.retry_attempts = 0
    config._normalize_and_create_paths()

    orchestrator = DubbingOrchestrator(config)
    job = orchestrator.job_manager.create("saved_input.mp4", "en", "fr")
    job_dir = Path(config.jobs_dir) / job.id
    job_dir.mkdir(parents=True, exist_ok=True)

    input_audio = job_dir / "input_audio.wav"
    segment_file = job_dir / "segments" / "segment_0000.wav"
    segment_file.parent.mkdir(parents=True, exist_ok=True)
    sf.write(input_audio, np.zeros(16000, dtype=np.float32), 16000)
    sf.write(segment_file, np.zeros(16000, dtype=np.float32), 16000)
    (job_dir / "segment_manifest.json").write_text(
        json.dumps([{"index": 0, "path": str(segment_file), "start_sec": 0.0, "end_sec": 1.0}]),
        encoding="utf-8",
    )

    orchestrator.state_manager.save(
        job.id,
        {
            "job_id": job.id,
            "input_video": "saved_input.mp4",
            "source_language": "en",
            "target_language": "fr",
            "status": "running",
            "stage": "segmented",
            "metadata": {"duration": 1.0},
        },
    )

    monkeypatch.setattr(orchestrator.ffmpeg, "probe", lambda *_: (_ for _ in ()).throw(AssertionError("probe should not run on resume")))

    class _STT:
        def __init__(self, *_, **__):
            pass

        def transcribe(self, *_args, **_kwargs):
            return {"text": "hello", "segments": [{"start": 0.0, "end": 1.0}]}

    class _Diar:
        def __init__(self, *_, **__):
            pass

        def diarize(self, *_args, **_kwargs):
            return []

    class _Trans:
        def translate(self, texts, *_args, **_kwargs):
            return [f"fr:{t}" for t in texts]

    class _TTS:
        def __init__(self, *_, **__):
            pass

        def synthesize_to_file(self, text, output_path, language, **_kwargs):
            sf.write(output_path, np.zeros(16000, dtype=np.float32), 16000)
            return output_path

    class _Timing:
        def stretch_to_duration(self, input_audio, target_duration_sec, output_audio):
            sf.write(output_audio, np.zeros(16000, dtype=np.float32), 16000)
            return output_audio

    class _Mix:
        def mix_voice_with_background(self, voice_audio, background_audio, output_audio, ducking_db=10.0):
            sf.write(output_audio, np.zeros(16000, dtype=np.float32), 16000)
            return output_audio

    class _Lip:
        def build_sync_markers(self, transcript_segments):
            return transcript_segments

    class _QC:
        def run(self, *_args, **_kwargs):
            return {"passed": True, "issues": []}

    monkeypatch.setattr("src.pipeline.orchestrator.WhisperSTTEngine", _STT)
    monkeypatch.setattr("src.pipeline.orchestrator.PyannoteDiarizationEngine", _Diar)
    monkeypatch.setattr("src.pipeline.orchestrator.MarianTranslationEngine", _Trans)
    monkeypatch.setattr("src.pipeline.orchestrator.XTTSv2Engine", _TTS)
    monkeypatch.setattr("src.pipeline.orchestrator.TimingEngine", _Timing)
    monkeypatch.setattr("src.pipeline.orchestrator.MixingEngine", _Mix)
    monkeypatch.setattr("src.pipeline.orchestrator.LipSyncEngine", _Lip)
    monkeypatch.setattr("src.pipeline.orchestrator.QualityControlEngine", _QC)
    monkeypatch.setattr(orchestrator.ffmpeg, "mux_video_with_audio", lambda *_: str(tmp_path / "out.mp4"))

    result = orchestrator.process("ignored.mp4", "es", "de", resume_job_id=job.id)
    assert result["source_language"] == "en"
    assert result["target_language"] == "fr"
    assert result["status"] == "completed"


def test_orchestrator_resume_without_manifest_rebuilds_segment_timing(tmp_path: Path, monkeypatch):
    config = Config()
    config.jobs_dir = str(tmp_path / "jobs")
    config.output_dir = str(tmp_path / "out")
    config.log_dir = str(tmp_path / "logs")
    config.retry_attempts = 0
    config._normalize_and_create_paths()

    orchestrator = DubbingOrchestrator(config)
    job = orchestrator.job_manager.create("saved_input.mp4", "en", "fr")
    job_dir = Path(config.jobs_dir) / job.id
    job_dir.mkdir(parents=True, exist_ok=True)

    input_audio = job_dir / "input_audio.wav"
    segment_file = job_dir / "segments" / "segment_0000.wav"
    segment_file.parent.mkdir(parents=True, exist_ok=True)
    sf.write(input_audio, np.zeros(16000, dtype=np.float32), 16000)
    sf.write(segment_file, np.zeros(16000, dtype=np.float32), 16000)

    orchestrator.state_manager.save(
        job.id,
        {
            "job_id": job.id,
            "input_video": "saved_input.mp4",
            "source_language": "en",
            "target_language": "fr",
            "status": "running",
            "stage": "segmented",
            "metadata": {"duration": 1.0},
        },
    )

    class _STT:
        def __init__(self, *_, **__):
            pass

        def transcribe(self, *_args, **_kwargs):
            return {"text": "hello", "segments": [{"start": 0.0, "end": 1.0}]}

    class _Diar:
        def __init__(self, *_, **__):
            pass

        def diarize(self, *_args, **_kwargs):
            return []

    class _Trans:
        def translate(self, texts, *_args, **_kwargs):
            return texts

    class _TTS:
        def __init__(self, *_, **__):
            pass

        def synthesize_to_file(self, text, output_path, language, **_kwargs):
            sf.write(output_path, np.zeros(16000, dtype=np.float32), 16000)
            return output_path

    class _Timing:
        def stretch_to_duration(self, input_audio, target_duration_sec, output_audio):
            sf.write(output_audio, np.zeros(16000, dtype=np.float32), 16000)
            return output_audio

    class _Mix:
        def mix_voice_with_background(self, voice_audio, background_audio, output_audio, ducking_db=10.0):
            sf.write(output_audio, np.zeros(16000, dtype=np.float32), 16000)
            return output_audio

    class _Lip:
        def build_sync_markers(self, transcript_segments):
            return transcript_segments

    class _QC:
        def run(self, *_args, **_kwargs):
            return {"passed": True, "issues": []}

    monkeypatch.setattr("src.pipeline.orchestrator.WhisperSTTEngine", _STT)
    monkeypatch.setattr("src.pipeline.orchestrator.PyannoteDiarizationEngine", _Diar)
    monkeypatch.setattr("src.pipeline.orchestrator.MarianTranslationEngine", _Trans)
    monkeypatch.setattr("src.pipeline.orchestrator.XTTSv2Engine", _TTS)
    monkeypatch.setattr("src.pipeline.orchestrator.TimingEngine", _Timing)
    monkeypatch.setattr("src.pipeline.orchestrator.MixingEngine", _Mix)
    monkeypatch.setattr("src.pipeline.orchestrator.LipSyncEngine", _Lip)
    monkeypatch.setattr("src.pipeline.orchestrator.QualityControlEngine", _QC)
    monkeypatch.setattr(orchestrator.ffmpeg, "mux_video_with_audio", lambda *_: str(tmp_path / "out.mp4"))

    result = orchestrator.process("ignored.mp4", "es", "de", resume_job_id=job.id)
    assert result["status"] == "completed"
    assert result["translated_segments"][0]["start"] >= 0.0
