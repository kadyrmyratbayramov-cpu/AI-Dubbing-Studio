# AI Dubbing Studio

AI Dubbing Studio is a real desktop application for large-video dubbing workflows with PyQt6 UI and disk-backed processing.

## What is implemented

- PyQt6 desktop GUI (main window, toolbar/menu, metadata panel, controls, progress, logs, settings, history)
- Real FFmpeg/ffprobe integration for metadata extraction, audio extraction, segmentation, merge, and final mux
- Disk-backed segmentation pipeline for large files (manifest + checkpoints in `jobs/`)
- Whisper STT (`tiny/base/small/medium`) with word timestamps and CUDA→CPU fallback
- pyannote diarization integration with token-based graceful fallback
- MarianMT translation with HuggingFace Transformers
- Coqui XTTS-v2 synthesis with optional reference voice and CPU fallback
- Timing sync engine (time stretching to segment durations)
- Audio ducking mix stage for background + synthesized speech
- Honest lip-sync status interface (`NOT AVAILABLE` unless implemented later)
- GPU/VRAM status reporting

## Install

> FFmpeg binaries must be installed and available in `PATH` (or set `FFMPEG_BINARY` and `FFPROBE_BINARY`).

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run desktop app

```bash
python -m src.main
```

## Configuration

Primary settings live in `config/config.yaml` and are loaded through `Config.load()`.

Important options:
- `whisper_model` (default `base`)
- `tts_model` (default XTTS-v2)
- `default_language_pairs` (e.g. `en-es`, `es-en`)
- `force_cpu`, `max_vram_gb`
- `huggingface_token` for diarization

## Job output structure

For each run, `jobs/<job_id>/` contains:
- `segments/` extracted audio chunks
- `synthesized/` synthesized and aligned chunks
- `artifacts/transcript.json`
- `artifacts/timing_manifest.json`
- `checkpoint.json`
- `segment_manifest.json`

## Tests

```bash
pytest
```
