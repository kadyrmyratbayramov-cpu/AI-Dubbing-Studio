# AI Dubbing Studio

AI Dubbing Studio is a local-first PyQt6 desktop app for end-to-end video dubbing.

## Current Capabilities

- Modern PyQt6 desktop UI (dark theme)
- Video picker and language pair selection
- Real metadata extraction via `ffprobe`
- End-to-end orchestrator with stage progress
- FFmpeg-based audio extraction and final muxing
- Segment-based audio processing with checkpoint state persistence
- Retry/recovery controls (pause/resume/cancel + stage retries)
- GPU/VRAM-aware device selection with CPU fallback
- Real model adapters:
  - Whisper STT
  - pyannote speaker diarization
  - MarianMT translation
  - Coqui XTTS-v2 TTS
  - Timing stretch, mixing, lip-sync markers, QC checks

## Architecture

```text
src/
├── main.py
├── app.py
├── ui/
├── media/
├── engines/
├── pipeline/
├── core/
├── models/
├── config/
└── utils/
```

## Requirements

- Python 3.10+
- FFmpeg + FFprobe in PATH
- Recommended GPU: NVIDIA RTX 3070 Ti (8GB VRAM)
- Optional but recommended: `HUGGINGFACE_TOKEN` for pyannote model access

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Desktop App

```bash
python -m src.main
```

## Quick Workflow

1. Open app.
2. Select video file.
3. Confirm source and target languages.
4. Click **Start**.
5. Track progress and logs in UI.
6. Output video is written to `output/`.

## Tests

```bash
python -m pytest tests/
```

## Notes

- The pipeline is local/offline-first.
- Marian model coverage is implemented for common language pairs.
- Large files are handled through segmented processing and on-disk checkpoints under `jobs/`.
