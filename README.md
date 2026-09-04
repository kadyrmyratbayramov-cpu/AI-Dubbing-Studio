# AI Dubbing Studio

AI Dubbing Studio is a local-first desktop application for inspecting video files and running the opening stages of a real dubbing pipeline.

## Current deliverable

This repository now includes:
- a desktop GUI built with Tkinter/ttk
- native video file selection
- source and target language selection
- start, pause, resume, and stop job controls
- real video metadata inspection through `ffprobe`
- FFmpeg-based chunked audio extraction for large files
- checkpointed orchestration with retry support
- a real Whisper adapter interface using `faster-whisper`
- adapter scaffolding for diarization, translation, TTS, lip-sync, timing, mixing, and quality control

The first milestone is supported: a user can open the app, select a video, inspect metadata, and start a real analysis pipeline that progresses into transcription when the STT dependency and local model are available.

## Installation

### Base environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Desktop and video metadata support

```bash
pip install -e .[desktop]
```

### Enable local Whisper transcription

```bash
pip install -e .[stt]
```

### Full local AI stack

```bash
pip install -e .[full]
```

## System dependencies

Install `ffmpeg` and `ffprobe` and ensure they are on `PATH`.

## Run the desktop app

```bash
ai-dubbing-studio
```

## Tests

```bash
python -m pytest
```
