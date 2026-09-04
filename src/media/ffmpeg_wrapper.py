"""FFmpeg integration layer for probing and streaming media processing."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict, List


class FFmpegError(RuntimeError):
    """Raised when an ffmpeg/ffprobe call fails."""


class FFmpegWrapper:
    def __init__(self, ffmpeg_bin: str = "ffmpeg", ffprobe_bin: str = "ffprobe"):
        self.ffmpeg_bin = ffmpeg_bin
        self.ffprobe_bin = ffprobe_bin

    def _run(self, command: List[str]) -> subprocess.CompletedProcess:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise FFmpegError(result.stderr.strip() or "ffmpeg command failed")
        return result

    def probe(self, input_path: str) -> Dict[str, object]:
        command = [
            self.ffprobe_bin,
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-print_format",
            "json",
            input_path,
        ]
        result = self._run(command)
        payload = json.loads(result.stdout)
        video_stream = next((s for s in payload.get("streams", []) if s.get("codec_type") == "video"), {})
        audio_stream = next((s for s in payload.get("streams", []) if s.get("codec_type") == "audio"), {})

        width = int(video_stream.get("width", 0) or 0)
        height = int(video_stream.get("height", 0) or 0)
        duration = float(payload.get("format", {}).get("duration", 0.0) or 0.0)
        fps_raw = video_stream.get("avg_frame_rate", "0/1")
        fps = 0.0
        if "/" in fps_raw:
            num, den = fps_raw.split("/")
            fps = float(num) / float(den) if float(den) else 0.0

        return {
            "duration": duration,
            "resolution": f"{width}x{height}" if width and height else "unknown",
            "fps": round(fps, 3),
            "video_codec": video_stream.get("codec_name", "unknown"),
            "audio_codec": audio_stream.get("codec_name", "unknown"),
            "format": payload.get("format", {}).get("format_name", "unknown"),
            "bit_rate": int(payload.get("format", {}).get("bit_rate", 0) or 0),
        }

    def extract_audio(self, input_path: str, output_wav: str, sample_rate: int = 16000, channels: int = 1) -> str:
        command = [
            self.ffmpeg_bin,
            "-y",
            "-i",
            input_path,
            "-vn",
            "-ac",
            str(channels),
            "-ar",
            str(sample_rate),
            "-f",
            "wav",
            output_wav,
        ]
        self._run(command)
        return output_wav

    def segment_audio(self, input_audio: str, segment_seconds: int, output_pattern: str) -> List[str]:
        command = [
            self.ffmpeg_bin,
            "-y",
            "-i",
            input_audio,
            "-f",
            "segment",
            "-segment_time",
            str(segment_seconds),
            "-c",
            "copy",
            output_pattern,
        ]
        self._run(command)
        segment_dir = Path(output_pattern).parent
        pattern_name = Path(output_pattern).name.replace("%04d", "")
        return sorted(str(path) for path in segment_dir.glob(f"*{pattern_name}"))

    def mux_video_with_audio(self, input_video: str, input_audio: str, output_video: str) -> str:
        command = [
            self.ffmpeg_bin,
            "-y",
            "-i",
            input_video,
            "-i",
            input_audio,
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            output_video,
        ]
        self._run(command)
        return output_video
