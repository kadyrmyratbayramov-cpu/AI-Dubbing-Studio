"""FFmpeg/ffprobe integration layer."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Dict, List, Optional

from src.core.types import AudioStreamInfo, Segment, VideoMetadata


ProgressCallback = Optional[Callable[[float, str], None]]


class FFmpegError(RuntimeError):
    """Raised for FFmpeg related failures."""


class FFmpegEngine:
    def __init__(self, ffmpeg_path: Optional[str] = None, ffprobe_path: Optional[str] = None) -> None:
        self.ffmpeg_path = ffmpeg_path or self._detect_binary("ffmpeg")
        self.ffprobe_path = ffprobe_path or self._detect_binary("ffprobe")

    def _detect_binary(self, binary_name: str) -> str:
        env_key = f"{binary_name.upper()}_BINARY"
        from_env = os.getenv(env_key)
        if from_env and Path(from_env).exists():
            return from_env

        found = shutil.which(binary_name)
        if found:
            return found

        windows_candidates = [
            Path("C:/ffmpeg/bin") / f"{binary_name}.exe",
            Path("C:/Program Files/ffmpeg/bin") / f"{binary_name}.exe",
        ]
        for candidate in windows_candidates:
            if candidate.exists():
                return str(candidate)

        raise FFmpegError(
            f"Could not locate {binary_name}. Install FFmpeg and add it to PATH or set {env_key}."
        )

    def probe_video(self, input_path: str) -> VideoMetadata:
        command = [
            self.ffprobe_path,
            "-v",
            "error",
            "-show_entries",
            "stream=index,codec_type,codec_name,width,height,r_frame_rate,channels,sample_rate,bit_rate:stream_tags=language:format=duration,bit_rate",
            "-of",
            "json",
            input_path,
        ]
        proc = subprocess.run(command, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise FFmpegError(proc.stderr.strip() or "ffprobe failed")

        data = json.loads(proc.stdout)
        streams = data.get("streams", [])
        format_info = data.get("format", {})

        video_stream = next((s for s in streams if s.get("codec_type") == "video"), {})
        audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), {})
        subtitle_stream_count = len([s for s in streams if s.get("codec_type") == "subtitle"])

        audio_streams: List[AudioStreamInfo] = []
        for stream in streams:
            if stream.get("codec_type") != "audio":
                continue
            tags = stream.get("tags", {})
            lang = tags.get("language", "und")
            audio_streams.append(
                AudioStreamInfo(
                    index=int(stream.get("index", 0)),
                    codec=stream.get("codec_name", "unknown"),
                    channels=int(stream.get("channels", 1)),
                    sample_rate=int(stream.get("sample_rate", 0) or 0),
                    language=lang,
                )
            )

        fps_raw = str(video_stream.get("r_frame_rate", "0/1"))
        fps_parts = fps_raw.split("/")
        fps = 0.0
        if len(fps_parts) == 2 and fps_parts[1] != "0":
            fps = float(fps_parts[0]) / float(fps_parts[1])

        return VideoMetadata(
            path=input_path,
            duration=float(format_info.get("duration", 0.0) or 0.0),
            width=int(video_stream.get("width", 0) or 0),
            height=int(video_stream.get("height", 0) or 0),
            fps=fps,
            video_codec=video_stream.get("codec_name", "unknown"),
            audio_codec=audio_stream.get("codec_name", "unknown"),
            bitrate=int(format_info.get("bit_rate", 0) or 0),
            audio_streams=audio_streams,
            subtitle_streams=subtitle_stream_count,
        )

    def extract_audio(
        self,
        input_path: str,
        output_wav: str,
        sample_rate: int = 16000,
        channels: int = 1,
        progress: ProgressCallback = None,
        duration_hint: Optional[float] = None,
    ) -> None:
        command = [
            self.ffmpeg_path,
            "-y",
            "-i",
            input_path,
            "-vn",
            "-ac",
            str(channels),
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            "-progress",
            "pipe:1",
            "-nostats",
            output_wav,
        ]
        self._run_with_progress(command, progress, duration_hint, "audio extraction")

    def split_audio_segments(
        self,
        input_wav: str,
        output_dir: str,
        segment_seconds: int,
        duration_seconds: float,
    ) -> List[Segment]:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        pattern = str(Path(output_dir) / "segment_%05d.wav")
        command = [
            self.ffmpeg_path,
            "-y",
            "-i",
            input_wav,
            "-f",
            "segment",
            "-segment_time",
            str(segment_seconds),
            "-c",
            "copy",
            pattern,
        ]
        proc = subprocess.run(command, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise FFmpegError(proc.stderr.strip() or "audio segmentation failed")

        files = sorted(Path(output_dir).glob("segment_*.wav"))
        segments: List[Segment] = []
        for idx, segment_file in enumerate(files):
            start = idx * segment_seconds
            end = min((idx + 1) * segment_seconds, duration_seconds)
            segments.append(Segment(id=f"seg-{idx:05d}", index=idx, start=float(start), end=float(end), path=str(segment_file)))
        return segments

    def merge_audio_segments(self, segment_paths: List[str], output_wav: str) -> None:
        if not segment_paths:
            raise FFmpegError("No segments to merge")
        concat_file = Path(output_wav).with_suffix(".concat.txt")
        escaped_paths = []
        for path in segment_paths:
            escaped = Path(path).as_posix()
            escaped = escaped.replace("\\", "\\\\").replace(" ", "\\ ").replace("'", "\\'")
            escaped_paths.append(escaped)
        concat_file.write_text("\n".join([f"file {p}" for p in escaped_paths]), encoding="utf-8")

        command = [
            self.ffmpeg_path,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-c",
            "copy",
            output_wav,
        ]
        proc = subprocess.run(command, capture_output=True, text=True, check=False)
        concat_file.unlink(missing_ok=True)
        if proc.returncode != 0:
            raise FFmpegError(proc.stderr.strip() or "segment merge failed")

    def mux_audio_with_video(
        self,
        source_video: str,
        dubbed_audio: str,
        output_path: str,
        progress: ProgressCallback = None,
        duration_hint: Optional[float] = None,
    ) -> None:
        copy_command = [
            self.ffmpeg_path,
            "-y",
            "-i",
            source_video,
            "-i",
            dubbed_audio,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-metadata",
            "comment=Dubbed with AI Dubbing Studio",
            "-progress",
            "pipe:1",
            "-nostats",
            output_path,
        ]

        try:
            self._run_with_progress(copy_command, progress, duration_hint, "final mux")
            return
        except FFmpegError:
            pass

        reencode_command = [
            self.ffmpeg_path,
            "-y",
            "-i",
            source_video,
            "-i",
            dubbed_audio,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-metadata",
            "comment=Dubbed with AI Dubbing Studio",
            "-progress",
            "pipe:1",
            "-nostats",
            output_path,
        ]
        self._run_with_progress(reencode_command, progress, duration_hint, "final mux re-encode")

    def _run_with_progress(
        self,
        command: List[str],
        callback: ProgressCallback,
        duration_hint: Optional[float],
        stage: str,
    ) -> None:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            while True:
                line = process.stdout.readline() if process.stdout else ""
                if not line and process.poll() is not None:
                    break
                if not line:
                    continue
                line = line.strip()
                if callback and line.startswith("out_time_ms=") and duration_hint and duration_hint > 0:
                    out_time_ms = int(line.split("=", 1)[1])
                    ratio = min(100.0, (out_time_ms / 1_000_000.0) / duration_hint * 100.0)
                    callback(ratio, stage)

            return_code = process.wait()
            if return_code != 0:
                stderr = process.stderr.read() if process.stderr else ""
                raise FFmpegError(stderr.strip() or f"FFmpeg failed for stage {stage}")
            if callback:
                callback(100.0, stage)
        finally:
            if process.stdout:
                process.stdout.close()
            if process.stderr:
                process.stderr.close()
