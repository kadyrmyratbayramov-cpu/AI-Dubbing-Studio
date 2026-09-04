"""Tkinter desktop UI for AI Dubbing Studio."""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Dict, Optional

from src.config.settings import Config
from src.core.job_state import JobController, PipelineEvent
from src.core.orchestrator import DubbingOrchestrator
from src.core.video_metadata import VideoMetadataReader


class MainWindow:
    def __init__(self, root: tk.Tk, config: Config):
        self.root = root
        self.config = config
        self.orchestrator = DubbingOrchestrator(config)
        self.metadata_reader = VideoMetadataReader(config)
        self.event_queue: "queue.Queue[PipelineEvent | Dict[str, object]]" = queue.Queue()
        self.controller: Optional[JobController] = None
        self.worker: Optional[threading.Thread] = None

        self.selected_file = tk.StringVar()
        self.source_language = tk.StringVar(value="auto")
        self.target_language = tk.StringVar(value="en")
        self.status_text = tk.StringVar(value="Ready")

        self._build()
        self.root.after(150, self._poll_queue)

    def _build(self) -> None:
        self.root.geometry("980x720")
        self.root.minsize(880, 640)
        main = ttk.Frame(self.root, padding=16)
        main.pack(fill=tk.BOTH, expand=True)

        file_frame = ttk.LabelFrame(main, text="Video Input", padding=12)
        file_frame.pack(fill=tk.X)
        ttk.Entry(file_frame, textvariable=self.selected_file).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        ttk.Button(file_frame, text="Select Video", command=self.select_video).pack(side=tk.LEFT)

        language_frame = ttk.LabelFrame(main, text="Languages", padding=12)
        language_frame.pack(fill=tk.X, pady=12)
        options = [entry["code"] for entry in self.config.available_languages]
        ttk.Label(language_frame, text="Source").grid(row=0, column=0, sticky=tk.W)
        ttk.Combobox(language_frame, values=options, textvariable=self.source_language, state="readonly", width=18).grid(row=0, column=1, padx=(8, 20), sticky=tk.W)
        ttk.Label(language_frame, text="Target").grid(row=0, column=2, sticky=tk.W)
        ttk.Combobox(language_frame, values=options[1:], textvariable=self.target_language, state="readonly", width=18).grid(row=0, column=3, padx=(8, 0), sticky=tk.W)

        control_frame = ttk.LabelFrame(main, text="Controls", padding=12)
        control_frame.pack(fill=tk.X)
        ttk.Button(control_frame, text="Start", command=self.start_pipeline).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(control_frame, text="Pause", command=self.pause_pipeline).pack(side=tk.LEFT, padx=8)
        ttk.Button(control_frame, text="Resume", command=self.resume_pipeline).pack(side=tk.LEFT, padx=8)
        ttk.Button(control_frame, text="Stop", command=self.stop_pipeline).pack(side=tk.LEFT, padx=8)

        status_frame = ttk.LabelFrame(main, text="Status", padding=12)
        status_frame.pack(fill=tk.X, pady=12)
        ttk.Label(status_frame, textvariable=self.status_text).pack(anchor=tk.W)
        self.progress = ttk.Progressbar(status_frame, mode="determinate", maximum=100)
        self.progress.pack(fill=tk.X, pady=(8, 0))

        metadata_frame = ttk.LabelFrame(main, text="Metadata", padding=12)
        metadata_frame.pack(fill=tk.X)
        self.metadata_text = tk.Text(metadata_frame, height=8, wrap=tk.WORD)
        self.metadata_text.pack(fill=tk.BOTH, expand=True)

        log_frame = ttk.LabelFrame(main, text="Pipeline Events", padding=12)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(12, 0))
        self.log_text = tk.Text(log_frame, height=16, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def select_video(self) -> None:
        path = filedialog.askopenfilename(
            title="Select a video file",
            filetypes=[("Video files", "*.mp4 *.mov *.mkv *.avi *.m4v *.webm"), ("All files", "*.*")],
        )
        if not path:
            return
        self.selected_file.set(path)
        try:
            metadata = self.metadata_reader.probe(path)
        except Exception as exc:
            self.status_text.set(f"Metadata probe failed: {exc}")
            messagebox.showerror("Metadata Error", str(exc))
            return
        self._render_metadata(metadata.to_dict())
        self.status_text.set(f"Loaded {Path(path).name}")

    def start_pipeline(self) -> None:
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("Pipeline Running", "A pipeline job is already running.")
            return
        input_file = self.selected_file.get().strip()
        if not input_file:
            messagebox.showwarning("Missing File", "Select a video file first.")
            return
        self.controller = JobController()
        self.worker = threading.Thread(target=self._run_pipeline, daemon=True)
        self.worker.start()
        self.status_text.set("Pipeline started")

    def _run_pipeline(self) -> None:
        try:
            result = self.orchestrator.run(
                request=type("Request", (), {
                    "input_file": self.selected_file.get().strip(),
                    "source_language": self.source_language.get(),
                    "target_language": self.target_language.get(),
                    "output_dir": None,
                })(),
                callback=self.event_queue.put,
                controller=self.controller,
            )
            self.event_queue.put({"kind": "result", "payload": result})
        except Exception as exc:
            self.event_queue.put({"kind": "error", "payload": str(exc)})

    def pause_pipeline(self) -> None:
        if self.controller:
            self.controller.pause()
            self.status_text.set("Pause requested")

    def resume_pipeline(self) -> None:
        if self.controller:
            self.controller.resume()
            self.status_text.set("Resume requested")

    def stop_pipeline(self) -> None:
        if self.controller:
            self.controller.stop()
            self.status_text.set("Stop requested")

    def _poll_queue(self) -> None:
        try:
            while True:
                item = self.event_queue.get_nowait()
                if isinstance(item, PipelineEvent):
                    self._handle_event(item)
                elif item.get("kind") == "error":
                    self.status_text.set(f"Pipeline failed: {item['payload']}")
                    messagebox.showerror("Pipeline Error", str(item["payload"]))
                elif item.get("kind") == "result":
                    self.status_text.set("Pipeline completed")
                    payload = item["payload"]
                    if isinstance(payload, dict) and payload.get("metadata"):
                        self._render_metadata(payload["metadata"])
        except queue.Empty:
            pass
        self.root.after(150, self._poll_queue)

    def _handle_event(self, event: PipelineEvent) -> None:
        self.progress["value"] = max(0, min(100, event.progress * 100))
        self.status_text.set(f"{event.stage}: {event.message}")
        self.log_text.insert(tk.END, f"[{event.status}] {event.stage}: {event.message}\n")
        self.log_text.see(tk.END)
        if event.payload.get("resources"):
            self.log_text.insert(tk.END, f"Resources: {event.payload['resources']}\n")
            self.log_text.see(tk.END)

    def _render_metadata(self, metadata: Dict[str, object]) -> None:
        self.metadata_text.delete("1.0", tk.END)
        for key, value in metadata.items():
            self.metadata_text.insert(tk.END, f"{key}: {value}\n")
