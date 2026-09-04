"""Tkinter desktop UI for AI Dubbing Studio."""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Dict, Optional, Union

from src.config.settings import Config
from src.core.job_state import JobController, PipelineEvent, PipelineRequest
from src.core.orchestrator import DubbingOrchestrator
from src.core.video_metadata import VideoMetadataReader
from src.utils.validators import validate_input_file

QueueItem = Union[PipelineEvent, Dict[str, object]]


class MainWindow:
    def __init__(self, root: tk.Tk, config: Config):
        self.root = root
        self.config = config
        self.orchestrator = DubbingOrchestrator(config)
        self.metadata_reader = VideoMetadataReader(config)
        self.event_queue: "queue.Queue[QueueItem]" = queue.Queue()
        self.controller: Optional[JobController] = None
        self.worker: Optional[threading.Thread] = None
        self.poller_id: Optional[str] = None
        self.is_closing = False
        self.latest_probe_path: Optional[str] = None
        self.source_combobox: Optional[ttk.Combobox] = None
        self.target_combobox: Optional[ttk.Combobox] = None

        self.selected_file = tk.StringVar()
        self.source_language = tk.StringVar(value=config.source_language)
        self.target_language = tk.StringVar(value=config.target_language)
        self.status_text = tk.StringVar(value="Ready")
        self.progress_text = tk.StringVar(value="Progress: 0%")

        self._build()
        self.poller_id = self.root.after(150, self._poll_queue)
        self.root.bind_all("<Alt-s>", self._focus_source_language)
        self.root.bind_all("<Alt-t>", self._focus_target_language)
        self.root.bind_all("<Control-Return>", self._start_shortcut)
        self.root.bind_all("<Control-p>", self._pause_shortcut)
        self.root.bind_all("<Control-r>", self._resume_shortcut)
        self.root.bind_all("<Control-q>", self._stop_shortcut)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build(self) -> None:
        self.root.geometry("980x720")
        self.root.minsize(880, 640)
        main = ttk.Frame(self.root, padding=16)
        main.pack(fill=tk.BOTH, expand=True)

        file_frame = ttk.LabelFrame(main, text="Video Input", padding=12)
        file_frame.pack(fill=tk.X)
        entry = ttk.Entry(file_frame, textvariable=self.selected_file)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        ttk.Button(
            file_frame,
            text="Select Video",
            command=self.select_video,
        ).pack(side=tk.LEFT)

        language_frame = ttk.LabelFrame(main, text="Languages", padding=12)
        language_frame.pack(fill=tk.X, pady=12)
        options = [entry["code"] for entry in self.config.available_languages]
        ttk.Label(language_frame, text="Source (Alt+S)").grid(
            row=0,
            column=0,
            sticky=tk.W,
        )
        self.source_combobox = ttk.Combobox(
            language_frame,
            values=options,
            textvariable=self.source_language,
            state="readonly",
            width=18,
        )
        self.source_combobox.grid(row=0, column=1, padx=(8, 20), sticky=tk.W)
        ttk.Label(language_frame, text="Target (Alt+T)").grid(
            row=0,
            column=2,
            sticky=tk.W,
        )
        self.target_combobox = ttk.Combobox(
            language_frame,
            values=options[1:],
            textvariable=self.target_language,
            state="readonly",
            width=18,
        )
        self.target_combobox.grid(row=0, column=3, padx=(8, 0), sticky=tk.W)

        control_frame = ttk.LabelFrame(main, text="Controls", padding=12)
        control_frame.pack(fill=tk.X)
        ttk.Button(
            control_frame,
            text="Start (Ctrl+Enter)",
            command=self.start_pipeline,
        ).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(
            control_frame,
            text="Pause (Ctrl+P)",
            command=self.pause_pipeline,
        ).pack(side=tk.LEFT, padx=8)
        ttk.Button(
            control_frame,
            text="Resume (Ctrl+R)",
            command=self.resume_pipeline,
        ).pack(side=tk.LEFT, padx=8)
        ttk.Button(
            control_frame,
            text="Stop (Ctrl+Q)",
            command=self.stop_pipeline,
        ).pack(side=tk.LEFT, padx=8)

        status_frame = ttk.LabelFrame(main, text="Status", padding=12)
        status_frame.pack(fill=tk.X, pady=12)
        ttk.Label(status_frame, textvariable=self.status_text).pack(anchor=tk.W)
        ttk.Label(status_frame, text="Pipeline progress").pack(anchor=tk.W)
        ttk.Label(status_frame, textvariable=self.progress_text).pack(anchor=tk.W)
        self.progress = ttk.Progressbar(
            status_frame,
            mode="determinate",
            maximum=100,
        )
        self.progress.pack(fill=tk.X, pady=(8, 0))

        metadata_frame = ttk.LabelFrame(main, text="Metadata", padding=12)
        metadata_frame.pack(fill=tk.X)
        metadata_container = ttk.Frame(metadata_frame)
        metadata_container.pack(fill=tk.BOTH, expand=True)
        metadata_scroll = ttk.Scrollbar(metadata_container, orient=tk.VERTICAL)
        metadata_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.metadata_text = tk.Text(
            metadata_container,
            height=8,
            wrap=tk.WORD,
            yscrollcommand=metadata_scroll.set,
            state=tk.DISABLED,
        )
        self.metadata_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        metadata_scroll.config(command=self.metadata_text.yview)

        log_frame = ttk.LabelFrame(main, text="Pipeline Events", padding=12)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(12, 0))
        log_container = ttk.Frame(log_frame)
        log_container.pack(fill=tk.BOTH, expand=True)
        log_scroll = ttk.Scrollbar(log_container, orient=tk.VERTICAL)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text = tk.Text(
            log_container,
            height=16,
            wrap=tk.WORD,
            yscrollcommand=log_scroll.set,
            state=tk.DISABLED,
        )
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scroll.config(command=self.log_text.yview)

    def select_video(self) -> None:
        path = filedialog.askopenfilename(
            title="Select a video file",
            filetypes=[
                ("Video files", "*.mp4 *.mov *.mkv *.avi *.m4v *.webm"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        self.selected_file.set(path)
        self.latest_probe_path = path
        self.status_text.set(f"Probing metadata for {Path(path).name}...")
        threading.Thread(
            target=self._probe_metadata,
            args=(path,),
            daemon=True,
        ).start()

    def _probe_metadata(self, path: str) -> None:
        try:
            metadata = self.metadata_reader.probe(path)
        except Exception as exc:
            self._enqueue(
                {"kind": "metadata_error", "payload": str(exc), "path": path}
            )
            return
        self._enqueue(
            {"kind": "metadata", "payload": metadata.to_dict(), "path": path}
        )

    def start_pipeline(self) -> None:
        if self.worker and self.worker.is_alive():
            messagebox.showinfo(
                "Pipeline Running",
                "A pipeline job is already running.",
            )
            return
        input_file = self.selected_file.get().strip()
        try:
            validate_input_file(input_file)
        except Exception as exc:
            messagebox.showerror("Invalid Input", str(exc))
            return
        request = PipelineRequest(
            input_file=input_file,
            source_language=self.source_language.get(),
            target_language=self.target_language.get(),
        )
        self.controller = JobController()
        self.worker = threading.Thread(
            target=self._run_pipeline,
            args=(request,),
            daemon=True,
        )
        self.worker.start()
        self.status_text.set("Pipeline started")
        self.progress_text.set("Progress: 0%")

    def _run_pipeline(self, request: PipelineRequest) -> None:
        try:
            result = self.orchestrator.run(
                request=request,
                callback=self._enqueue,
                controller=self.controller,
            )
            self._enqueue({"kind": "result", "payload": result})
        except Exception as exc:
            self._enqueue({"kind": "error", "payload": str(exc)})

    def _enqueue(self, item: QueueItem) -> None:
        if not self.is_closing:
            self.event_queue.put(item)

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

    def _focus_source_language(self, _event=None):
        if self.source_combobox:
            self.source_combobox.focus_set()
        return "break"

    def _focus_target_language(self, _event=None):
        if self.target_combobox:
            self.target_combobox.focus_set()
        return "break"

    def _start_shortcut(self, _event=None):
        self.start_pipeline()
        return "break"

    def _pause_shortcut(self, _event=None):
        self.pause_pipeline()
        return "break"

    def _resume_shortcut(self, _event=None):
        self.resume_pipeline()
        return "break"

    def _stop_shortcut(self, _event=None):
        self.stop_pipeline()
        return "break"

    def _on_close(self) -> None:
        self.is_closing = True
        if self.controller:
            self.controller.stop()
        if self.worker and self.worker.is_alive():
            self.worker.join(timeout=1)
        if self.poller_id is not None:
            try:
                self.root.after_cancel(self.poller_id)
            except tk.TclError:
                pass
            self.poller_id = None
        self.root.destroy()

    def _poll_queue(self) -> None:
        try:
            while True:
                item = self.event_queue.get_nowait()
                if isinstance(item, PipelineEvent):
                    self._handle_event(item)
                elif isinstance(item, dict) and item.get("kind") == "metadata":
                    if item.get("path") != self.latest_probe_path:
                        continue
                    payload = item["payload"]
                    if isinstance(payload, dict):
                        self._render_metadata(payload)
                        self.status_text.set(
                            f"Loaded {Path(str(item['path'])).name}"
                        )
                elif isinstance(item, dict) and item.get("kind") == "metadata_error":
                    if item.get("path") != self.latest_probe_path:
                        continue
                    self.status_text.set(f"Metadata probe failed: {item['payload']}")
                    messagebox.showerror("Metadata Error", str(item["payload"]))
                elif isinstance(item, dict) and item.get("kind") == "error":
                    self.status_text.set(f"Pipeline failed: {item['payload']}")
                    messagebox.showerror("Pipeline Error", str(item["payload"]))
                elif isinstance(item, dict) and item.get("kind") == "result":
                    self.status_text.set("Pipeline completed")
                    self.progress["value"] = 100
                    self.progress_text.set("Progress: 100%")
                    payload = item["payload"]
                    if isinstance(payload, dict) and payload.get("metadata"):
                        self._render_metadata(payload["metadata"])
        except queue.Empty:
            pass
        if self.root.winfo_exists() and not self.is_closing:
            self.poller_id = self.root.after(150, self._poll_queue)

    def _handle_event(self, event: PipelineEvent) -> None:
        progress_value = max(0, min(100, event.progress * 100))
        self.progress["value"] = progress_value
        self.progress_text.set(f"Progress: {progress_value:.0f}%")
        self.status_text.set(f"{event.stage}: {event.message}")
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(
            tk.END,
            f"[{event.status}] {event.stage}: {event.message}\n",
        )
        resources = event.payload.get("resources")
        if resources:
            self.log_text.insert(tk.END, f"Resources: {resources}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def _render_metadata(self, metadata: Dict[str, object]) -> None:
        self.metadata_text.config(state=tk.NORMAL)
        self.metadata_text.delete("1.0", tk.END)
        for key, value in metadata.items():
            self.metadata_text.insert(tk.END, f"{key}: {value}\n")
        self.metadata_text.config(state=tk.DISABLED)
