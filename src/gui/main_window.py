"""PyQt6 desktop UI for AI Dubbing Studio."""

from __future__ import annotations

import logging
import traceback
from pathlib import Path
from typing import Dict, Optional

from PyQt6.QtCore import QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
    QDialog,
    QDialogButtonBox,
)

from src.config.settings import Config
from src.core.dubbing_pipeline import DubbingPipeline
from src.core.types import PipelineStatus


class SettingsDialog(QDialog):
    def __init__(self, config: Config, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.config = config

        layout = QFormLayout(self)

        self.lang_pairs = QLineEdit(", ".join(self.config.default_language_pairs))
        self.output_dir = QLineEdit(self.config.output_dir)
        self.whisper_model = QComboBox()
        self.whisper_model.addItems(["tiny", "base", "small", "medium"])
        self.whisper_model.setCurrentText(self.config.whisper_model)

        self.tts_model = QLineEdit(self.config.tts_model)
        self.force_cpu = QCheckBox("Force CPU")
        self.force_cpu.setChecked(self.config.force_cpu)
        self.max_vram = QSpinBox()
        self.max_vram.setRange(1, 64)
        self.max_vram.setValue(int(self.config.max_vram_gb))
        self.logging_level = QComboBox()
        self.logging_level.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        self.logging_level.setCurrentText(self.config.logging_level)
        self.hf_token = QLineEdit(self.config.huggingface_token)
        self.hf_token.setEchoMode(QLineEdit.EchoMode.Password)

        layout.addRow("Default language pairs", self.lang_pairs)
        layout.addRow("Output directory", self.output_dir)
        layout.addRow("Whisper model", self.whisper_model)
        layout.addRow("TTS model", self.tts_model)
        layout.addRow("", self.force_cpu)
        layout.addRow("VRAM limit (GB)", self.max_vram)
        layout.addRow("Log level", self.logging_level)
        layout.addRow("HuggingFace token", self.hf_token)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def apply(self) -> None:
        pairs = [pair.strip() for pair in self.lang_pairs.text().split(",") if pair.strip()]
        self.config.default_language_pairs = pairs or ["en-es"]
        self.config.output_dir = self.output_dir.text().strip() or self.config.output_dir
        self.config.whisper_model = self.whisper_model.currentText()
        self.config.tts_model = self.tts_model.text().strip() or self.config.tts_model
        self.config.force_cpu = self.force_cpu.isChecked()
        self.config.max_vram_gb = float(self.max_vram.value())
        self.config.logging_level = self.logging_level.currentText()
        self.config.huggingface_token = self.hf_token.text().strip()
        self.config.normalize_paths()
        self.config.ensure_runtime_dirs()
        self.config.save("config/config.yaml")


class PipelineWorker(QThread):
    progress_signal = pyqtSignal(float, str, str)
    log_signal = pyqtSignal(str, str)
    finished_signal = pyqtSignal(dict)
    failed_signal = pyqtSignal(str)

    def __init__(self, pipeline: DubbingPipeline, input_file: str, output_file: str, src: str, tgt: str):
        super().__init__()
        self.pipeline = pipeline
        self.input_file = input_file
        self.output_file = output_file
        self.src = src
        self.tgt = tgt

    def run(self) -> None:
        try:
            result = self.pipeline.process(
                input_file=self.input_file,
                output_file=self.output_file,
                source_lang=self.src,
                target_lang=self.tgt,
                progress_callback=self._on_progress,
                log_callback=self._on_log,
            )
            self.finished_signal.emit(result)
        except Exception as exc:
            self.failed_signal.emit(f"{exc}\n{traceback.format_exc()}")

    def _on_progress(self, status: PipelineStatus) -> None:
        self.progress_signal.emit(status.progress, status.stage, status.message)

    def _on_log(self, severity: str, message: str) -> None:
        self.log_signal.emit(severity, message)


class MainWindow(QMainWindow):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.pipeline = DubbingPipeline(config)
        self.worker: Optional[PipelineWorker] = None
        self.last_request: Optional[Dict[str, str]] = None

        self.setWindowTitle("AI Dubbing Studio")
        self.resize(1280, 860)
        self._apply_dark_theme()
        self._build_ui()
        self.refresh_history()

        self.gpu_timer = QTimer(self)
        self.gpu_timer.timeout.connect(self.refresh_gpu_status)
        self.gpu_timer.start(1500)

    def _build_ui(self) -> None:
        self._build_menu()
        self._build_toolbar()
        self.setStatusBar(QStatusBar(self))

        root = QWidget(self)
        main_layout = QVBoxLayout(root)

        top_group = QGroupBox("Input & Language")
        top_layout = QFormLayout(top_group)
        file_row = QHBoxLayout()
        self.video_path = QLineEdit()
        browse = QPushButton("Browse")
        browse.clicked.connect(self.select_video)
        file_row.addWidget(self.video_path)
        file_row.addWidget(browse)

        self.lang_combo = QComboBox()
        for src, tgt in self.config.language_pairs():
            self.lang_combo.addItem(f"{src} → {tgt}", (src, tgt))
        if self.lang_combo.count() == 0:
            self.lang_combo.addItem("en → es", ("en", "es"))

        self.output_path = QLineEdit(str(Path(self.config.output_dir) / "dubbed_output.mp4"))
        out_browse = QPushButton("Output")
        out_browse.clicked.connect(self.select_output)
        out_row = QHBoxLayout()
        out_row.addWidget(self.output_path)
        out_row.addWidget(out_browse)

        top_layout.addRow("Video file", file_row)
        top_layout.addRow("Language pair", self.lang_combo)
        top_layout.addRow("Output file", out_row)

        metadata_group = QGroupBox("Video Metadata")
        meta_layout = QFormLayout(metadata_group)
        self.meta_duration = QLabel("-")
        self.meta_resolution = QLabel("-")
        self.meta_codec = QLabel("-")
        self.meta_fps = QLabel("-")
        self.meta_audio = QLabel("-")
        meta_layout.addRow("Duration", self.meta_duration)
        meta_layout.addRow("Resolution", self.meta_resolution)
        meta_layout.addRow("Codec", self.meta_codec)
        meta_layout.addRow("FPS", self.meta_fps)
        meta_layout.addRow("Audio Tracks", self.meta_audio)

        control_group = QGroupBox("Processing Controls")
        controls = QHBoxLayout(control_group)
        self.start_btn = QPushButton("Start")
        self.pause_btn = QPushButton("Pause")
        self.resume_btn = QPushButton("Resume")
        self.stop_btn = QPushButton("Stop")
        self.cancel_btn = QPushButton("Cancel")

        self.start_btn.clicked.connect(self.start_processing)
        self.pause_btn.clicked.connect(self.pipeline.pause)
        self.resume_btn.clicked.connect(self.pipeline.resume)
        self.stop_btn.clicked.connect(self.pipeline.stop)
        self.cancel_btn.clicked.connect(self.pipeline.cancel)

        for btn in (self.start_btn, self.pause_btn, self.resume_btn, self.stop_btn, self.cancel_btn):
            controls.addWidget(btn)

        self.progress_bar = QProgressBar()
        self.stage_label = QLabel("Stage: idle")
        self.eta_label = QLabel("ETA: -")

        info_row = QHBoxLayout()
        self.gpu_label = QLabel("Device: CPU")
        self.vram_label = QLabel("VRAM: 0/0 GB")
        self.lipsync_label = QLabel("Lip-sync: NOT AVAILABLE")
        info_row.addWidget(self.gpu_label)
        info_row.addWidget(self.vram_label)
        info_row.addWidget(self.lipsync_label)

        history_group = QGroupBox("Job History")
        history_layout = QVBoxLayout(history_group)
        self.history_list = QListWidget()
        history_layout.addWidget(self.history_list)

        log_group = QGroupBox("Logs")
        log_layout = QVBoxLayout(log_group)
        filter_row = QHBoxLayout()
        self.log_filter = QComboBox()
        self.log_filter.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        self.log_filter.setCurrentText("INFO")
        filter_row.addWidget(QLabel("Min severity"))
        filter_row.addWidget(self.log_filter)
        self.log_panel = QTextEdit()
        self.log_panel.setReadOnly(True)
        log_layout.addLayout(filter_row)
        log_layout.addWidget(self.log_panel)

        top_split = QSplitter()
        top_left = QWidget()
        top_left_layout = QVBoxLayout(top_left)
        top_left_layout.addWidget(top_group)
        top_left_layout.addWidget(metadata_group)
        top_left_layout.addWidget(control_group)
        top_left_layout.addWidget(self.progress_bar)
        top_left_layout.addWidget(self.stage_label)
        top_left_layout.addWidget(self.eta_label)
        top_left_layout.addLayout(info_row)

        top_split.addWidget(top_left)
        top_split.addWidget(history_group)

        bottom_split = QSplitter()
        bottom_split.addWidget(top_split)
        bottom_split.addWidget(log_group)

        main_layout.addWidget(bottom_split)
        self.setCentralWidget(root)

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("File")
        open_action = QAction("Open Video", self)
        open_action.triggered.connect(self.select_video)
        file_menu.addAction(open_action)

        settings_action = QAction("Settings", self)
        settings_action.triggered.connect(self.open_settings)
        file_menu.addAction(settings_action)

        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Main", self)
        self.addToolBar(toolbar)

        start_action = QAction("Start", self)
        start_action.triggered.connect(self.start_processing)
        toolbar.addAction(start_action)

        pause_action = QAction("Pause", self)
        pause_action.triggered.connect(self.pipeline.pause)
        toolbar.addAction(pause_action)

        resume_action = QAction("Resume", self)
        resume_action.triggered.connect(self.pipeline.resume)
        toolbar.addAction(resume_action)

        stop_action = QAction("Stop", self)
        stop_action.triggered.connect(self.pipeline.stop)
        toolbar.addAction(stop_action)

        cancel_action = QAction("Cancel", self)
        cancel_action.triggered.connect(self.pipeline.cancel)
        toolbar.addAction(cancel_action)

    def _apply_dark_theme(self) -> None:
        self.setStyleSheet(
            """
            QWidget { background-color: #1e1e1e; color: #e0e0e0; }
            QLineEdit, QTextEdit, QListWidget, QComboBox, QSpinBox { background-color: #2a2a2a; color: #f0f0f0; border: 1px solid #3b3b3b; }
            QPushButton { background-color: #333333; border: 1px solid #555555; padding: 6px; }
            QPushButton:hover { background-color: #3d3d3d; }
            QProgressBar { border: 1px solid #555; text-align: center; }
            QProgressBar::chunk { background-color: #3a86ff; }
            """
        )

    def select_video(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video",
            "",
            "Video Files (*.mp4 *.mkv *.mov *.avi *.m4v *.webm)",
        )
        if not selected:
            return
        self.video_path.setText(selected)
        if not self.output_path.text().strip():
            self.output_path.setText(str(Path(self.config.output_dir) / (Path(selected).stem + "_dubbed.mp4")))

        try:
            metadata = self.pipeline.get_ffmpeg().probe_video(selected)
            self.meta_duration.setText(f"{metadata.duration:.2f}s")
            self.meta_resolution.setText(f"{metadata.width}x{metadata.height}")
            self.meta_codec.setText(f"V:{metadata.video_codec} A:{metadata.audio_codec}")
            self.meta_fps.setText(f"{metadata.fps:.2f}")
            self.meta_audio.setText(
                ", ".join(
                    f"#{s.index} {s.codec} {s.channels}ch {s.language}" for s in metadata.audio_streams
                )
                or "None"
            )
        except Exception as exc:
            self._show_error(f"Metadata extraction failed: {exc}")

    def select_output(self) -> None:
        selected, _ = QFileDialog.getSaveFileName(self, "Save Dubbed Video", self.output_path.text(), "MP4 Files (*.mp4)")
        if selected:
            self.output_path.setText(selected)

    def start_processing(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self._show_error("A job is already running. Please wait for it to finish.")
            return

        video = self.video_path.text().strip()
        output = self.output_path.text().strip()
        if not video:
            self._show_error("Please select a video file.")
            return

        src, tgt = self.lang_combo.currentData()
        self.last_request = {"video": video, "output": output, "src": src, "tgt": tgt}
        self.progress_bar.setValue(0)
        self.stage_label.setText("Stage: starting")

        self.worker = PipelineWorker(self.pipeline, video, output, src, tgt)
        self.worker.progress_signal.connect(self.on_progress)
        self.worker.log_signal.connect(self.append_log)
        self.worker.finished_signal.connect(self.on_complete)
        self.worker.failed_signal.connect(self.on_failed)
        self.start_btn.setEnabled(False)
        self.worker.start()

    def on_progress(self, progress: float, stage: str, message: str) -> None:
        self.progress_bar.setValue(int(progress))
        self.stage_label.setText(f"Stage: {stage}")
        if message:
            self.statusBar().showMessage(message, 3000)
        self.eta_label.setText(f"Progress: {progress:.1f}%")

    def append_log(self, severity: str, message: str) -> None:
        order = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}
        if order.get(severity, 20) < order.get(self.log_filter.currentText(), 20):
            return
        self.log_panel.append(f"[{severity}] {message}")

    def on_complete(self, result: dict) -> None:
        self.start_btn.setEnabled(True)
        self.append_log("INFO", f"Completed job {result.get('job_id')}")
        self.refresh_history()
        self.stage_label.setText("Stage: done")
        self.progress_bar.setValue(100)
        self.statusBar().showMessage("Dubbing completed", 5000)

    def on_failed(self, error: str) -> None:
        self.start_btn.setEnabled(True)
        self.append_log("ERROR", error)
        self.refresh_history()
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Critical)
        msg.setWindowTitle("Processing Failed")
        msg.setText("Dubbing failed.")
        msg.setInformativeText("Retry with the same job inputs?")
        retry_btn = msg.addButton("Retry", QMessageBox.ButtonRole.AcceptRole)
        msg.addButton("Close", QMessageBox.ButtonRole.RejectRole)
        msg.exec()
        if msg.clickedButton() == retry_btn and self.last_request:
            self.video_path.setText(self.last_request["video"])
            self.output_path.setText(self.last_request["output"])
            self.start_processing()

    def refresh_gpu_status(self) -> None:
        status = self.pipeline.get_gpu_status()
        if status["available"]:
            self.gpu_label.setText(f"Device: {status['device_name']}")
            self.vram_label.setText(f"VRAM: {status['used_vram_gb']:.2f}/{status['total_vram_gb']:.2f} GB")
        else:
            self.gpu_label.setText("Device: CPU")
            self.vram_label.setText("VRAM: N/A")

    def refresh_history(self) -> None:
        self.history_list.clear()
        for entry in self.pipeline.get_job_history():
            status = entry.get("status", "unknown")
            label = f"{entry.get('job_id')} | {status} | {entry.get('source_lang')}→{entry.get('target_lang')}"
            self.history_list.addItem(label)

    def open_settings(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self._show_error("Cannot change settings while a job is running.")
            return
        dialog = SettingsDialog(self.config, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            current_pair = self.lang_combo.currentData()
            dialog.apply()
            self.pipeline = DubbingPipeline(self.config)
            self.lang_combo.clear()
            restored_index = -1
            for src, tgt in self.config.language_pairs():
                self.lang_combo.addItem(f"{src} → {tgt}", (src, tgt))
                if current_pair == (src, tgt):
                    restored_index = self.lang_combo.count() - 1
            if self.lang_combo.count() == 0:
                self.lang_combo.addItem("en → es", ("en", "es"))
                self.lang_combo.setCurrentIndex(0)
            elif restored_index >= 0:
                self.lang_combo.setCurrentIndex(restored_index)
            else:
                self.lang_combo.setCurrentIndex(0)

    def _show_error(self, text: str) -> None:
        logging.error(text)
        QMessageBox.critical(self, "Error", text)
