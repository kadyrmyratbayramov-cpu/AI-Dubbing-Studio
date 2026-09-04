"""Main PyQt6 desktop UI for AI Dubbing Studio."""

from __future__ import annotations

import traceback
from pathlib import Path
from threading import Thread
from typing import Dict, Optional

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QComboBox,
    QProgressBar,
    QStatusBar,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from src.pipeline.orchestrator import DubbingOrchestrator
from src.ui.dialogs.settings_dialog import SettingsDialog
from src.ui.widgets.log_panel import LogPanel

LANGUAGES = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Russian": "ru",
    "Chinese": "zh",
    "Japanese": "ja",
    "Arabic": "ar",
}


class MainWindow(QMainWindow):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.orchestrator = DubbingOrchestrator(config)
        self.current_video: Optional[str] = None
        self.current_result: Optional[Dict[str, object]] = None
        self.worker: Optional[Thread] = None
        self.metadata_worker: Optional[Thread] = None
        self._metadata_request_id = 0
        self.signals = _MainWindowSignals()
        self.signals.progress.connect(self._handle_progress)
        self.signals.metadata_success.connect(self._apply_metadata_success)
        self.signals.metadata_error.connect(self._apply_metadata_error)
        self.signals.processing_success.connect(self._finish_success)
        self.signals.processing_error.connect(self._finish_error)

        self._build_ui()
        self._apply_style()

    def _build_ui(self) -> None:
        self.setWindowTitle("AI Dubbing Studio")
        self.resize(1200, 760)

        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)
        open_btn = QPushButton("Open Video")
        open_btn.clicked.connect(self.select_video)
        open_btn.setAccessibleName("Open video file")
        open_btn.setAccessibleDescription("Open a file picker to choose input video")
        toolbar.addWidget(open_btn)

        preferences_btn = QPushButton("Preferences")
        preferences_btn.clicked.connect(self.open_preferences)
        preferences_btn.setAccessibleName("Open preferences")
        preferences_btn.setAccessibleDescription("Open settings dialog for output and pipeline options")
        toolbar.addWidget(preferences_btn)

        central = QWidget()
        root = QVBoxLayout()

        video_group = QGroupBox("Video Selection")
        video_layout = QGridLayout()
        self.video_path = QLineEdit()
        self.video_path.setReadOnly(True)
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.select_video)
        video_layout.addWidget(QLabel("Input video"), 0, 0)
        video_layout.addWidget(self.video_path, 0, 1)
        video_layout.addWidget(browse_btn, 0, 2)
        video_group.setLayout(video_layout)

        lang_group = QGroupBox("Language Pair")
        lang_layout = QFormLayout()
        self.source_lang = QComboBox()
        self.target_lang = QComboBox()
        self.source_lang.setAccessibleName("Source language")
        self.source_lang.setAccessibleDescription("Language spoken in the original video audio")
        self.target_lang.setAccessibleName("Target language")
        self.target_lang.setAccessibleDescription("Language for generated dubbed audio")
        for label, code in LANGUAGES.items():
            self.source_lang.addItem(label, code)
            self.target_lang.addItem(label, code)
        self.target_lang.setCurrentText("Spanish")
        lang_layout.addRow("Source", self.source_lang)
        lang_layout.addRow("Target", self.target_lang)
        lang_group.setLayout(lang_layout)

        meta_group = QGroupBox("Video Metadata")
        meta_layout = QFormLayout()
        self.meta_duration = QLabel("-")
        self.meta_resolution = QLabel("-")
        self.meta_codec = QLabel("-")
        self.meta_fps = QLabel("-")
        meta_layout.addRow("Duration", self.meta_duration)
        meta_layout.addRow("Resolution", self.meta_resolution)
        meta_layout.addRow("Codecs", self.meta_codec)
        meta_layout.addRow("FPS", self.meta_fps)
        meta_group.setLayout(meta_layout)

        control_group = QGroupBox("Processing Controls")
        control_layout = QGridLayout()
        self.start_btn = QPushButton("Start")
        self.pause_btn = QPushButton("Pause")
        self.resume_btn = QPushButton("Resume")
        self.cancel_btn = QPushButton("Cancel")
        self.start_btn.clicked.connect(self.start_processing)
        self.pause_btn.clicked.connect(self.pause_processing)
        self.resume_btn.clicked.connect(self.resume_processing)
        self.cancel_btn.clicked.connect(self.cancel_processing)
        self.pause_btn.setEnabled(False)
        self.resume_btn.setEnabled(False)
        self.cancel_btn.setEnabled(False)
        control_layout.addWidget(self.start_btn, 0, 0)
        control_layout.addWidget(self.pause_btn, 0, 1)
        control_layout.addWidget(self.resume_btn, 0, 2)
        control_layout.addWidget(self.cancel_btn, 0, 3)
        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.stage_label = QLabel("Idle")
        control_layout.addWidget(self.progress, 1, 0, 1, 4)
        control_layout.addWidget(self.stage_label, 2, 0, 1, 4)
        control_group.setLayout(control_layout)

        preview_group = QGroupBox("Output Preview")
        preview_layout = QVBoxLayout()
        self.output_preview = QTextEdit()
        self.output_preview.setReadOnly(True)
        preview_layout.addWidget(self.output_preview)
        preview_group.setLayout(preview_layout)

        self.log_panel = LogPanel()

        root.addWidget(video_group)
        root.addWidget(lang_group)
        root.addWidget(meta_group)
        root.addWidget(control_group)
        root.addWidget(preview_group)
        root.addWidget(self.log_panel)

        central.setLayout(root)
        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar())

    def _apply_style(self) -> None:
        style_path = Path(__file__).resolve().parent / "styles" / "dark.qss"
        if style_path.exists():
            self.setStyleSheet(style_path.read_text(encoding="utf-8"))

    def open_preferences(self) -> None:
        if self.worker and self.worker.is_alive():
            QMessageBox.information(self, "Busy", "Wait for current processing job to finish before editing preferences.")
            return
        dialog = SettingsDialog(self.config, self)
        if dialog.exec():
            dialog.apply()
            self.config._normalize_and_create_paths()
            self.orchestrator = DubbingOrchestrator(self.config)
            self.log_panel.append("Preferences updated")

    def select_video(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select video",
            str(Path.home()),
            "Video Files (*.mp4 *.mkv *.mov *.avi *.webm)",
        )
        if not file_path:
            return

        self.video_path.setText(file_path)
        self.statusBar().showMessage("Reading metadata...")
        self.current_video = None
        self._metadata_request_id += 1
        request_id = self._metadata_request_id

        def worker() -> None:
            try:
                metadata = self.orchestrator.ffmpeg.probe(file_path)
                self.signals.metadata_success.emit(file_path, {"request_id": request_id, "metadata": metadata})
            except Exception as exc:
                self.signals.metadata_error.emit(file_path, {"request_id": request_id, "error": str(exc)})

        self.metadata_worker = Thread(target=worker, daemon=True)
        self.metadata_worker.start()

    def _apply_metadata_success(self, file_path: str, payload: Dict[str, object]) -> None:
        request_id = int(payload.get("request_id", -1))
        if request_id != self._metadata_request_id or self.video_path.text() != file_path:
            return
        metadata = payload.get("metadata", {}) if isinstance(payload.get("metadata", {}), dict) else {}
        self.current_video = file_path
        self.meta_duration.setText(f"{float(metadata.get('duration', 0.0)):.2f} s")
        self.meta_resolution.setText(str(metadata.get("resolution", "unknown")))
        self.meta_codec.setText(f"V:{metadata.get('video_codec', 'unknown')} | A:{metadata.get('audio_codec', 'unknown')}")
        self.meta_fps.setText(str(metadata.get("fps", "0.0")))
        self.statusBar().showMessage("Metadata loaded")
        self.log_panel.append(f"Loaded metadata for {file_path}")

    def _apply_metadata_error(self, file_path: str, payload: Dict[str, object]) -> None:
        request_id = int(payload.get("request_id", -1))
        if request_id != self._metadata_request_id or self.video_path.text() != file_path:
            return
        message = str(payload.get("error", "unknown error"))
        self.current_video = None
        self.video_path.clear()
        self.meta_duration.setText("-")
        self.meta_resolution.setText("-")
        self.meta_codec.setText("-")
        self.meta_fps.setText("-")
        error_message = f"Failed to probe metadata for '{file_path}': {message}"
        QMessageBox.critical(self, "Metadata Error", error_message)
        self.log_panel.append(error_message)

    def _on_progress(self, stage: str, percent: float, message: str) -> None:
        self.signals.progress.emit(stage, percent, message)

    def _handle_progress(self, stage: str, percent: float, message: str) -> None:
        pct = max(0, min(100, int(percent * 100)))
        self.progress.setValue(pct)
        self.stage_label.setText(f"{stage}: {message}")
        self.statusBar().showMessage(message)
        self.log_panel.append(f"[{stage}] {message}")

    def start_processing(self) -> None:
        if not self.current_video:
            QMessageBox.warning(self, "Missing input", "Please choose a video file first.")
            return

        source = self.source_lang.currentData()
        target = self.target_lang.currentData()

        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.cancel_btn.setEnabled(True)
        self.resume_btn.setEnabled(False)
        self.progress.setValue(0)

        def worker() -> None:
            try:
                result = self.orchestrator.process(self.current_video, source, target, progress=self._on_progress)
                self.current_result = result
                self.signals.processing_success.emit(result)
            except Exception as exc:
                trace = traceback.format_exc()
                self.signals.processing_error.emit(str(exc), trace)

        self.worker = Thread(target=worker, daemon=True)
        self.worker.start()

    def pause_processing(self) -> None:
        self.orchestrator.pause()
        self.pause_btn.setEnabled(False)
        self.resume_btn.setEnabled(True)
        self.statusBar().showMessage("Paused")

    def resume_processing(self) -> None:
        self.orchestrator.resume()
        self.pause_btn.setEnabled(True)
        self.resume_btn.setEnabled(False)
        self.statusBar().showMessage("Resumed")

    def cancel_processing(self) -> None:
        self.orchestrator.cancel()
        self.pause_btn.setEnabled(False)
        self.resume_btn.setEnabled(False)
        self.cancel_btn.setEnabled(False)
        self.statusBar().showMessage("Cancelling...")

    def _finish_success(self, result: Dict[str, object]) -> None:
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.resume_btn.setEnabled(False)
        self.cancel_btn.setEnabled(False)
        self.progress.setValue(100)
        self.stage_label.setText("Completed")
        self.output_preview.setText(
            "\n".join(
                [
                    f"Job ID: {result.get('job_id')}",
                    f"Status: {result.get('status')}",
                    f"Output video: {result.get('output_video')}",
                    f"QC passed: {result.get('qc', {}).get('passed')}",
                ]
            )
        )
        self.log_panel.append("Processing completed successfully")

    def _finish_error(self, error_message: str, trace: str) -> None:
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.resume_btn.setEnabled(False)
        self.cancel_btn.setEnabled(False)
        self.stage_label.setText("Failed")
        self.log_panel.append(f"ERROR: {error_message}")
        self.log_panel.append(trace)
        QMessageBox.critical(self, "Processing failed", error_message)


def launch_app(config) -> int:
    app = QApplication.instance() or QApplication([])
    window = MainWindow(config)
    window.show()
    return app.exec()


class _MainWindowSignals(QObject):
    progress = pyqtSignal(str, float, str)
    metadata_success = pyqtSignal(str, object)
    metadata_error = pyqtSignal(str, object)
    processing_success = pyqtSignal(object)
    processing_error = pyqtSignal(str, str)
