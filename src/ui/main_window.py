"""Main PyQt6 desktop UI for AI Dubbing Studio."""

from __future__ import annotations

import traceback
from pathlib import Path
from threading import Thread
from typing import Dict, Optional

from PyQt6.QtCore import QTimer
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

        self._build_ui()
        self._apply_style()

    def _build_ui(self) -> None:
        self.setWindowTitle("AI Dubbing Studio")
        self.resize(1200, 760)

        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)
        open_btn = QPushButton("Open Video")
        open_btn.clicked.connect(self.select_video)
        toolbar.addWidget(open_btn)

        preferences_btn = QPushButton("Preferences")
        preferences_btn.clicked.connect(self.open_preferences)
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

        self.progress_timer = QTimer(self)

    def _apply_style(self) -> None:
        style_path = Path(__file__).resolve().parent / "styles" / "dark.qss"
        if style_path.exists():
            self.setStyleSheet(style_path.read_text(encoding="utf-8"))

    def open_preferences(self) -> None:
        dialog = SettingsDialog(self.config, self)
        if dialog.exec():
            dialog.apply()
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

        self.current_video = file_path
        self.video_path.setText(file_path)
        self.statusBar().showMessage("Reading metadata...")

        try:
            metadata = self.orchestrator.ffmpeg.probe(file_path)
            self.meta_duration.setText(f"{metadata['duration']:.2f} s")
            self.meta_resolution.setText(str(metadata["resolution"]))
            self.meta_codec.setText(f"V:{metadata['video_codec']} | A:{metadata['audio_codec']}")
            self.meta_fps.setText(str(metadata["fps"]))
            self.statusBar().showMessage("Metadata loaded")
            self.log_panel.append(f"Loaded metadata for {file_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Metadata Error", str(exc))
            self.log_panel.append(f"Metadata error: {exc}")

    def _on_progress(self, stage: str, percent: float, message: str) -> None:
        pct = max(0, min(100, int(percent * 100)))

        def update_ui() -> None:
            self.progress.setValue(pct)
            self.stage_label.setText(f"{stage}: {message}")
            self.statusBar().showMessage(message)
            self.log_panel.append(f"[{stage}] {message}")

        QTimer.singleShot(0, update_ui)

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
                QTimer.singleShot(0, lambda: self._finish_success(result))
            except Exception as exc:
                trace = traceback.format_exc()
                QTimer.singleShot(0, lambda: self._finish_error(exc, trace))

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

    def _finish_error(self, exc: Exception, trace: str) -> None:
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.resume_btn.setEnabled(False)
        self.cancel_btn.setEnabled(False)
        self.stage_label.setText("Failed")
        self.log_panel.append(f"ERROR: {exc}")
        self.log_panel.append(trace)
        QMessageBox.critical(self, "Processing failed", str(exc))


def launch_app(config) -> int:
    app = QApplication.instance() or QApplication([])
    window = MainWindow(config)
    window.show()
    return app.exec()
