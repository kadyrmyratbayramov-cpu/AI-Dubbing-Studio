"""Settings/preferences dialog."""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
)


class SettingsDialog(QDialog):
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.setWindowTitle("Preferences")
        self.resize(420, 220)

        self.output_dir = QLineEdit(str(config.output_dir))
        self.jobs_dir = QLineEdit(str(config.jobs_dir))
        self.segment_seconds = QSpinBox()
        self.segment_seconds.setRange(10, 300)
        self.segment_seconds.setValue(config.segment_seconds)

        form = QFormLayout()
        form.addRow("Output directory", self.output_dir)
        form.addRow("Jobs directory", self.jobs_dir)
        form.addRow("Segment size (sec)", self.segment_seconds)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(buttons)
        self.setLayout(layout)

    def apply(self) -> None:
        self.config.output_dir = str(self.config.resolve_path(self.output_dir.text()))
        self.config.jobs_dir = str(self.config.resolve_path(self.jobs_dir.text()))
        self.config.segment_seconds = int(self.segment_seconds.value())
