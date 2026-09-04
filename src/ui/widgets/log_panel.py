"""Collapsible log panel widget."""

from __future__ import annotations

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QTextEdit


class LogPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.toggle_button = QPushButton("Hide Logs")
        self.text = QTextEdit()
        self.text.setReadOnly(True)
        self.toggle_button.setAccessibleName("Toggle processing logs")
        self.toggle_button.setAccessibleDescription("Shows or hides the processing log output panel")
        self.text.setAccessibleName("Processing logs")
        self.text.setAccessibleDescription("Live processing logs with stage status and errors")
        self._visible = True

        self.toggle_button.clicked.connect(self.toggle)

        layout = QVBoxLayout()
        layout.addWidget(self.toggle_button)
        layout.addWidget(self.text)
        self.setLayout(layout)

    def toggle(self) -> None:
        self._visible = not self._visible
        self.text.setVisible(self._visible)
        self.toggle_button.setText("Hide Logs" if self._visible else "Show Logs")

    def append(self, message: str) -> None:
        self.text.append(message)
