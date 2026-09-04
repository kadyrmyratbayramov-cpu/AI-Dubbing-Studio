"""Desktop application entrypoint."""

from __future__ import annotations

import logging
import sys

from PyQt6.QtWidgets import QApplication

from src.config.settings import Config
from src.gui.main_window import MainWindow


def main() -> int:
    config = Config.load()
    logging.basicConfig(level=getattr(logging, config.logging_level.upper(), logging.INFO))
    app = QApplication(sys.argv)
    window = MainWindow(config)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
