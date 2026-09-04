"""Desktop application entry point."""

from __future__ import annotations

import tkinter as tk

from src.config.settings import Config
from src.ui.main_window import MainWindow


def main() -> None:
    root = tk.Tk()
    root.title("AI Dubbing Studio")
    config = Config()
    MainWindow(root, config)
    root.mainloop()


if __name__ == "__main__":
    main()
