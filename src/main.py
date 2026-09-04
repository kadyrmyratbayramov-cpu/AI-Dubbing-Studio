"""Desktop application entrypoint."""

from __future__ import annotations

import sys

from src.config.settings import Config
from src.ui.main_window import launch_app


def main() -> int:
    config = Config()
    return launch_app(config)


if __name__ == "__main__":
    sys.exit(main())
