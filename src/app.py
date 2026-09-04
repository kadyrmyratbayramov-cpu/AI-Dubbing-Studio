"""Backward-compatible app entrypoint."""

from src.main import main


if __name__ == "__main__":
    raise SystemExit(main())
