"""CLI entry points for AI Dubbing Studio."""

from __future__ import annotations

import argparse
import json
from typing import Optional, Sequence

from api.server import run_api_server
from app.factory import create_application
from web import run_web_frontend


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        description="AI Dubbing Studio application runner"
    )
    parser.add_argument(
        "--config",
        help="Path to a YAML or JSON configuration file",
    )
    parser.add_argument(
        "--environment",
        default=None,
        help="Runtime environment name",
    )

    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = False

    subparsers.add_parser("info", help="Print runtime metadata")
    subparsers.add_parser(
        "check",
        help="Initialize the application and verify setup",
    )

    api_parser = subparsers.add_parser(
        "serve-api",
        help="Start the API server",
    )
    api_parser.add_argument("--host", default=None)
    api_parser.add_argument("--port", type=int, default=None)

    web_parser = subparsers.add_parser(
        "serve-web",
        help="Start the web preview server",
    )
    web_parser.add_argument("--host", default=None)
    web_parser.add_argument("--port", type=int, default=None)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command or "info"
    application = create_application(
        config_path=args.config,
        environment=args.environment,
    )

    if command == "info":
        print(json.dumps(application.to_metadata(), indent=2, sort_keys=True))
        return 0

    if command == "check":
        print(
            f"Initialized {application.name} {application.version} in "
            f"{application.environment} mode"
        )
        return 0

    if command == "serve-api":
        host = args.host or str(
            application.runtime_settings.get("host", "127.0.0.1")
        )
        port = int(
            args.port or application.runtime_settings.get("api_port", 8000)
        )
        run_api_server(host=host, port=port, application=application)
        return 0

    if command == "serve-web":
        host = args.host or str(
            application.runtime_settings.get("host", "127.0.0.1")
        )
        port = int(
            args.port or application.runtime_settings.get("web_port", 8080)
        )
        run_web_frontend(host=host, port=port, application=application)
        return 0

    parser.error(f"Unsupported command: {command}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
