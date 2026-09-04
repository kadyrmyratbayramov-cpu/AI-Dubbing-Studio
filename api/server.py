"""Minimal backend API server for AI Dubbing Studio."""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional, Type
from urllib.parse import urlparse

from app.factory import Application, create_application
from web import render_index_page


class ApplicationRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler bound to an application instance."""

    application: Application

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._send_json(
                {
                    "status": "ok",
                    "application": self.application.name,
                    "version": self.application.version,
                    "environment": self.application.environment,
                }
            )
            return

        if parsed.path == "/config":
            self._send_json(self._public_config())
            return

        if parsed.path in {"/", "/index.html"}:
            html = render_index_page(self.application)
            self._send_response(
                HTTPStatus.OK,
                html.encode("utf-8"),
                "text/html; charset=utf-8",
            )
            return

        if parsed.path == "/static/index.html":
            content = (self.application.static_path / "index.html").read_text(
                encoding="utf-8"
            )
            self._send_response(
                HTTPStatus.OK,
                content.encode("utf-8"),
                "text/html; charset=utf-8",
            )
            return

        self._send_json({"detail": "Not Found"}, status=HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return

    def _public_config(self) -> dict:
        return {
            "application": self.application.name,
            "environment": self.application.environment,
            "debug": bool(self.application.config.debug),
            "audio": {
                "sample_rate": self.application.config.sample_rate,
                "channels": self.application.config.channels,
            },
            "defaults": {
                "voice": self.application.config.default_voice,
                "speaker": self.application.config.default_speaker,
            },
        }

    def _send_json(
        self,
        payload: dict,
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self._send_response(status, body, "application/json")

    def _send_response(
        self,
        status: HTTPStatus,
        body: bytes,
        content_type: str,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def build_handler(application: Application) -> Type[ApplicationRequestHandler]:
    """Create a request handler bound to an application instance."""

    class Handler(ApplicationRequestHandler):
        pass

    Handler.application = application
    return Handler


def create_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    application: Optional[Application] = None,
) -> ThreadingHTTPServer:
    """Create an API server instance."""
    app = application or create_application()
    return ThreadingHTTPServer((host, port), build_handler(app))


def run_api_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    application: Optional[Application] = None,
) -> None:
    """Run the API server until interrupted."""
    server = create_server(host=host, port=port, application=application)
    try:
        server.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover
        pass
    finally:
        server.server_close()


if __name__ == "__main__":  # pragma: no cover
    run_api_server()
