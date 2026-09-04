"""Web frontend helpers and preview server."""

from __future__ import annotations

from html import escape
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional, Type

from app.factory import Application, create_application



def render_index_page(application: Optional[Application] = None) -> str:
    """Render the basic frontend page."""
    app = application or create_application()
    base_template = (app.templates_path / "base.html").read_text(encoding="utf-8")
    content = (app.static_path / "index.html").read_text(encoding="utf-8")
    placeholders = {
        "{{ title }}": escape(app.name),
        "{{ environment }}": escape(app.environment),
        "{{ version }}": escape(app.version),
        "{{ content }}": content,
    }
    rendered = base_template
    for placeholder, value in placeholders.items():
        rendered = rendered.replace(placeholder, value)
    return rendered


class WebRequestHandler(BaseHTTPRequestHandler):
    """Serve the rendered web scaffold."""

    application: Application

    def do_GET(self) -> None:  # noqa: N802
        if self.path not in {"/", "/index.html"}:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = render_index_page(self.application).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return



def build_web_handler(application: Application) -> Type[WebRequestHandler]:
    """Create a request handler bound to an application instance."""

    class Handler(WebRequestHandler):
        pass

    Handler.application = application
    return Handler



def create_web_server(
    host: str = "127.0.0.1",
    port: int = 8080,
    application: Optional[Application] = None,
) -> ThreadingHTTPServer:
    """Create a frontend preview server."""
    app = application or create_application()
    return ThreadingHTTPServer((host, port), build_web_handler(app))



def run_web_frontend(
    host: str = "127.0.0.1",
    port: int = 8080,
    application: Optional[Application] = None,
) -> None:
    """Run the frontend preview server."""
    server = create_web_server(host=host, port=port, application=application)
    try:
        server.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover
        pass
    finally:
        server.server_close()


__all__ = [
    "create_web_server",
    "render_index_page",
    "run_web_frontend",
]
