"""Tests for the runnable application foundation."""

from __future__ import annotations

import json
from http.client import HTTPConnection
from threading import Thread

from api.server import create_server
from app.factory import Application, create_application
from cli.main import main as cli_main
from src.config.settings import Config
from web import create_web_server, render_index_page


def test_create_application_loads_runtime_metadata():
    application = create_application()

    assert application.name == "AI Dubbing Studio"
    assert application.environment == "development"
    assert application.config.sample_rate == 22050
    assert application.runtime_settings["host"] == "127.0.0.1"
    assert application.output_path.exists()
    assert application.log_path.exists()


def test_render_index_page_contains_expected_sections():
    html = render_index_page(create_application())

    assert "AI Dubbing Studio" in html
    assert "serve-api" in html
    assert "Runnable scaffold only" in html


def test_render_index_page_escapes_metadata():
    application = Application(
        config=Config(),
        environment="dev<script>",
        root_path=create_application().root_path,
        config_path=None,
        runtime_settings={},
    )

    html = render_index_page(application)
    assert "dev&lt;script&gt;" in html
    assert "dev<script>" not in html


def test_cli_info_outputs_json(capsys):
    exit_code = cli_main(["info"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["name"] == "AI Dubbing Studio"


def test_cli_check_reports_success(capsys):
    exit_code = cli_main(["check"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Initialized AI Dubbing Studio" in captured.out


def test_api_server_health_endpoint():
    application = create_application()
    server = create_server(host="127.0.0.1", port=0, application=application)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        connection = HTTPConnection(
            "127.0.0.1",
            server.server_address[1],
            timeout=5,
        )
        connection.request("GET", "/health")
        response = connection.getresponse()
        body = response.read().decode("utf-8")
        payload = json.loads(body)
        assert response.status == 200
        assert payload["status"] == "ok"
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_web_server_renders_frontend_routes():
    application = create_application()
    server = create_web_server(
        host="127.0.0.1",
        port=0,
        application=application,
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        connection = HTTPConnection(
            "127.0.0.1",
            server.server_address[1],
            timeout=5,
        )
        connection.request("GET", "/index.html?refresh=1")
        response = connection.getresponse()
        body = response.read().decode("utf-8")
        assert response.status == 200
        assert "AI Dubbing Studio" in body

        connection.request("GET", "/missing")
        missing_response = connection.getresponse()
        missing_response.read()
        assert missing_response.status == 404
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
