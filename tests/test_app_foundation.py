"""Tests for the runnable application foundation."""

from __future__ import annotations

import importlib
import json
from http.client import HTTPConnection
from threading import Thread

import pytest

from api.server import create_server
from app.factory import Application, create_application
from src.config.settings import Config
from web import create_web_server, render_index_page

cli_module = importlib.import_module("cli.main")


def test_create_application_loads_runtime_metadata():
    application = create_application()

    assert application.name == "AI Dubbing Studio"
    assert application.environment == "development"
    assert application.config.sample_rate == 22050
    assert application.runtime_settings["host"] == "127.0.0.1"
    assert application.output_path.exists()
    assert application.log_path.exists()


def test_create_application_rejects_missing_explicit_config():
    with pytest.raises(FileNotFoundError):
        create_application(config_path="missing-config.yaml")


def test_render_index_page_contains_expected_sections():
    html = render_index_page(create_application())

    assert "AI Dubbing Studio" in html
    assert "/static/index.html" in html
    assert "Runnable scaffold only" in html
    assert "Preview unavailable. Open" in html
    assert "Open the frontend preview directly" in html


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
    exit_code = cli_module.main(["info"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["name"] == "AI Dubbing Studio"


def test_cli_check_reports_success(capsys):
    exit_code = cli_module.main(["check"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Initialized AI Dubbing Studio" in captured.out


def test_cli_accepts_zero_port(monkeypatch):
    recorded = {}

    def fake_run_api_server(host: str, port: int, application):
        recorded["host"] = host
        recorded["port"] = port
        recorded["application"] = application

    monkeypatch.setattr(cli_module, "run_api_server", fake_run_api_server)

    exit_code = cli_module.main(["serve-api", "--port", "0"])

    assert exit_code == 0
    assert recorded["port"] == 0


def test_api_server_endpoints():
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
        health_payload = json.loads(response.read().decode("utf-8"))
        assert response.status == 200
        assert health_payload["status"] == "ok"

        connection.request("GET", "/config")
        config_response = connection.getresponse()
        config_payload = json.loads(config_response.read().decode("utf-8"))
        assert config_response.status == 200
        assert "paths" not in config_payload
        assert config_payload["audio"]["sample_rate"] == 22050
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_web_server_renders_escaped_metadata_and_routes():
    application = Application(
        config=Config(),
        environment="dev<script>",
        root_path=create_application().root_path,
        config_path=None,
        runtime_settings={},
    )
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
        assert "/static/index.html" in body
        assert "dev&lt;script&gt;" in body
        assert "dev<script>" not in body

        connection.request("GET", "/static/index.html")
        static_response = connection.getresponse()
        static_body = static_response.read().decode("utf-8")
        assert static_response.status == 200
        assert "Available run modes" in static_body

        connection.request("GET", "/missing")
        missing_response = connection.getresponse()
        missing_response.read()
        assert missing_response.status == 404
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
