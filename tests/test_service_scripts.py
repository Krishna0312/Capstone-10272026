from unittest.mock import MagicMock

import start_docker_services
import stop_docker_services


def test_docker_start_uses_configured_ports_and_waits_for_each_service(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(
        "API_PORT=18080\nGRAFANA_PORT=13001\nPROMETHEUS_PORT=19091\nPUSHGATEWAY_PORT=19092\n",
        encoding="utf-8",
    )
    commands = []
    urls = []
    monkeypatch.setattr(start_docker_services, "ROOT", tmp_path)
    monkeypatch.setattr(start_docker_services.shutil, "which", lambda _: "docker")
    monkeypatch.setattr(
        start_docker_services.subprocess,
        "run",
        lambda *args, **kwargs: commands.append((args, kwargs)),
    )

    def ready(url, timeout):
        urls.append(url)
        response = MagicMock()
        response.status = 200
        response.__enter__.return_value = response
        return response

    monkeypatch.setattr(start_docker_services, "urlopen", ready)

    assert start_docker_services.main() == 0

    assert commands == [
        ((["docker", "compose", "up", "-d"],), {"cwd": tmp_path, "check": True})
    ]
    assert urls == [
        "http://127.0.0.1:18080/health",
        "http://127.0.0.1:13001/api/health",
        "http://127.0.0.1:19091/-/ready",
        "http://127.0.0.1:19092/-/healthy",
    ]


def test_docker_start_rejects_invalid_port_before_starting_compose(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("API_PORT=70000\n", encoding="utf-8")
    commands = []
    monkeypatch.setattr(start_docker_services, "ROOT", tmp_path)
    monkeypatch.setattr(start_docker_services.shutil, "which", lambda _: "docker")
    monkeypatch.setattr(
        start_docker_services.subprocess,
        "run",
        lambda *args, **kwargs: commands.append(args),
    )

    assert start_docker_services.main() == 1
    assert not commands


def test_docker_stop_brings_down_compose_without_deleting_volumes(tmp_path, monkeypatch):
    commands = []
    monkeypatch.setattr(stop_docker_services, "ROOT", tmp_path)
    monkeypatch.setattr(stop_docker_services.shutil, "which", lambda _: "docker")
    monkeypatch.setattr(
        stop_docker_services.subprocess,
        "run",
        lambda *args, **kwargs: commands.append((args, kwargs)),
    )

    assert stop_docker_services.main() == 0

    assert commands == [
        ((["docker", "compose", "down"],), {"cwd": tmp_path, "check": True})
    ]
