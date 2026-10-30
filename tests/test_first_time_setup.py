import base64
import json

import pytest
from cryptography.fernet import Fernet

import first_time_setup


def test_compose_environment_generates_secrets_and_preserves_them(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    credentials_file = tmp_path / "credentials.json"
    monkeypatch.setattr(first_time_setup, "COMPOSE_ENV", env_file)
    monkeypatch.setattr(first_time_setup, "CREDENTIALS_FILE", credentials_file)

    settings, generated_password = first_time_setup._ensure_compose_env()

    assert generated_password
    assert settings["MYSQL_ROOT_PASSWORD"] != "replace-with-a-strong-random-password"
    assert settings["MYSQL_PASSWORD"] != "replace-with-a-strong-random-password"
    Fernet(settings["APP_DATA_ENCRYPTION_KEY"].encode("ascii"))
    assert len(settings["AUTH_BOOTSTRAP_ADMIN_PASSWORD"]) >= 12
    assert len(settings["GRAFANA_ADMIN_PASSWORD"]) >= 12
    first_password = settings["AUTH_BOOTSTRAP_ADMIN_PASSWORD"]
    first_grafana_password = settings["GRAFANA_ADMIN_PASSWORD"]
    first_time_setup._save_credentials(
        "compose",
        settings["AUTH_BOOTSTRAP_ADMIN_USERNAME"],
        first_password,
    )
    saved_credentials = json.loads(credentials_file.read_text(encoding="utf-8"))
    assert saved_credentials["compose"]["password"] == first_password

    rerun_settings, generated_password = first_time_setup._ensure_compose_env()

    assert not generated_password
    assert rerun_settings["AUTH_BOOTSTRAP_ADMIN_PASSWORD"] == first_password
    assert rerun_settings["GRAFANA_ADMIN_PASSWORD"] == first_grafana_password


def test_compose_setup_waits_for_api_and_monitoring_services(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    credentials_file = tmp_path / "credentials.json"
    commands = []
    urls = []
    monkeypatch.setattr(first_time_setup, "COMPOSE_ENV", env_file)
    monkeypatch.setattr(first_time_setup, "CREDENTIALS_FILE", credentials_file)
    monkeypatch.setattr(first_time_setup.shutil, "which", lambda _: "docker")
    monkeypatch.setattr(first_time_setup, "_run", lambda args, **kwargs: commands.append(args))
    monkeypatch.setattr(first_time_setup, "_wait_for_url", urls.append)

    first_time_setup._compose_setup()

    assert commands == [["docker", "compose", "up", "--build", "-d"]]
    assert urls == [
        "http://127.0.0.1:8001/health",
        "http://127.0.0.1:3001/api/health",
        "http://127.0.0.1:9091/-/ready",
        "http://127.0.0.1:9092/-/healthy",
    ]


def test_compose_setup_rejects_invalid_port_before_starting_services(monkeypatch):
    commands = []
    settings = {"API_PORT": "70000"}
    monkeypatch.setattr(first_time_setup, "_ensure_compose_env", lambda: (settings, False))
    monkeypatch.setattr(first_time_setup.shutil, "which", lambda _: "docker")
    monkeypatch.setattr(first_time_setup, "_run", lambda args, **kwargs: commands.append(args))

    with pytest.raises(RuntimeError, match="API_PORT"):
        first_time_setup._compose_setup()

    assert commands == []


def test_existing_kubernetes_secret_is_never_replaced_or_incompletely_accepted():
    with pytest.raises(RuntimeError, match="missing keys"):
        first_time_setup._secret_values({"data": {}}, None)

    complete_secret = {
        "data": {key: "existing" for key in first_time_setup.SECRET_KEYS}
    }
    assert first_time_setup._secret_values(complete_secret, None) == {}


def test_new_kubernetes_secret_reuses_compose_admin_credentials():
    compose_settings = {
        "AUTH_BOOTSTRAP_ADMIN_USERNAME": "first-admin",
        "AUTH_BOOTSTRAP_ADMIN_PASSWORD": "known-bootstrap-password",
    }
    secret_values = first_time_setup._secret_values(None, compose_settings)
    secret = json.loads(first_time_setup._secret_manifest(secret_values))
    decoded = {
        key: base64.b64decode(value).decode("utf-8")
        for key, value in secret["data"].items()
    }

    assert set(decoded) == set(first_time_setup.SECRET_KEYS)
    assert decoded["AUTH_BOOTSTRAP_ADMIN_USERNAME"] == "first-admin"
    assert decoded["AUTH_BOOTSTRAP_ADMIN_PASSWORD"] == "known-bootstrap-password"
    assert "known-bootstrap-password" not in first_time_setup._secret_manifest(secret_values)


def test_install_monitoring_applies_manifest_and_waits_for_all_services(monkeypatch):
    commands = []
    monkeypatch.setattr(first_time_setup, "_run", lambda args, **kwargs: commands.append(args))

    first_time_setup._install_monitoring()

    assert commands[0] == ["kubectl", "apply", "-f", "k8s/monitoring.yaml"]
    assert [
        command[3]
        for command in commands[1:]
    ] == [
        "deployment/prometheus",
        "deployment/grafana",
        "deployment/pushgateway",
    ]
    assert all(command[4:7] == ["-n", "monitoring", "--timeout=300s"] for command in commands[1:])
