import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parent
DEFAULT_PORTS = {
    "API_PORT": "8000",
    "GRAFANA_PORT": "3001",
    "PROMETHEUS_PORT": "9091",
    "PUSHGATEWAY_PORT": "9092",
}


def _ports() -> dict[str, str]:
    env_file = ROOT / ".env"
    values = DEFAULT_PORTS.copy()
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            match = re.match(r"^\s*(API_PORT|GRAFANA_PORT|PROMETHEUS_PORT|PUSHGATEWAY_PORT)\s*=\s*(.*?)\s*$", line)
            if match:
                values[match.group(1)] = match.group(2).strip("\"'")
    for key, value in values.items():
        if not value.isdecimal() or not 1 <= int(value) <= 65535:
            raise RuntimeError(f"{key} in .env must be a port number between 1 and 65535.")
    return values


def _wait_for_url(url: str, service: str) -> None:
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        try:
            with urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except (OSError, URLError):
            time.sleep(1)
    raise RuntimeError(f"{service} did not become ready at {url} within 120 seconds.")


def main() -> int:
    if not shutil.which("docker"):
        print("Docker CLI was not found. Install/start Docker Desktop or Docker Engine.", file=sys.stderr)
        return 1

    try:
        ports = _ports()
        subprocess.run(
            ["docker", "compose", "up", "-d"],
            cwd=ROOT,
            check=True,
        )
        endpoints = (
            ("API_PORT", "/health", "API"),
            ("GRAFANA_PORT", "/api/health", "Grafana"),
            ("PROMETHEUS_PORT", "/-/ready", "Prometheus"),
            ("PUSHGATEWAY_PORT", "/-/healthy", "Pushgateway"),
        )
        for setting, path, service in endpoints:
            url = f"http://127.0.0.1:{ports[setting]}{path}"
            _wait_for_url(url, service)
            print(f"{service} is ready: http://127.0.0.1:{ports[setting]}")
        return 0
    except (OSError, subprocess.SubprocessError, RuntimeError) as error:
        print(f"Could not start Docker services: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
