import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Set
from urllib.error import URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent
OUTPUT_FILE = ROOT / ".run" / "logs" / "cold_starts.jsonl"
POLL_INTERVAL_SECONDS = 2
PUSHGATEWAY_URL = "http://127.0.0.1:9092"
APPLICATIONS = (
    ("default", "readmission-risk-api", "readmission-risk-api"),
    ("monitoring", "prometheus", "prometheus"),
    ("monitoring", "grafana", "grafana"),
)


def _escape_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _push_metric(record: Dict[str, Any]) -> None:
    namespace = quote(str(record["namespace"]), safe="")
    application = quote(str(record["application"]), safe="")
    endpoint = (
        f"{PUSHGATEWAY_URL}/metrics/job/application-cold-start"
        f"/namespace/{namespace}/application/{application}"
    )
    pod = _escape_label(str(record["pod"]))
    body = (
        f'cold_start_seconds{{pod="{pod}"}} '
        f'{record["cold_start_seconds"]}\n'
    ).encode("utf-8")
    request = Request(endpoint, data=body, method="PUT")
    with urlopen(request, timeout=5) as response:
        if response.status not in (200, 202):
            raise RuntimeError(f"Pushgateway returned HTTP {response.status}.")


def _read_recorded_pods() -> Set[str]:
    recorded: Set[str] = set()
    if not OUTPUT_FILE.exists():
        return recorded

    with OUTPUT_FILE.open(encoding="utf-8") as records:
        for line_number, line in enumerate(records, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"Invalid JSON in {OUTPUT_FILE} on line {line_number}: {error}"
                ) from error
            pod_uid = record.get("pod_uid")
            if isinstance(pod_uid, str):
                recorded.add(pod_uid)
    return recorded


def _ready_time(pod: Dict[str, Any]) -> str | None:
    for condition in pod.get("status", {}).get("conditions", []):
        if condition.get("type") == "Ready" and condition.get("status") == "True":
            ready_time = condition.get("lastTransitionTime")
            if isinstance(ready_time, str):
                return ready_time
    return None


def _capture_application(
    namespace: str,
    application: str,
    label: str,
    recorded: Set[str],
    exported: Set[str],
) -> None:
    result = subprocess.run(
        [
            "kubectl",
            "get",
            "pods",
            "-n",
            namespace,
            "-l",
            f"app={label}",
            "-o",
            "json",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        print(
            f"Could not inspect {namespace}/{application}: {detail}",
            file=sys.stderr,
            flush=True,
        )
        return

    try:
        pod_list = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        print(
            f"Could not parse pod status for {namespace}/{application}: {error}",
            file=sys.stderr,
            flush=True,
        )
        return

    for pod in pod_list.get("items", []):
        metadata = pod.get("metadata", {})
        pod_uid = metadata.get("uid")
        creation_time = metadata.get("creationTimestamp")
        ready_time = _ready_time(pod)
        if (
            not isinstance(pod_uid, str)
            or pod_uid in exported
            or not isinstance(creation_time, str)
            or ready_time is None
        ):
            continue

        try:
            created = datetime.fromisoformat(creation_time.replace("Z", "+00:00"))
            ready = datetime.fromisoformat(ready_time.replace("Z", "+00:00"))
        except ValueError as error:
            print(
                f"Could not parse startup timestamps for pod {metadata.get('name')}: {error}",
                file=sys.stderr,
                flush=True,
            )
            continue

        cold_start_seconds = (ready - created).total_seconds()
        if cold_start_seconds < 0:
            print(
                f"Ready timestamp predates pod creation for {metadata.get('name')}; skipping.",
                file=sys.stderr,
                flush=True,
            )
            continue

        record = {
            "application": application,
            "namespace": namespace,
            "pod": metadata.get("name"),
            "pod_uid": pod_uid,
            "pod_created_at": created.isoformat(),
            "pod_ready_at": ready.isoformat(),
            "cold_start_seconds": round(cold_start_seconds, 3),
            "captured_at": datetime.now().astimezone().isoformat(),
        }
        try:
            _push_metric(record)
        except (OSError, RuntimeError) as error:
            print(
                f"Could not publish cold-start metric for "
                f"{namespace}/{application}/{metadata.get('name')}: {error}",
                file=sys.stderr,
                flush=True,
            )
            continue

        exported.add(pod_uid)
        if pod_uid not in recorded:
            OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
            with OUTPUT_FILE.open("a", encoding="utf-8") as records:
                records.write(json.dumps(record) + "\n")
            recorded.add(pod_uid)
            print(json.dumps(record), flush=True)


def capture_once(recorded: Set[str], exported: Set[str]) -> None:
    for namespace, application, label in APPLICATIONS:
        _capture_application(namespace, application, label, recorded, exported)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Record Kubernetes pod creation-to-readiness startup times."
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Capture currently ready application pods, then exit.",
    )
    args = parser.parse_args()

    try:
        recorded = _read_recorded_pods()
        exported: Set[str] = set()
        if args.once:
            capture_once(recorded, exported)
            return 0

        print(
            f"Recording pod cold starts for API, Prometheus, and Grafana in {OUTPUT_FILE}",
            flush=True,
        )
        while True:
            capture_once(recorded, exported)
            time.sleep(POLL_INTERVAL_SECONDS)
    except (OSError, RuntimeError, subprocess.SubprocessError, URLError) as error:
        print(f"Could not capture application startup times: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
