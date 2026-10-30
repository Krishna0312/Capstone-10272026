# Project Instructions

- Read [README.md](../README.md) for run/deploy instructions and MySQL access. Avoid duplicating its procedures here.
- API entry point: `app/main.py`; authentication: `app/auth.py`; routes: `app/routes.py`; schemas: `app/schemas.py`; inference: `app/prediction.py`; audit/metrics middleware: `app/observability.py`; encrypted persistence, accounts, sessions, and retention: `app/data_store.py`.
- Preserve the `uvicorn app.main:app` entry point and existing API field aliases.
- Prediction and account-management routes require an authenticated session and CSRF validation; preserve the public health/metrics endpoints. Store passwords only as scrypt hashes and session tokens only as hashes. Keep health payloads, identifiers, encryption keys, and database credentials out of logs, fixtures, and committed files. Persist request/audit payloads only encrypted; never commit `.env`.
- After Python changes, run `python -m pytest` and `python -m ruff check .`. For deployment config changes, validate with `docker compose config` (provide required env vars) and `kubectl create --dry-run=client -f <manifest> -o yaml`.
- Start/stop Kubernetes local port-forwards with `start_kube_services.py` and `stop_kube_services.py`; start/stop Docker Compose with `start_docker_services.py` and `stop_docker_services.py`. These scripts do not start model training.
