# Project Instructions

- Read [README.md](../README.md) for run/deploy instructions and MySQL access. Avoid duplicating its procedures here.
- API entry point: `app/main.py`; routes: `app/routes.py`; schemas: `app/schemas.py`; inference: `app/prediction.py`; audit/metrics middleware: `app/observability.py`; encrypted persistence and retention: `app/data_store.py`.
- Preserve the `uvicorn app.main:app` entry point and existing API field aliases.
- Keep health payloads, identifiers, encryption keys, and database credentials out of logs, fixtures, and committed files. Persist payloads only encrypted; never commit `.env`.
- After Python changes, run `python -m pytest` and `python -m ruff check .`. For deployment config changes, validate with `docker compose config` (provide required env vars) and `kubectl create --dry-run=client -f <manifest> -o yaml`.
- Start/stop the local Kubernetes workflow with `start_services.py` and `stop_services.py`; these scripts also manage model training, so do not start or stop them casually.
