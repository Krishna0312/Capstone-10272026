# Readmission Risk API

FastAPI service that returns readmission risk predictions using the model in `artifacts/`. The API exposes request metrics for Prometheus and is configured for Kubernetes deployment with Grafana dashboards.

## Run Locally

Requirements: Python 3.12+ and Docker (optional for the Docker workflow).

### Start or stop the complete local workflow

The cross-platform Python scripts below start model training and create local port-forwards to the existing Kubernetes API and monitoring services. Ensure the Kubernetes cluster and the `readmission-risk-api`, `grafana`, and `prometheus` services are running first. Install the project dependencies, then run these commands from the repository root on Windows, macOS, or Linux:

```powershell
python start_all.py
```

```bash
python3 start_all.py
```

Use the matching Python command to stop only the training and port-forward processes started by the start script:

```powershell
python stop_all.py
```

```bash
python3 stop_all.py
```

The scripts write logs and process state under `.run/`. They leave processes that were already running alone; they also avoid starting another training run if one is already active. The startup capture records each API, Prometheus, and Grafana pod's creation-to-ready duration in `.run/logs/cold_starts.jsonl` and publishes the latest timings to Prometheus through Pushgateway. Apply `k8s/monitoring.yaml` to install Pushgateway and configure Prometheus to scrape it, then re-import the updated Grafana dashboard to see the **Application Cold Start** panel. The watcher records existing ready pods on startup and newly created pods; it measures Kubernetes pod startup through readiness, not first-prediction model loading. Stopping training early discards its unfinished work. Training updates the model artifact on the local machine; rebuilding and redeploying the API image is required for the Kubernetes API to use that newly trained model.

Install the full development and training environment, then run the API:

```powershell
python -m pip install -r requirements-dev.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Or run in Docker:

```powershell
docker build -t readmission-risk-api:metrics .
docker run --rm -p 8000:8000 readmission-risk-api:metrics
```

API pages and endpoints:

- Form: <http://127.0.0.1:8000/>
- Swagger UI: <http://127.0.0.1:8000/docs>
- ReDoc: <http://127.0.0.1:8000/redoc>
- Health: <http://127.0.0.1:8000/health>
- Prometheus metrics: <http://127.0.0.1:8000/metrics>

## Deploy to Docker Desktop Kubernetes

The manifests expect the API image `readmission-risk-api:metrics` to be available to the Kubernetes node. Docker Desktop Kubernetes uses containerd, so building an image in Docker alone may leave the cluster using an older cached image. The following PowerShell commands build and import the image into the `desktop-control-plane` node, then apply the application and monitoring manifests:

```powershell
docker build -t readmission-risk-api:metrics .
$archive = Join-Path $env:TEMP 'readmission-risk-api.tar'
docker save -o $archive readmission-risk-api:metrics
docker cp $archive desktop-control-plane:/root/readmission-risk-api.tar
docker exec desktop-control-plane ctr -n k8s.io images import /root/readmission-risk-api.tar
docker exec desktop-control-plane rm -f /root/readmission-risk-api.tar
Remove-Item $archive

kubectl apply -f k8s/readmission-api.yml
kubectl apply -f k8s/monitoring.yaml
kubectl rollout status deployment/readmission-risk-api -n default
kubectl rollout status deployment/prometheus -n monitoring
kubectl rollout status deployment/grafana -n monitoring
```

The `desktop-control-plane` image-import commands are specific to this Docker Desktop Kubernetes setup. For another Kubernetes cluster, push the image to a registry accessible by the nodes and update the image in `k8s/readmission-api.yml`.

### Rebuild the API image

Use a new tag for each rebuild so Kubernetes cannot reuse an old cached tag. For example, set `$tag` to a new value, then run:

```powershell
$tag = 'metrics-v2'
docker build -t "readmission-risk-api:$tag" .
$archive = Join-Path $env:TEMP 'readmission-risk-api.tar'
docker save -o $archive "readmission-risk-api:$tag"
docker cp $archive desktop-control-plane:/root/readmission-risk-api.tar
docker exec desktop-control-plane ctr -n k8s.io images import /root/readmission-risk-api.tar
docker exec desktop-control-plane rm -f /root/readmission-risk-api.tar
Remove-Item $archive
kubectl set image deployment/readmission-risk-api -n default "readmission-risk-api=readmission-risk-api:$tag"
kubectl rollout status deployment/readmission-risk-api -n default
```

Update the `image` value in `k8s/readmission-api.yml` to the same new tag so future `kubectl apply` commands keep using that build.

## Monitoring and Grafana

Prometheus scrapes `readmission-risk-api.default.svc.cluster.local:8000/metrics` every 15 seconds. Grafana's default Prometheus datasource is provisioned by `k8s/monitoring.yaml`.

To access Grafana locally, run this in a PowerShell terminal and leave it running:

```powershell
kubectl port-forward -n monitoring svc/grafana 3001:3000
```

Open <http://127.0.0.1:3001>. The configured initial Grafana credentials are `admin` / `admin`; change the password before exposing Grafana beyond a local development environment.

Import the dashboard using **Dashboards → New → Import**, then select `k8s/readmission-api-grafana-dashboard.json`. The datasource is named `Prometheus`. Dashboard query data appears after Prometheus scrapes the app and requests have reached the API. The error-rate panel can be empty when no 4xx or 5xx requests have occurred.

The start script forwards Prometheus to port `9091`. To create the same forward manually, use:

```powershell
kubectl port-forward -n monitoring svc/prometheus 9091:9090
```

Then open <http://127.0.0.1:9091/targets> and confirm the `readmission-api` target is `UP`.

If a local port is already in use, choose another local port, for example `13001:3000` for Grafana or `19090:9090` for Prometheus.

## Useful Kubernetes Checks

```powershell
kubectl get pods,services -n default
kubectl get pods,services -n monitoring
kubectl logs -n default deployment/readmission-risk-api --tail=100
kubectl logs -n monitoring deployment/prometheus --tail=100
```

The API service is internal DNS `readmission-risk-api.default.svc.cluster.local:8000`; the Prometheus and Grafana services are in the `monitoring` namespace.

## Tests

Install `requirements-dev.txt` first, then run:

```powershell
python -m pytest -q
```
