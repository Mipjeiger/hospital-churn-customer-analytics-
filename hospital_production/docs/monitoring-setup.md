# Monitoring Setup for Hospital ML Engineering Production

## Overview
This guide walks you through adding **Prometheus**, **Grafana**, **Langfuse**, and API‑level metrics to the `hospital_production` codebase.  After completing the steps you will have:
- End‑to‑end observability of Prefect flows, model health, and business KPIs.
- A Grafana dashboard (`hospital_api.json`) automatically provisioned.
- Alerts for flow failures, model downtime, and KPI thresholds.
- Langfuse UI running in Docker for LLM traceability.
- Production‑ready Docker‑Compose configuration (pinned tags, healthchecks, restart policies).

---

## 1. Prometheus & Grafana Provisioning
| Step | Action | Command / Code | Verify |
|------|--------|----------------|--------|
|1.1|Add Prometheus scrape target for FastAPI|Edit `app/prometheus/prometheus.yml` to include:
```yaml
scrape_configs:
  - job_name: "hospital_api"
    static_configs:
      - targets: ["host.docker.internal:8000"]
```
|`curl http://localhost:9090/api/v1/targets` lists `hospital_api`|
|1.2|Expose FastAPI port in Docker|In `docker-compose.yml` ensure the API service has `ports: ["8000:8000"]`|`docker compose ps` shows the container reachable on `localhost:8000`|
|1.3|Install `prometheus_client` in API image|Add to `docker/requirements_api.txt`: `prometheus_client>=0.20`
|`docker compose build api && docker compose up -d api` succeeds|
|1.4|Instrument Prefect flows|Create `app/flows/_metrics.py`:
```python
from prometheus_client import Counter
TRAIN_RUN = Counter('prefect_train_runs_total','Prefect train flow runs')
PREDICT_RUN = Counter('prefect_predict_runs_total','Prefect predict flow runs')
LLM_EVAL_RUN = Counter('prefect_llm_eval_runs_total','Prefect LLM eval runs')
```
Import and increment the appropriate counter at the start/end of each flow (see `app/flows/*_flow.py`).|Run a flow locally, then `curl http://localhost:9090/api/v1/query?query=prefect_train_runs_total` returns a non‑zero value|
|1.5|(Optional) Pushgateway for short‑lived jobs|Add service `pushgateway` to `docker-compose.yml` (image `prom/pushgateway`).|`docker compose logs pushgateway` shows it listening|
|1.6|Grafana datasource provisioning|File `app/grafana/provisioning/datasources/prometheus.yml` already points to `http://prometheus:9090`. No change needed unless you renamed the service.|Open Grafana → **Configuration → Data Sources** → Prometheus is default|
|1.7|Dashboard provisioning|Dashboard JSON is at `app/grafana/dashboards/hospital_api.json` and is referenced by `dashboards.yml`.|Grafana UI shows the dashboard without manual import|
|1.8|Add alerts (example) for flow inactivity|Create `app/grafana/provisioning/rules/alerts.yml` with:
```yaml
groups:
  - name: ml_flow_alerts
    rules:
      - alert: PrefectFlowFailure
        expr: increase(prefect_train_runs_total[5m]) == 0
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Prefect train flow has not run in the last 5 min"
```
|`docker exec prometheus promtool check rules /etc/prometheus/rules/alerts.yml` outputs **OK**|

---

## 2. Model‑Alive / Dead Tracking
1. **Expose health endpoint** – add to `app/api/main.py`:
```python
from app.api.metrics import MODEL_LOADED
@app.get("/model/{name}/health")
def model_health(name: str):
    return {"model": name, "loaded": bool(MODEL_LOADED.labels(model_name=name)._value.get())}
```
2. **Set gauge when loading models** – modify `app/api/dependencies.py` (or wherever `get_model_bundle` lives):
```python
if success:
    MODEL_LOADED.labels(model_name=concept).set(1)
else:
    MODEL_LOADED.labels(model_name=concept).set(0)
```
3. **Grafana panel** – duplicate a *Stat* panel in `hospital_api.json` using `model_loaded{model_name="<model>"}`.
4. **Alert for dead models** – add to `alerts.yml`:
```yaml
- alert: ModelNotLoaded
  expr: model_loaded == 0
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "Model {{ $labels.model_name }} is not loaded"
```

---

## 3. Business KPI Monitoring (ML/LLM)
1. **Define KPI gauges** – add to `app/api/metrics.py`:
```python
BUSINESS_KPI = Gauge('business_kpi','Business KPI derived from ML/LLM', ['kpi_name'])
```
2. **Set KPI values in code** – after computing a KPI in a flow, call:
```python
BUSINESS_KPI.labels(kpi_name='model_accuracy').set(accuracy)
```
3. **Grafana panels** – create Stat/Time‑Series panels with expressions like `business_kpi{kpi_name="model_accuracy"}`.
4. **Alert example (accuracy < 0.70)**:
```yaml
- alert: ModelAccuracyDrop
  expr: business_kpi{kpi_name="model_accuracy"} < 0.70
  for: 10m
  labels:
    severity: critical
  annotations:
    summary: "Model accuracy dropped below 70 %"
```

---

## 4. Run Langfuse in Docker & Access UI
Add the following services to `docker-compose.yml` (replace any existing `langfuse` entry):
```yaml
postgres:
  image: postgres:16-alpine
  environment:
    POSTGRES_USER: postgres
    POSTGRES_PASSWORD: postgres
    POSTGRES_DB: langfuse
  volumes:
    - pgdata:/var/lib/postgresql/data
  healthcheck:
    test: ["CMD", "pg_isready", "-U", "postgres"]
    interval: 10s
    timeout: 5s
    retries: 5

langfuse:
  image: langfuse/langfuse:2.1.0   # pin to a concrete version
  container_name: langfuse
  ports:
    - "4000:3000"
  environment:
    - LANGFUSE_HOST=0.0.0.0
    - LANGFUSE_PORT=3000
    - DATABASE_URL=postgresql://postgres:postgres@postgres:5432/langfuse
    - NEXTAUTH_URL=http://localhost:4000
  depends_on:
    postgres:
      condition: service_healthy
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:3000/api/public/health"]
    interval: 15s
    timeout: 5s
    retries: 5
```
After `docker compose up -d`:
- Open **http://localhost:4000** → sign‑up/login screen.
- (Optional) Install the Python SDK in your API image and emit traces.

---

## 5. API Monitoring Enhancements
- `/metrics` endpoint already exists (see `app/api/main.py`).
- **Request/Response size metrics** – add to `metrics_middleware` after `call_next`:
```python
REQUEST_SIZE = Counter('http_request_body_bytes_total','Request body size bytes', ['endpoint','method'])
RESPONSE_SIZE = Counter('http_response_body_bytes_total','Response body size bytes', ['endpoint','status'])
REQUEST_SIZE.labels(endpoint=endpoint, method=request.method).inc_by(len(await request.body()))
RESPONSE_SIZE.labels(endpoint=endpoint, status=response.status_code).inc_by(len(response.body) if hasattr(response,'body') else 0)
```
- **Prediction counter** – already present (`ML_PREDICTIONS`). Ensure each prediction route increments it.
- Add Grafana panels for error rate, latency distribution, and request volume (already in `hospital_api.json`).

---

## 6. Production‑Ready Docker‑Compose
- **Pin image tags** – replace all `latest` entries with explicit versions (e.g., `grafana/grafana:10.4.0`).
- **Healthchecks** – see the `healthcheck` sections added for `postgres` and `langfuse` above. Add similar checks for `prometheus` and `grafana` if desired.
- **Restart policy** – add `restart: unless-stopped` to every service.
- **Environment variables** – store secrets in a `.env` file (git‑ignored) and reference them as `${VAR}` in the compose file.
- **CI linting** – add a simple GitHub Action that runs `jsonlint` on `*.json` and `yamllint` on `*.yml`.
- **Run‑book** – document the commands (`docker compose up`, `docker compose logs`, health‑check URLs) in `README.md`.

---

## 7. Quick “All‑in‑One” Setup Script
Create `setup.sh` at the repo root (make it executable with `chmod +x setup.sh`):
```bash
#!/usr/bin/env bash
set -euo pipefail

# 1️⃣ Install missing Python deps inside the API container
docker compose exec api pip install -r /app/requirements_api.txt prometheus_client

# 2️⃣ Apply metric module for Prefect flows (creates file if absent)
docker compose exec api bash -c "
cat > /app/flows/_metrics.py <<'PY'
from prometheus_client import Counter
TRAIN_RUN = Counter('prefect_train_runs_total','Prefect train flow runs')
PREDICT_RUN = Counter('prefect_predict_runs_total','Prefect predict flow runs')
LLM_EVAL_RUN = Counter('prefect_llm_eval_runs_total','Prefect LLM eval runs')
PY
"

# 3️⃣ Re‑build and start the full stack
docker compose up -d --build

# 4️⃣ Wait for health (simple poll loop)
for svc in prometheus grafana api langfuse; do
  echo "Waiting for $svc…"
  while ! docker compose exec $svc curl -sf http://localhost:9090/-/healthy > /dev/null 2>&1 || ! docker compose exec $svc curl -sf http://localhost:3000/api/health > /dev/null 2>&1; do
    sleep 2
  done
done

echo "All services healthy."

echo "Grafana → http://localhost:3000"

echo "Prometheus → http://localhost:9090"

echo "Langfuse → http://localhost:4000"
```
Run it with `./setup.sh` – it will rebuild containers, apply the metric file, start everything, and print the UI URLs.

---

## 8. Verification Checklist
- **Prometheus UI** (`http://localhost:9090`) shows targets `hospital_api` and any `prefect_*` counters.
- **Grafana UI** (`http://localhost:3000`) loads the *Hospital API* dashboard without manual import.
- **Langfuse UI** (`http://localhost:4000`) reachable, you can create an admin account.
- **API health endpoint** (`http://localhost:8000/health`) returns `{"status":"healthy"}`.
- **Metrics endpoint** (`http://localhost:8000/metrics`) contains all custom metrics (`http_requests_total`, `ml_predictions_total`, `model_loaded`, `business_kpi`).
- **Alerts** fire in Grafana when you manually trigger a condition (e.g., stop the API to cause error‑rate alert).

---

## 9. Further Reading & References
- Prometheus client docs: https://github.com/prometheus/client_python
- Grafana provisioning format: https://grafana.com/docs/grafana/latest/administration/provisioning/
- Langfuse docs: https://langfuse.com/docs
- Prefect metrics integration: https://docs.prefect.io/latest/concepts/monitoring/

---

*Keep this file version‑controlled; any future changes to monitoring should be made here first and then propagated to the code.*
