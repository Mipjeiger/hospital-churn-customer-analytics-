# 🏥 Hospital Production End-to-end ML Engineer system

A practical runbook for configuring the local environment, starting infrastructure, registering LLM prompts, configuring Feast, training models, running predictions and evaluations, and serving the API.

> **Security:** Never commit `.env`, API keys, passwords, access tokens, private keys, or production data. Use `.env.example` with placeholder values only. Rotate any credential that has already been exposed.

## Table of Contents

- [Architecture and services](#-architecture-and-services)
- [Step 0 — One-time setup](#-step-0--one-time-setup)
- [Step 1 — Start infrastructure](#-step-1--start-infrastructure)
- [Step 2 — Register LLM prompts](#-step-2--register-llm-prompts-in-mlflow)
- [Step 3 — Configure Feast](#-step-3--configure-feast-feature-store)
- [Step 4 — Train models](#-step-4--train-models)
- [Step 5 — Run predictions](#-step-5--run-predictions)
- [Step 6 — Evaluate LLM providers](#-step-6--evaluate-llm-providers)
- [Step 7 — Launch the API](#-step-7--launch-the-api)
- [Daily and weekly operations](#-daily-and-weekly-operations)
- [Troubleshooting](#-troubleshooting)
- [Teardown](#-teardown)

## 🧭 Architecture and services

| Service | Local URL | Purpose |
|---|---|---|
| MinIO Console | http://localhost:9001 | Browse local object storage |
| MLflow | http://localhost:5000 | Track experiments, models, and prompts |
| Langfuse | http://localhost:3000 | LLM observability and tracing |
| Prefect | http://localhost:4200 | Workflow orchestration |
| Feast UI | http://localhost:8888 | Inspect the feature store |
| API docs | http://localhost:8000/docs | Explore API endpoints |

URLs assume the default local port mappings in your Compose configuration. Check `docker-compose.yml` if your ports differ.

## ⚙️ Step 0 — One-time setup

Run from the repository root:

```bash
cd hospital_production

# Create a local environment file if one does not already exist.
# Prefer copying a placeholder-only example; do not overwrite an existing .env.
if [ ! -f .env ]; then
  cp .env.example .env 2>/dev/null || touch .env
fi

# Create and activate a host-side Python virtual environment.
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install dependencies used by the host-side CLI flows.
pip install -r docker/requirements_api.txt
```

Fill in the required variables in your local `.env` file. Depending on the enabled integrations, these may include `HUGGINGFACE_API_KEY`, `GROQ_API_KEY`, `NINEROUTER_*`, and `LANGFUSE_*`. Use actual credentials only in the ignored local `.env` file or a secrets manager.

**Why?** Commands such as `python -m app.flows.train_flow` run on the host and need their Python dependencies installed. Docker runs the configured services.

### 🔐 Recommended `.gitignore`

Ensure the repository-root `.gitignore` includes:

```gitignore
# Secrets and environment files
.env
.env.*
!.env.example
secrets/
*.pem
*.key

# Python
.venv/
venv/
__pycache__/
*.py[cod]

# Local outputs (adjust to your project)
mlruns/
chroma_store/
```

Do not commit real values to `.env.example`; use placeholders such as `YOUR_API_KEY_HERE`.

## 🐳 Step 1 — Start infrastructure

```bash
docker compose up -d minio mlflow langfuse langfuse-db prefect
```

Wait for services to initialize, then inspect their status and logs:

```bash
docker compose ps
docker compose logs --tail=100 mlflow langfuse prefect
curl -fsS http://localhost:5000/health || echo "MLflow is not ready yet"
curl -fsS http://localhost:4200/api/health | head -c 100
```

| Service | Verification |
|---|---|
| MinIO | Open http://localhost:9001 and sign in using credentials configured for your local environment |
| MLflow | Open http://localhost:5000 and confirm the UI loads |
| Langfuse | Open http://localhost:3000 and complete first-run setup |
| Prefect | Open http://localhost:4200 and confirm the dashboard loads |

**Langfuse first run:** Open http://localhost:3000, create an organization and project, then configure the project keys in your local `.env`. Do not paste those keys into this README, screenshots, issue reports, or commit messages.

> **Note:** Health endpoints and startup timing vary by service/version. If a command fails, inspect `docker compose ps` and the relevant service logs rather than assuming a fixed wait time is sufficient.

## 📝 Step 2 — Register LLM prompts in MLflow

Ensure the virtual environment is active and run from the repository root:

```bash
python -c "from app.llm.prompts import register_all_prompts; register_all_prompts()"
```

**Verify:** Open MLflow and check the Prompts section for:

- `hospital_agent_system`
- `churn_insight_generator`
- `service_performance_score`

**Why before evaluation?** Registering prompts first allows downstream evaluation workflows to use versioned prompts.

## 🧩 Step 3 — Configure Feast feature store

```bash
cd app/feast
feast apply
```

Return to the repository root before running the application flows:

```bash
cd ../..
```

To start the Feast UI in a separate terminal:

```bash
cd hospital_production/app/feast
feast ui --port 8888
```

Then open http://localhost:8888.

**Important:** `feast apply` applies the feature-store definitions; it does not, by itself, materialize historical features into an online store. If your workflow requires online features, confirm the configured online store and run the appropriate materialization or ingestion workflow for your Feast version and project.

## 🤖 Step 4 — Train models

From the repository root, with `.venv` active:

```bash
python -m app.flows.train_flow
```

Example log messages may resemble:

```text
Loaded data: (10000, N)
Engineered features: (10000, M)
Training integrity_audit ...
Training service_performance ...
Training staff_morale ...
Training staff_attendance_rate ...
Training risk_score ...
Training risk_level ...
Training churn ...
All configured targets trained and registered.
```

The actual row counts, feature counts, targets, and number of runs depend on your dataset and implementation.

**Verify:**

- MLflow: confirm the expected experiment and runs exist.
- MinIO: confirm the configured model bucket contains the expected model artifacts, such as `models/churn_bundle.joblib`.
- MinIO: confirm expected manifest files exist, if your training flow writes them.

If training fails with `NoSuchBucket`, inspect the configured MinIO endpoint, bucket name, and credentials. If your project provides a bucket-initialization helper, run it from the repository root:

```bash
python -c "from app.ml.minio_store import ensure_bucket; ensure_bucket()"
```

## 🔮 Step 5 — Run predictions

```bash
python -m app.flows.predict_flow
```

Expected behavior: the flow runs a sample prediction and may print fields such as `patient_id`, `churn_pred`, and `churn_prob`, depending on the implementation.

**Why separate training and prediction?** Training is often a scheduled batch workflow, while prediction may run more frequently. Prefect can orchestrate and schedule each workflow independently once deployments and workers are configured.

## 🧪 Step 6 — Evaluate LLM providers

```bash
python -m app.flows.llm_eval_flow
```

Example log messages may resemble:

```text
Registering prompts ... OK
Evaluating huggingface ...
Evaluating groq ...
Evaluating ninerouter ...
LLM evaluation complete.
```

**Verify:** In MLflow, find the evaluation runs (for example, run names containing `llm_eval_`) and compare the metrics your evaluation code records, such as:

- `answer_correctness`
- `relevance`
- `faithfulness`

Metric names and available providers depend on the project implementation and environment configuration.

**Why?** Re-running evaluations after prompt changes helps detect regressions before updated prompts reach users.

## 🚀 Step 7 — Launch the API

```bash
docker compose up -d api
```

Check service status and logs:

```bash
docker compose ps api
docker compose logs --tail=100 api
```

Test the health endpoint:

```bash
curl -fsS http://localhost:8000/health
```

If the endpoint is configured as described by this project, the expected response is similar to:

```json
{"status": "ok"}
```

Test the LLM endpoint:

```bash
curl -X POST http://localhost:8000/llm/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the churn rate for surgery?","role":"ops"}'
```

Explore the API interactively at http://localhost:8000/docs.

> **Security:** Do not put credentials or personal/production patient data into example requests, public logs, screenshots, or issue reports.

## ⚡ Optional — One-shot script (`run_all.sh`)

The following is a convenience template. Review the Compose service names, paths, environment variables, and flow dependencies before using it. It assumes infrastructure becomes ready within a fixed wait, which is not guaranteed; health checks are more reliable.

```bash
#!/usr/bin/env bash
set -euo pipefail

cd hospital_production
source .venv/bin/activate

echo "▶ [1/7] Starting infrastructure"
docker compose up -d minio mlflow langfuse langfuse-db prefect

echo "▶ [2/7] Registering prompts"
python -c "from app.llm.prompts import register_all_prompts; register_all_prompts()"

echo "▶ [3/7] Applying Feast"
(cd app/feast && feast apply)

echo "▶ [4/7] Training models"
python -m app.flows.train_flow

echo "▶ [5/7] Running predictions"
python -m app.flows.predict_flow

echo "▶ [6/7] Evaluating LLM providers"
python -m app.flows.llm_eval_flow

echo "▶ [7/7] Starting API"
docker compose up -d api

echo "✅ Workflow commands completed."
echo "MLflow:   http://localhost:5000"
echo "Langfuse: http://localhost:3000"
echo "MinIO:    http://localhost:9001"
echo "Prefect:  http://localhost:4200"
echo "Feast UI: http://localhost:8888 (start separately)"
echo "API docs: http://localhost:8000/docs"
```

Save as `run_all.sh`, then run:

```bash
chmod +x run_all.sh
./run_all.sh
```

The script does not start the Feast UI, configure Prefect deployments/workers, or guarantee service readiness. Confirm those requirements for your setup.

## 📅 Daily and weekly operations

### 🔁 Retrain (for example, weekly)

```bash
source .venv/bin/activate
python -m app.flows.train_flow
```

### 🔮 Run predictions (for example, daily)

```bash
source .venv/bin/activate
python -m app.flows.predict_flow
```

### 🧪 Run LLM evaluation (for example, weekly)

```bash
source .venv/bin/activate
python -m app.flows.llm_eval_flow
```

### 🗓️ Prefect scheduling

Deployment commands differ across Prefect versions. Check the installed version and use its current deployment and worker workflow before relying on a schedule. For example:

```bash
prefect version
prefect --help
```

Configure the deployment, schedule, and worker/serve process according to the installed Prefect version and project configuration.

## 🛠️ Troubleshooting

| Symptom | Checks and possible fix |
|---|---|
| `NoSuchBucket` during training | Check MinIO endpoint, bucket configuration, and initialization helper |
| MLflow connection refused | Check `docker compose ps`, then inspect `docker compose logs mlflow` |
| `feast apply` reports no repository | Run from `app/feast/` and confirm `feature_store.yaml` exists |
| Langfuse traces are missing | Check local `.env` values and project keys in Langfuse; never publish the keys |
| `ModuleNotFoundError: app` | Run module commands from the `hospital_production/` repository root |
| API cannot reach MinIO | Inside Docker, use the Compose service hostname (often `minio:9000`); from the host, use the published host port (often `localhost:9000`) |
| `predict_proba` fails | Check whether the estimator supports `predict_proba`; handle unsupported estimators in the model pipeline |
| Service starts slowly | Inspect service logs and use health checks rather than relying only on `sleep` |

## 🧹 Teardown

Stop services while retaining named volumes:

```bash
docker compose down
```

Stop services and delete named volumes:

```bash
docker compose down -v
```

> **Warning:** `docker compose down -v` can permanently delete local persisted data, including MinIO and MLflow data stored in named volumes. Confirm what your Compose file mounts before running it.

## ✅ Production-readiness checklist

- [ ] Local `.env` is configured and excluded from Git.
- [ ] `.env.example` contains placeholders only.
- [ ] Infrastructure services are healthy.
- [ ] LLM prompts are registered.
- [ ] Feast configuration is applied and required feature materialization is handled.
- [ ] Training completes and expected artifacts are stored.
- [ ] Predictions are verified.
- [ ] LLM evaluation completes and metrics are reviewed.
- [ ] API health check passes.
- [ ] Logs and screenshots contain no secrets or sensitive patient data.
- [ ] Destructive teardown commands are used only when intended.

---

**Security reminder:** This README intentionally contains no real API keys or passwords. Review the repository and Git history before publishing; `.gitignore` does not remove secrets that were committed previously.
