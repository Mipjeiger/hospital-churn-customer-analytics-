from prometheus_client import Counter, start_http_server

# Start a Prometheus exporter on port 8001 (once per process)
if not getattr(start_http_server, "_started", False):
    start_http_server(8001)
    setattr(start_http_server, "_started", True)

# Counters for Prefect flow executions (default registry)
TRAIN_RUN = Counter(
    "prefect_train_runs_total",
    "Number of Prefect training flow executions",
)

PREDICT_RUN = Counter(
    "prefect_predict_runs_total",
    "Number of Prefect prediction flow executions",
)

LLM_EVAL_RUN = Counter(
    "prefect_llm_eval_runs_total",
    "Number of Prefect LLM‑evaluation flow executions",
)