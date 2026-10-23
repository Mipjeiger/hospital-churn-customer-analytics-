import os
import logging
from prometheus_client import Counter, start_http_server

log = logging.getLogger(__name__)

METRICS_PORT = int(os.getenv("PROMETHEUS_EXPORTER_PORT", "8001"))

def start_prometheus() -> None:
    if os.getenv("ENABLE_PROMETHEUS_EXPORTER", "false").lower() != "true":
        return
    
    try:
        start_http_server(METRICS_PORT)
        log.info(f"Prometheus exporter started on port {METRICS_PORT}")
    except OSError as exc:
        if exc.errno == 98:  # Address already in use
            log.warning(f"Prometheus exporter port {METRICS_PORT} is already in use; skipping start")
        else:
            log.error(f"Failed to start Prometheus exporter on port {METRICS_PORT}: {exc}")
            raise

start_prometheus()

# ---------------------------------------------------------------
# Counters for Prefect flow executions (default registry)
# ---------------------------------------------------------------
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
