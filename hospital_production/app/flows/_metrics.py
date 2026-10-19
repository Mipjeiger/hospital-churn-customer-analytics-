import os
import logging
from prometheus_client import Counter, start_http_server

log = logging.getLogger(__name__)

METRICS_PORT = int(os.getenv("PROMETHEUS_EXPORTER_PORT", 8001))

if os.getenv("ENABLE_PROMETHEUS_EXPORTER", "false").lower() == "true":
    global _exporter_started
    if globals().get('_exporter_started'):
        log.info("Prometheus exporter already started; skipping")
    else:
        try:
            start_http_server(METRICS_PORT)
            _exporter_started = True
            log.info("Prometheus exporter started on port %s", METRICS_PORT)
        except OSError as exc:
            log.warning("Could not start Prometheus exporter: %s", exc)

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
