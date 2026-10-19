import os
import logging
from prometheus_client import Counter, start_http_server

log = logging.getLogger(__name__)

# ---------------------------------------------------------------
# Configurable exporter port (default 8001). When this module is
# imported we try to start a Prometheus HTTP exporter. If the port
# is already bound (e.g. because the container already started it
# on a previous import) we catch the OSError and simply continue –
# the existing exporter will keep serving metrics.
# ---------------------------------------------------------------
METRICS_PORT = int(os.getenv("PROMETHEUS_EXPORTER_PORT", 8001))

try:
    if not getattr(start_http_server, "_started", False):
        start_http_server(METRICS_PORT)
        setattr(start_http_server, "_started", True)
        log.info(f"Prometheus exporter started on port {METRICS_PORT}")
except OSError as exc:
    # Most common cause: port already in use – the exporter is already
    # running in this process (or another process in the container).
    log.warning(
        f"Could not start Prometheus exporter on port {METRICS_PORT}: {exc}. "
        "If another exporter is already running this warning can be ignored."
    )

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
