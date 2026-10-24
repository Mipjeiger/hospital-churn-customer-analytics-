import prometheus_client
from prometheus_client import CollectorRegistry, Counter, Gauge, start_http_server

STREAMLIT_METRICS_PORT = 8502


def _get_registry() -> CollectorRegistry:
    registry = getattr(
        prometheus_client,
        "_hospital_streamlit_registry",
        None,
    )

    if registry is None:
        registry = CollectorRegistry()
        setattr(
            prometheus_client,
            "_hospital_streamlit_registry",
            registry,
        )

    return registry


REGISTRY = _get_registry()


def _get_metric(name: str, factory):
    metric = getattr(
        prometheus_client,
        f"_hospital_streamlit_{name}",
        None,
    )

    if metric is None:
        metric = factory()
        setattr(
            prometheus_client,
            f"_hospital_streamlit_{name}",
            metric,
        )

    return metric


PAGE_VIEWS = _get_metric(
    "page_views",
    lambda: Counter(
        "streamlit_page_views_total",
        "Number of Streamlit script runs",
        registry=REGISTRY,
    ),
)

FILTERED_ROWS = _get_metric(
    "filtered_rows",
    lambda: Gauge(
        "streamlit_filtered_rows",
        "Number of rows in the currently filtered dashboard data",
        registry=REGISTRY,
    ),
)

FILTERED_CHURN_RATE = _get_metric(
    "filtered_churn_rate",
    lambda: Gauge(
        "streamlit_filtered_churn_rate",
        "Churn rate in the currently filtered dashboard data",
        registry=REGISTRY,
    ),
)

FILTERED_AVG_SATISFACTION = _get_metric(
    "filtered_average_satisfaction",
    lambda: Gauge(
        "streamlit_filtered_average_satisfaction",
        "Average satisfaction in the currently filtered dashboard data",
        registry=REGISTRY,
    ),
)


def start_metrics_server() -> None:
    server_started = getattr(
        prometheus_client,
        "_hospital_streamlit_server_started",
        False,
    )

    if server_started:
        return

    start_http_server(
        STREAMLIT_METRICS_PORT,
        addr="0.0.0.0",
        registry=REGISTRY,
    )

    setattr(
        prometheus_client,
        "_hospital_streamlit_server_started",
        True,
    )


def update_dashboard_metrics(
    row_count: int,
    churn_rate: float,
    average_satisfaction: float,
) -> None:
    PAGE_VIEWS.inc()
    FILTERED_ROWS.set(row_count)
    FILTERED_CHURN_RATE.set(churn_rate)
    FILTERED_AVG_SATISFACTION.set(average_satisfaction)