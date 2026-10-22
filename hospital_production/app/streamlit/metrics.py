from prometheus_client import Counter, Gauge, start_http_server

STREAMLIT_METRICS_PORT = 8502

# Logic landing page for monitoring on the prometheus
PAGE_VIEWS = Counter(
    "streamlit_page_views_total",
    "Number of Streamlit script runs",
)
FILTERED_ROWS = Gauge(
    "streamlit_filtered_rows",
    "Number of rows in the currently filtered dashboard data",
)
FILTERED_CHURN_RATE = Gauge(
    "streamlit_filtered_churn_rate",
    "Churn rate in the currently filtered dashboard data",
)
FILTERED_AVG_SATISFACTION = Gauge(
    "streamlit_filtered_average_satisfaction",
    "Average satisfaction in the currently filtered dashboard data",
)

def start_metrics_server() -> None:
    start_http_server(STREAMLIT_METRICS_PORT, addr="0.0.0.0")

# ==========================
# Function to call and ensuring that metrics are updated in the prometheus dashboard
# ==========================
def update_dashboard_metrics(
    row_count: int,
    churn_rate: float,
    average_satisfaction: float,
) -> None:
    PAGE_VIEWS.inc()
    FILTERED_ROWS.set(row_count)
    FILTERED_CHURN_RATE.set(churn_rate)
    FILTERED_AVG_SATISFACTION.set(average_satisfaction)
