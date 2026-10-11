from prometheus_client import Counter, Histogram, Gauge

"""Prometheus metrics for the Hospital Operations API.

Label contract
    
    REQUEST_COUNT -> endpoint, method, status
    REQUEST_LATENCY -> endpoint
    ML_PREDICTIONS -> model_name, prediction_type
    LLM_RESPONSES -> provider, role
    LLM_FALLBACKS -> from_provider, to_provider
    RISK_SCORE -> patient_id, service
    MODEL_LOADED -> model_name
"""

# HTTP Layers metrics
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total number of requests to the Hospital Operations API",
    ["endpoint", "method", "status"]
)

REQUEST_LATENCY = Histogram(
    "http_request_latency_seconds",
    "Latency of requests to the Hospital Operations API in seconds",
    ["endpoint"]
)

# ML model metrics
ML_PREDICTIONS = Counter(
    "ml_predictions_total",
    "Total number of predictions made by the ML model",
    ["model_name", "prediction_type"]
)

LLM_RESPONSES = Counter(
    "llm_responses_total",
    "Total number of responses from the LLM providers",
    ["provider", "role"]
)

LLM_FALLBACKS = Counter(
    "llm_fallbacks_total",
    "Total number of fallbacks between LLM providers",
    ["from_provider", "to_provider"]
)

RISK_SCORE = Gauge(
    "risk_score",
    "Risk score for a patient based on the ML model prediction",
    ["patient_id", "service"]
)

# Service health metrics
MODEL_LOADED = Gauge(
    "model_loaded",
    "Indicates whether the ML model is loaded and ready for predictions",
    ["model_name"]
)