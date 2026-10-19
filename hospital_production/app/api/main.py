import app.flows._metrics  # ensure Prometheus exporter starts with FastAPI
import time
import logging
from fastapi import FastAPI, Request
from api.routes import llm, predict
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from contextlib import asynccontextmanager
from app.ml.feast_store import get_store
from app.api.dependencies import get_model_bundle
from app.api.metrics import REQUEST_COUNT, REQUEST_LATENCY, MODEL_LOADED, BUSINESS_KPI
from app.api.config import settings
from app.ml.pg_store import query
from app.ml.registry import load_concept
from app.ml.multimodel import pick_best

# Define logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------
# Healper -> expose the model accuracy for prediction and inference
# ---------------------------------------------
def update_model_accuracy_metrics() -> None:
    """Update the model accuracy metrics for all loaded models."""
    for concept in ["integrity_audit", "risk_level", "patient_churn",
    ]:
        try:
            bundle = load_concept(concept)

            if bundle["task"] != "classification":
                continue  # Skip non-classification tasks and continue to the regression

            _, best_model = pick_best(bundle["models"], bundle["task"])
            accuracy = best_model["metrics"].get("accuracy")

            if accuracy is not None:
                BUSINESS_KPI.labels(kpi_name=f"{concept}_accuracy").set(float(accuracy))

        except Exception as e:
            logger.error(f"❌ Failed to update model accuracy metrics for '{concept}': {e}")

# ---------------------------------------------
# Lifestapn - warm up Feast + models at startup
# ---------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to warm up Feast and load models at startup."""
    # 1. Feast FeatureStore
    try:
        get_store()  # Warm up Feast FeatureStore
        logger.info("✅ Feast FeatureStore initialized successfully.")

        update_model_accuracy_metrics()
        logger.info("Model accuracy metrics updated.")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Feast FeatureStore: {e}")
        logger.exception("Failed to update model accuracy metrics.")

    # 2. Warm load ML models bundles from MinIO
    try:
        for concept in ["churn", "risk_score", "service_performance"]:
            try:
                get_model_bundle(concept)
                logger.info(f"✅ Model bundle for '{concept}' loaded successfully.")
            except Exception as e:
                logger.warning(f"⚠️ Warm-up failed for model bundle '{concept}': {e}")

    except Exception as e:
        logger.error(f"❌ Failed to warm up ML models: {e}")

    yield
    logger.info("🛑 API shutting down")

# Build FastAPI app
app = FastAPI(title="Hospital Operations API", version="1.0.0", lifespan=lifespan)

# ──────────────────────────────────────────────
# Middleware — request metrics + latency header
# ──────────────────────────────────────────────
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    latency = time.time() - start

    endpoint = request.url.path
    REQUEST_COUNT.labels(
        endpoint=endpoint,
        method=request.method,
        status=response.status_code
    ).inc()

    # Request latency in seconds
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency)

    response.headers["X-Process-Time"] = f"{latency:.4f}"
    return response

# Include API routers
app.include_router(llm.router)
app.include_router(predict.router)

# Endpoint helper
@app.get("/health")
def health_check():
    return {"status": "healthy", "message": "Hospital Operations API is running."}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/model/{name}/health")
def model_health(name: str):
    return {"model": name, "loaded": bool(MODEL_LOADED.labels(model_name=name)._value.get())}

# Set the business KPI
@app.get("/business-kpi")
def business_kpi():
    """Endpoint to set a business KPI metrics."""
    table = f'"{settings.POSTGRES_SCHEMA}"."{settings.POSTGRES_TABLE}"'

    # Query the database to get the KPI value
    result = query(
        f"""
        SELECT
            COUNT(*) AS total_patients,
            AVG(satisfaction) AS average_satisfaction,
            AVG(patient_churn::float) AS churn_rate,
            AVG(
                patients_admitted::float
                / NULLIF(patients_request, 0)
            ) AS admission_rate,
            AVG(
                staff_present::float
                / NULLIF(total_staff, 0)
            ) AS staff_attendance_rate
        FROM {table}
        """
    ).iloc[0]

    # create kpis in a dict
    kpis = {
        "total_patients": float(result["total_patients"] or 0),
        "average_satisfaction": float(result["average_satisfaction"] or 0),
        "churn_rate": float(result["churn_rate"] or 0),
        "admission_rate": float(result["admission_rate"] or 0),
        "staff_attendance_rate": float(
            result["staff_attendance_rate"] or 0
        )
    }

    for kpi_name, value in kpis.items():
        BUSINESS_KPI.labels(kpi_name=kpi_name).set(value)

    return kpis

@app.post("/business-kpi/model-accuracy/refresh")
def refresh_model_accuracy():
    update_model_accuracy_metrics()

    return {
        "status": "updated",
        "metrics": [
            "integrity_audit_accuracy",
            "risk_level_accuracy",
            "patient_churn_accuracy",
        ],
    }