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
from app.ml.minio_store import _client
from minio.error import S3Error

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
        logger.info("✅ Model accuracy metrics updated.")

    except Exception as e:
        logger.error(f"❌ Failed to initialize Feast FeatureStore: {e}")
        logger.exception("Failed to update model accuracy metrics.")

    # 2. Warm load ML models bundles from MinIO
    for concept in ["patient_churn", "risk_score", "service_performance"]:
        try:
            bundle = get_model_bundle(concept)

            if bundle is None:
                logger.warning(f"⚠️ Model bundle for '{concept}' is not loaded.")
                continue

            logger.info(f"✅ Model bundle for '{concept}' loaded successfully.")

        except Exception as e:
            logger.exception(f"❌ Warm-up is Failed to load model bundle for '{concept}': {e}")

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

# Get the list models loaded by MinIO Store
MODEL_NAME_ALIASES = {
    "churn": "patient_churn",
    "patient_churn": "patient_churn",
    "risk": "risk_level",
    "risk_level": "risk_level",
    "risk_score": "risk_score",
    "integrity": "integrity_audit",
    "integrity_audit": "integrity_audit",
    "service": "service_performance",
    "service_performance": "service_performance",
    "staff_attendance_rate": "staff_staff_attendance_rate",
    "staff_morale": "staff_staff_morale",
}
@app.get("/model/{name}/health")
def model_health(name: str):
    """Check whether a model bundle exist"""
    requested_names = name.lower()
    stored_name = MODEL_NAME_ALIASES.get(requested_names, requested_names)

    try:
        bundle = get_model_bundle(stored_name)
        loaded = bundle is not None

    except Exception as e:
        logger.error(f"❌ Error checking model health for '{stored_name}': {e}")
        loaded = False

    # Labels for Prometheus metrics
    MODEL_LOADED.labels(model_name=stored_name).set(1 if loaded else 0)

    return {
        "model": requested_names,
        "stored_name": stored_name,
        "loaded": loaded,
        "object": f"models/{stored_name}_bundle.joblib"
    }

@app.get("/model")
def list_models():
    """Endpoint to list all loaded models."""
    client = _client()
    bucket = settings.MINIO_BUCKET
    loaded_models= []

    try:
        objects = client.list_objects(bucket, prefix="models/", recursive=True)

        # Loop though the objects and filter for model bundles
        for obj in objects:
            object_name = obj.object_name

            if not object_name.endswith("_bundle.joblib"):
                continue  # Skip non-model bundle files

            filename = object_name.removeprefix("models/")
            model_name = filename.removesuffix("_bundle.joblib")

            try:
                bundle = get_model_bundle(model_name)
                loaded = bundle is not None
            except Exception as e:
                logger.error(f"❌ Error checking model health for '{model_name}': {e}")
                loaded = False

            MODEL_LOADED.labels(model_name=model_name).set(1 if loaded else 0)

            # Append models loaded
            loaded_models.append({
                "model": model_name,
                "loaded": loaded,
                "object": object_name,
                "size_bytes": obj.size,
                "last_modified": obj.last_modified.isoformat()
            })            

    except S3Error:
        logger.exception(
            "Failed to list model bundles from MinIO bucket '%s'.",
            bucket,
        )
        raise

    return {
        "bucket": bucket,
        "count": len(loaded_models),
        "models": loaded_models
    }


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