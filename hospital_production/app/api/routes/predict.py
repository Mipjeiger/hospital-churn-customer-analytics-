import logging
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# Modules integration
from app.api.dependencies import get_model_bundle
from app.api.cache import get_cache
from app.api.metrics import ML_PREDICTIONS, REQUEST_LATENCY, RISK_SCORE, MODEL_LOADED, CACHE_SIZE
from app.ml.pg_store import fetch_patient, query
from app.api.schemas import ChurnRequest, ChurnResponse, PredictionRequest, PredictionResponse, PredictionItem
from app.ml.feast_store import get_online_row, FEATURE_NAMES
from app.api.config import settings
from app.ml.registry import load_concept

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/predict", tags=["predictions"])

# Helper components
MODEL_ALIASES = {
    "churn": "patient_churn",
    "patient_churn": "patient_churn",
    "risk": "risk_level",
    "risk_level": "risk_level",
    "risk_score": "risk_score",
    "service": "service_performance",
    "service_performance": "service_performance",
    "staff_morale": "staff_staff_morale",
    "staff_staff_morale": "staff_staff_morale",
    "staff_attendance_rate": "staff_staff_attendance_rate",
    "staff_staff_attendance_rate": "staff_staff_attendance_rate",
    "integrity": "integrity_audit",
    "integrity_audit": "integrity_audit",
}

# =============================================
# Function for inference insight at endpoint
# =============================================
def resolve_model_concept(concept: str) -> str:
    """Resolve model concept and return to model aliases"""
    normalized = concept.strip().lower()
    return MODEL_ALIASES.get(normalized, normalized)

def quote_identifier(identifier: str) -> str:
    """Safely quote a PostgreSQL identifier."""
    if not identifier.replace("_", "").isalnum():
        raise ValueError(f"Invalid PostgreSQL identifier: {identifier}")
    return f'"{identifier}"'

def fetch_prediction_rows(patient_ids: list[str]) -> pd.DataFrame:
    """Fetch prediction rows from PostgreSQL for given patient IDs."""
    table = (
        f"{quote_identifier(settings.POSTGRES_SCHEMA)}."
        f"{quote_identifier(settings.POSTGRES_TABLE)}"
    )
    result = query(
        f"""
        SELECT *
        FROM {table}
        WHERE patient_id = ANY(:patient_ids)
        """,
        params={"patient_ids": patient_ids},
    )
    return result

def build_prediction_frame(rows: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Build a DataFrame for prediction using the provided rows and features."""
    result = rows.copy()

    # Features engineering
    if "staff_absens" in result.columns and "staff_absent" not in result.columns:
        result["staff_absent"] = result["staff_absens"]
    if "capacity_pressure" not in result.columns:
        result["capacity_pressure"] = (result["patients_request"] / (result["available_beds"] + 1))
    if "admission_rate_row" not in result.columns:
        result["admission_rate_row"] = (result["patients_admitted"] / result["patients_request"].replace(0, np.nan))
    if "refusal_rate_row" not in result.columns:
        result["refusal_rate_row"] = (result["patients_refused"] / result["patients_request"].replace(0, np.nan))
    if "staff_attendance_rate" not in result.columns:
        result["staff_attendance_rate"] = (result["staff_present"] / result["total_staff"].replace(0, np.nan))
    if "health_score" not in result.columns:
        result["health_score"] = (
            0.40 * (result["satisfaction"] / 100)
            + 0.20 * (1 - result["patient_churn"])
            + 0.20 * result["admission_rate_row"]
            + 0.20 * result["staff_attendance_rate"]
        ) * 100

    missing = [f for f in features if f not in result.columns]

    if missing:
        raise HTTPException(
            status_code=422,
            detail={"message": "Required model features are missing.", "missing_features": missing},
        )
    
    result[features] = result[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return result

# ===== Prediction Endpoint =====
@router.post("/churn", response_model=ChurnResponse)
def predict_churn(request: ChurnRequest):
    """End-to-end: cache -> feature lookup (Postgres) -> model -> response"""
    cache = get_cache()
    key = cache.make_key({"concept": "churn", "patient_id": request.patient_id})

    # 1. Cache hit
    if (hit := cache.get(key)) is not None:
        logger.info(f"Cache hit for patient_id: {request.patient_id}")
        ML_PREDICTIONS.labels(model_name=hit["best_submodel"], prediction_type="cache_hit").inc()
        return ChurnResponse(**hit, cached=True)
    
    # 2. Feature lookup using Feast ONLINE store
    with REQUEST_LATENCY.labels(endpoint="/predict/churn").time():
        try:
            feat_dict = get_online_row(request.patient_id)
        except Exception as e:
            logger.error(f"Feature lookup failed for patient_id: {request.patient_id} | Error: {e}")
            raise HTTPException(status_code=500, detail=f"Feast look up failed: {str(e)}")
        
        if not feat_dict or all(v is None for v in feat_dict.values()):
            raise HTTPException(status_code=404, detail=f"No features found for patient_id: {request.patient_id}")
    
    # 3. Load model bundle – strict validation
    bundle = get_model_bundle("churn")
    if not isinstance(bundle, dict):
        logger.error("Model bundle for 'churn' not found in MinIO.")
        raise HTTPException(status_code=500, detail="Model bundle not available.")

    # ==============================================================
    # Determine model source – single pipeline or multi‑model bundle
    # ==============================================================
    
    # Getting Simple bundle with a single pipeline
    if "pipeline" in bundle and "best_submodel" in bundle:
        pipeline = bundle["pipeline"]
        best_submodel = bundle["best_submodel"]
        features = bundle["features"]

    # Multi‑model bundle – pick the best submodel (fallback to first)    
    elif "models" in bundle:
        models = bundle["models"]
        features = bundle.get("features")
        if not features:
            logger.error("Model bundle missing 'features' entry.")
            raise HTTPException(status_code=500, detail="Model bundle incomplete: missing features.")
        
        # Choose submodel with highest ROC AUC (or any if metric missing)
        best_submodel = max(models, key=lambda n: models[n].get("metrics", {}).get("roc_auc", float("-inf")))
        pipeline = models[best_submodel].get("pipeline")
        if pipeline is None:
            logger.error(f"Pipeline missing for selected submodel '{best_submodel}'.")
            raise HTTPException(status_code=500, detail="Model pipeline not available.")
    else:
        logger.error("Model bundle missing both 'pipeline' and 'models' entries.")
        raise HTTPException(status_code=500, detail="Model bundle incomplete.")

    # Build input DataFrame using declared features
    X = pd.DataFrame([feat_dict])[features]
    
    try:
        prediction = pipeline.predict(X)
        pred = int(prediction[0])

        if not hasattr(pipeline, "predict_proba"):
            prob = float(pred)
        else:
            probabilities = np.asarray(pipeline.predict_proba(X))
            
            if probabilities.ndim != 2 or probabilities.shape[0] == 0:
                raise ValueError(f"Unexpected shape for predict_proba output: {probabilities.shape}")
            if probabilities.shape[1] == 1:
                class_value = getattr(pipeline, "classes_", [pred])[0]
                class_value = int(class_value)
                prob = float(probabilities[0, 0]) if class_value == pred else 1.0 - float(probabilities[0, 0])
            elif probabilities.shape[1] == 2:
                classes = getattr(pipeline, "classes_", [0, 1])
                positive_index = list(classes).index(1)
                prob = float(probabilities[0, positive_index])
            else:
                raise ValueError("Churn model must have binary classification output for predict_proba.")
            
        pred = int(prob >= 0.5)
    except (IndexError, TypeError, ValueError) as e:
        logger.exception("Failed to calculate churn probability for patient_id: %s | Error: %s", request.patient_id, e)
        raise HTTPException(status_code=500, detail="Model prediction failed due to unexpected output format.") from e
    
    result = {
        "patient_id": request.patient_id,
        "churn_probability": round(prob, 4),
        "churn_prediction": pred,
        "best_submodel": best_submodel,
    }
    
    # Metrics and caching
    cache.set(key, result)
    CACHE_SIZE.set(cache.size())
    ML_PREDICTIONS.labels(model_name=best_submodel, prediction_type="fresh").inc()
    RISK_SCORE.labels(patient_id=request.patient_id, service=str(feat_dict.get("service", "unknown"))).set(prob)
    MODEL_LOADED.labels(model_name=best_submodel).set(1)
    return ChurnResponse(**result, cached=False)

# ==== Batch Prediction Endpoint =====
@router.post("/{concept}/batch", response_model=PredictionResponse)
def predict_batch(concept: str, request: PredictionRequest):
    stored_concept = resolve_model_concept(concept)
    bundle = get_model_bundle(stored_concept)
    
    if bundle is None:
        logger.error(f"Model bundle for concept '{stored_concept}' not found in MinIO.")
        raise HTTPException(status_code=500, detail="Model bundle not available.")
    
    # Fetch rows patient data from PostgreSQL
    rows = fetch_prediction_rows(request.patient_ids)
    if rows.empty:
        raise HTTPException(status_code=404, detail="No patient data found for the provided IDs.")

    # Load bundle and determine the submodel to use for prediction
    models = bundle.get("models", {})
    features = bundle["features"]
    task = bundle["task"]

    if not models:
        raise HTTPException(status_code=500, detail=f"No models found for concept '{stored_concept}'.")

    submodel = request.submodel
    if submodel is None:
        metric_name = "roc_auc" if task == "classification" else "r2"
        submodel = max(models, key=lambda n: models[n]["metrics"].get(metric_name) or float("-inf"))

    if submodel not in models:
        raise HTTPException(status_code=400, detail=f"Submodel '{submodel}' is not available. Available submodels: {list(models)}")

    # Perform batch prediction using the selected submodel
    model_entry = models[submodel]
    pipeline = model_entry.get("pipeline")
    prediction_frame = build_prediction_frame(rows, features)

    X = prediction_frame[features]
    try:
        raw_predictions = pipeline.predict(X)
        label_encoder = model_entry.get("label_encoder")

        # Inverse transform the predictions if a label encoder is available
        if label_encoder is not None:
            raw_predictions = label_encoder.inverse_transform(raw_predictions)
        probabilities = [None] * len(raw_predictions)
        multi_probabilities = [None] * len(raw_predictions)
        
        if task == "classification" and hasattr(pipeline, "predict_proba"):
            proba = np.asarray(pipeline.predict_proba(X))
            if proba.ndim == 2 and proba.shape[1] == 2:
                probabilities = [float(v) for v in proba[:, 1]]
            elif proba.ndim == 2 and proba.shape[1] > 2:
                multi_probabilities = [[float(v) for v in row] for row in proba]
    
    except (ValueError, TypeError, IndexError) as exc:
        logger.exception("Batch model prediction failed for concept '%s'.", stored_concept)
        raise HTTPException(status_code=500, detail="Model prediction failed.") from exc
    
    output = [
        PredictionItem(
            patient_id=str(prediction_frame.iloc[i]["patient_id"]),
            prediction=(p.item() if hasattr(p, "item") else p),
            probability=probabilities[i],
            probabilities=multi_probabilities[i],
            model_user=stored_concept,
            submodel=submodel,
        )
        for i, p in enumerate(raw_predictions)
    ]
    ML_PREDICTIONS.labels(model_name=submodel, prediction_type="batch").inc(len(output))
    return PredictionResponse(concept=stored_concept, task=task, count=len(output), predictions=output)

@router.get("/features/{patient_id}")
def debug_features(patient_id: str):
    """Return online Feast features for one patient."""
    try:
        return get_online_row(patient_id)
    except Exception as exc:
        logger.exception("Failed to load features for patient %s", patient_id)
        raise HTTPException(status_code=500, detail="Unable to load online features.") from exc

@router.get("/features/schema")
def feature_schema():
    return {"features": FEATURE_NAMES}
