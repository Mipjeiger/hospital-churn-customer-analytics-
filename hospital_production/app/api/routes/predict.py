import logging
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from app.api.dependencies import get_model_bundle
from app.api.cache import get_cache
from app.api.metrics import ML_PREDICTIONS, REQUEST_LATENCY, RISK_SCORE, MODEL_LOADED
from app.ml.pg_store import fetch_patient
from app.api.schemas import ChurnRequest, ChurnResponse
from app.ml.feast_store import get_online_row, FEATURE_NAMES

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/predict", tags=["predictions"])

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

    # 3. Load model bundle
    bundle = get_model_bundle("churn")
    X = pd.DataFrame([feat_dict])[bundle["features"]]
    prob = float(bundle["pipeline"].predict_proba(X)[0, 1] if hasattr(bundle["pipeline"], "predict_proba") else bundle["pipeline"].predict(X)[0])
    pred = int(prob >= 0.5)

    # 4. Cache + metrics
    result = {
        "patient_id": request.patient_id,
        "churn_probability": round(prob, 4),
        "churn_prediction": pred,
        "best_submodel": bundle["best_submodel"],
    }
    cache.set(key, result)
    ML_PREDICTIONS.labels(model_name=bundle["best_submodel"], prediction_type="fresh").inc()
    RISK_SCORE.labels(
        patient_id=request.patient_id, 
        service=str(feat_dict.get("service", "unknown"))
    ).set(prob)
    MODEL_LOADED.labels(model_name=bundle["best_submodel"]).set(1)

    return ChurnResponse(**result, cached=False)

@router.get("/features/{patient_id}")
def debug_features(patient_id: str):
    """Debug endpoint — shows exactly what the model sees."""
    try:
        return get_online_row(patient_id)
    except Exception as e:
        raise HTTPException(500, str(e))

@router.get("/features/schema")
def feature_schema():
    return {"features": FEATURE_NAMES}