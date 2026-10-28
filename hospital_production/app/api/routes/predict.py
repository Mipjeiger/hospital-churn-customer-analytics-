import logging
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# Modules integration
from app.api.dependencies import get_model_bundle
from app.api.cache import get_cache
from app.api.metrics import ML_PREDICTIONS, REQUEST_LATENCY, RISK_SCORE, MODEL_LOADED
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
        SELECT * FROM {table}
        WHERE patient_id = ANY(%(patient_ids)s)
        """,
        params={"patient_ids": patient_ids},
    )

    return result

def build_prediction_frame(
    rows: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    """Build a DataFrame for prediction using the provided rows and features."""
    result = rows.copy()

    if "staff_absens" in result.columns and "staff_absent" not in result.columns:
        result["staff_absent"] = result["staff_absens"]

    if "capacity_pressure" not in result.columns:
        result["capacity_pressure"] = (result["patient_request"] / (result["available_beds"] + 1))

    if "admission_rate_row" not in result.columns:
        result["admission_rate_row"] = (
            result["patients_admitted"]
            / result["patients_request"].replace(0, np.nan)
        )

    if "refusal_rate_row" not in result.columns:
        result["refusal_rate_row"] = (
            result["patients_refused"]
            / result["patients_request"].replace(0, np.nan)
        )

    if "staff_attendance_rate" not in result.columns:
        result["staff_attendance_rate"] = (
            result["staff_present"]
            / result["total_staff"].replace(0, np.nan)
        )


    result[features] = (result[features].replace([np.inf, -np.inf], np.nan).fillna(0))
    missing = [feature for feature in features if feature not in result.columns]

    if missing:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Required model features are missing.",
                "missing_features": missing,
            }
        )

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

@router.post(
    "/{concept}/batch",
    response_model=PredictionResponse,
)
def predict_batch(
    concept: str,
    request: PredictionRequest,
):
    """Retrieve patient data from PostgreSQL and predict with a MinIO model."""
    stored_concept = resolve_model_concept(concept)

    try:
        bundle = get_model_bundle(stored_concept)

        if bundle is None:
            raise HTTPException(status_code=503, detail=f"Model bundle '{stored_concept}' is not available.")

        rows = fetch_prediction_rows(request.patient_ids)
        if rows.empty:
            raise HTTPException(status_code=404, detail="No patient data found for the provided IDs.")

        # Submodel selection
        submodel = request.submodel

        if submodel is None:
            submodel = max(bundle["models"], key=lambda name: (bundle["models"][name]["metrics"].get(
                "roc_auc" if bundle["task"] == "classification" else "r2", float("-inf"), ) or float("-inf")
                )
            )

        if submodel not in bundle["models"]:
            raise HTTPException(status_code=400, detail=f"Submodel '{submodel}' is not available for concept '{stored_concept}'."
                                f" Available submodels: {list(bundle['models'])}")

        # Build features models prediction and pipeline integration
        features = bundle["features"]
        prediction_frame = build_prediction_frame(rows, features)
        model_entry = bundle["models"][submodel]
        pipeline = model_entry["pipeline"]

        # Split into X and predict
        X = prediction_frame[features]
        predictions = pipeline.predict(X)

        label_encoder = model_entry.get("label_encoder")
        if label_encoder is not None:
            try:
                predictions = label_encoder.inverse_transform(predictions)
            except (TypeError, ValueError):
                logger.warning("Could not decode predictions for model '%s'.", stored_concept)

        # Probabilities for classification tasks
        probabilities: list[float | None] = [None] * len(predictions)
        multi_probabilities: list[list[float] | None] = [None] * len(predictions)

        if hasattr(pipeline, "predict_proba"):
            proba = pipeline.predict_proba(X)

            if proba.ndim == 2 and proba.shape[1] == 2:
                probabilities = [float(value) for value in proba[:, 1]]
            elif proba.ndim == 2:
                multi_probabilities = [[float(value) for value in row] for row in proba]

        output: list[PredictionItem] = []

        for index, prediction in enumerate(predictions):
            output.append(
                PredictionItem(
                    patient_id=str(prediction_frame.iloc[index]["patient_id"]),
                    prediction=prediction.item() if hasattr(prediction, "item") else prediction,
                    probability=probabilities[index],
                    probabilities=multi_probabilities[index],
                    model_user=stored_concept,
                    submodel=submodel
                )
            )

        # Label metrics and return response
        ML_PREDICTIONS.labels(model_name=submodel, prediction_type="batch").inc(len(output))

        return PredictionResponse(
            concept=stored_concept,
            task=bundle["task"],
            count=len(output),
            predictions=output
        )

    except HTTPException:
        raise

    except Exception as e:
        logger.exception("Error during batch prediction for concept '%s': %s", stored_concept)
        raise HTTPException(status_code=500, detail=f"An error occurred during batch prediction: {str(e)}")

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