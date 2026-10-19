import pandas as pd
import numpy as np
import logging
from prefect import flow, task
from app.api.config import settings
from app.ml.preprocess import feature_engineer
from app.ml.registry import load_concept
from sklearn.preprocessing import LabelEncoder
from app.flows._metrics import PREDICT_RUN

# Define logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""Prefect prediction flow"""

@task
def load_and_prepare():
    df = pd.read_parquet(settings.DATA_PATH)
    return feature_engineer(df)

@task
def predict(df, concept, features):
    """Load the model bundle from MinIO and make predictions."""
    bundle = load_concept(concept)

    metric_key = "roc_auc" if bundle["task"] == "classification" else "r2"
    
    # pick best submodel (highgest metrics) - bundle carries metrics
    best_name, best = max(((name, m) for name, m in bundle["models"].items() if m["metrics"].get(metric_key) is not None), key=lambda x: x[1]["metrics"][metric_key])

    pipe = best["pipeline"]
    preds = pipe.predict(df[features])

    # Label encoding for classification tasks
    if bundle["task"] == "classification":
        try:
            le = best.get("label_encoder")
            if le is not None:
                preds = le.inverse_transform(preds)
        except Exception as e:
            logger.warning(f"⚠️ inverse_transform failed: {e}")

    prob = None
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(df[features])

        # Ndim on fetching array logic
        if proba.ndim == 2 and proba.shape[1] == 2:
            # binary -> probabilty of positive class
            prob = proba[:, 1]
        elif proba.ndim == 2 and proba.shape[1] > 2:
            # multi-class -> keep full matrix as list of class probabilities
            prob = proba
        else:
            prob = proba

    # Return to out
    out = df[["patient_id"]].copy()
    out[f"{concept}_pred"] = preds

    if prob is not None:
        if isinstance(prob, np.ndarray) and prob.ndim == 2 and prob.shape[1] > 2:
            # Write one column per class
            for i in range(prob.shape[1]):
                out[f"{concept}_prob_class_{i}"] = prob[:, i]

        else:
            out[f"{concept}_prob"] = prob

    return out

@flow(name="hospital-predict-flow", log_prints=True)
def predict_flow():
    PREDICT_RUN.inc()  # increment Prometheus counter
    df = load_and_prepare()
    churn = predict(df, "patient_churn", 
                    ["age","satisfaction","length_of_stay","available_beds",
                     "patients_request","patients_admitted","patients_refused",
                     "staff_morale","staff_present","total_staff","staff_absent",
                     "capacity_pressure","admission_rate_row","refusal_rate_row",
                     "staff_attendance_rate","service"])
    print(churn.head(50))
    return churn

if __name__ == "__main__":
    predict_flow()