import pandas as pd
from prefect import flow, task
from app.api.config import DATA_PATH
from app.ml.preprocess import feature_engineer
from app.ml.registry import load_concept

"""Prefect prediction flow"""

@task
def load_and_prepare():
    df = pd.read_parquet(DATA_PATH)
    return feature_engineer(df)

@task
def predict(df, concept, features):
    """Load the model bundle from MinIO and make predictions."""
    bundle = load_concept(concept)
    
    # pick best submodel (highgest metrics) - bundle carries metrics
    best_name, best = max(bundle["models"].items(), key=lambda x: x[1]["metrics"].get("roc_auc" if bundle["task"] == "classification" else "r2", 0))
    preds = best["pipeline"].predict(df[features])
    prob = (best["pipeline"].predict_proba(df[features])[:, 1] if hasattr(best["pipeline"], "predict_proba") else None)
    out = df[["patient_id"]].copy()
    out[f"{concept}_pred"] = preds

    if prob is not None:
        out[f"{concept}_prob"] = prob

    return out

@flow(name="hospital-predict-flow", log_prints=True)
def predict_flow():
    df = load_and_prepare()
    churn = predict(df, "churn", 
                    ["age","satisfaction","length_of_stay","available_beds",
                     "patients_request","patients_admitted","patients_refused",
                     "staff_morale","staff_present","total_staff","staff_absent",
                     "capacity_pressure","admission_rate_row","refusal_rate_row",
                     "staff_attendance_rate","service"])
    print(churn.head())
    return churn

if __name__ == "__main__":
    predict_flow()