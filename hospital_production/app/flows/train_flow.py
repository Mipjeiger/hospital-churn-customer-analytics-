import pandas as pd
from prefect import flow, task
from sklearn.model_selection import train_test_split
from app.api.config import DATA_PATH
from app.ml.preprocess import feature_engineer
from app.ml.multimodel import train_multimodel
from app.ml.registry import log_and_register

"""Prefect training pipeline flow for multi-model training."""

# ═══════════ Data: Collect & Prepare ═══════════
@task(name="load-data")
def load_data() -> pd.DataFrame:
    """Load the hospital database from Parquet file."""
    return pd.read_parquet(DATA_PATH)

@task(name="engineer-features")
def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering and preprocessing."""
    return feature_engineer(df)

# ═══════════ Model: Represent Patterns ═══════════
@task(name="split")
def split(X, y, stratify=True):
    """Split the dataset into training and testing sets."""
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if stratify else None)

# ═══════════ Training: Learn & Improve ═══════════
@task(name="train-multimodel")
def train(X_train, y_train, X_test, y_test, task_type):
    return train_multimodel(X_train, y_train, X_test, y_test, task=task_type)

# ═══════════ Five models concept ═══════════
@task
def train_integrity(df):
    """Train the integrity audit model based on rule-based flags."""
    features = ["patients_request","patients_admitted","patients_refused",
                "available_beds","staff_present","staff_absent","total_staff",
                "length_of_stay","satisfaction","staff_morale",
                "admission_rate_row","refusal_rate_row","staff_attendance_rate"]

    # week label from rule-based flags
    df["integrity_flagged"] = (
        (df["patients_refused"] > df["patients_request"]) |
        (df["staff_present"] > df["total_staff"]) |
        (df["length_of_stay"] < 0)
    ).astype(int)

    X_train, X_test, y_train, y_test = split(df[features], df["integrity_flagged"])
    res = train(X_train, y_train, X_test, y_test, task_type="classification")
    log_and_register("integrity_audit", res, task="classification", features=features)

@task
def train_service_performance(df):
    """Train the service performance model based on health score."""
    features = ["satisfaction","patient_churn","patients_request","patients_admitted",
                "patients_refused","available_beds","staff_morale","staff_present",
                "total_staff","length_of_stay","capacity_pressure","service"]
    X_train, X_test, y_train, y_test = split(df[features], df["health_score"], stratify=False)
    res = train(X_train, y_train, X_test, y_test, task_type="regression")
    log_and_register("service_performance", res, task="regression", features=features)

@task
def train_staff(df):
    """Train the staff model based on staff morale."""
    features = ["total_staff","staff_present","staff_absent","patients_request",
                "patients_admitted","available_beds","capacity_pressure",
                "length_of_stay","satisfaction","patient_churn","service"]
    for target in ["staff_morale", "staff_attendance_rate"]:
        X_train, X_test, y_train, y_test = split(df[features], df[target], stratify=False)
        res = train(X_train, y_train, X_test, y_test, task_type="regression")
        log_and_register(f"staff_{target}", res, task="regression", features=features)

@task
def train_risk(df):
    """Train the risk model based on risk score."""
    features = ["satisfaction","length_of_stay","refusal_rate_row","staff_morale",
                "patients_request","patients_refused","available_beds",
                "capacity_pressure","admission_rate_row","staff_attendance_rate","service"]

    # Regression on risk score
    X_train, X_test, y_train, y_test = split(df[features], df["risk_score"], stratify=False)
    res_reg = train(X_train, y_train, X_test, y_test, task_type="regression")
    log_and_register("risk_score", res_reg, task="regression", features=features)

    # Classification on risk level
    y_cls = df["risk_level"].astype(str)
    X_train, X_test, y_train, y_test = split(df[features], y_cls, stratify=True)
    res_clf = train_multimodel(X_train, y_train, X_test, y_test, task="classification")
    log_and_register("risk_level", res_clf, task="classification", features=features)

@task
def train_churn(df):
    """Train the patient churn model based on churn label."""
    features = ["age","satisfaction","length_of_stay","available_beds",
                "patients_request","patients_admitted","patients_refused",
                "staff_morale","staff_present","total_staff","staff_absent",
                "capacity_pressure","admission_rate_row","refusal_rate_row",
                "staff_attendance_rate","service"]
    X_train, X_test, y_train, y_test = split(df[features], df["patient_churn"], stratify=True)
    res = train(X_train, y_train, X_test, y_test, task_type="classification")
    log_and_register("patient_churn", res, task="classification", features=features)

# ═══════════ Flow entrypoint ═══════════
@flow(name="hospital-training-flow", log_prints=True)
def training_flow():
    """Prefect flow to train multiple models for hospital data."""
    df = load_data()
    df = engineer(df)
    train_integrity(df)
    train_service_performance(df)
    train_staff(df)
    train_risk(df)
    train_churn(df)
    print("✅ All 5 concepts trained & registered.")

if __name__ == "__main__":
    training_flow()