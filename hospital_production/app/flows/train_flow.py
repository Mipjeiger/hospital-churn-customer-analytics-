import pandas as pd
import logging
from prefect import flow, task
from sklearn.model_selection import train_test_split
from pathlib import Path
from dotenv import load_dotenv

# ---- Load .env BEFORE any app import that reads settings ----
DIR_PATH = Path(__file__).parents[2]
load_dotenv(DIR_PATH / ".env")

from app.api.config import settings
from app.ml.feast_store import FEATURE_NAMES
from app.ml.preprocess import feature_engineer
from app.ml.multimodel import train_multimodel
from app.ml.registry import log_and_register
from app.ml.feast_store import get_training_df

"""Prefect training pipeline — features sourced from Feast."""

# ═══════════ Data loading ═══════════
@task(name="add-new-features")
def add_new_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add new features to the DataFrame before training."""
    df = df.copy()

    # Add health_score as a weighted combination of patient_satisfaction, satisfaction, and staff_morale
    df["health_score"] = (
        (df["patient_satisfaction"] >= 78).astype(int) * 0.5 +
        (df["satisfaction"] >= 78).astype(int) * 0.5 +
        (df["staff_morale"] >= 74).astype(int) * 0.5
    )

    # Add attendance_rate as a ratio of staff_present to total_staff
    df["staff_attendance_rate"] = (
        df["staff_present"] / df["total_staff"].replace(0, pd.NA)
        ).fillna(0.0)

    # Add risk_score as a combination of patients_request, patients_refused, and patients_admitted
    df["risk_score"] = (
        (df["patients_request"] <= df["patients_request"].mean()).astype(float)
        + (df["patients_refused"] >= df["patients_refused"].mean()).astype(float)
        + (df["patients_admitted"] <= df["patients_admitted"].mean()).astype(float)
        + (df["satisfaction"] <= df["satisfaction"].mean()).astype(float)
    ) / 4.0

    # Build risk_level as a categorical label based on risk_score
    df["risk_level"] = pd.cut(df["risk_score"], bins=[-1, 0.25, 0.5, 0.75, 1], labels=["low", "medium", "high", "critical"], include_lowest=True)

    return df

@task(name="load-entity-frame")
def load_entity_frame() -> pd.DataFrame:
    """Load only entity + target columns. Features will come from Feast + any new engineered features."""
    if settings.DATA_SOURCE == "postgres":
        from app.ml.pg_store import load_data
        df = load_data()
        df = add_new_features(df)

    else:
        df = pd.read_parquet(settings.DATA_PATH)
        df = add_new_features(df)

    # Everything Feast + the target-labels need
    keep = [
        "patient_id", "arrival_date", "health_score", "risk_score", "risk_level",
        "patient_churn", "service", "staff_attendance_rate",
        "staff_morale", "patients_request", "patients_admitted", "patients_refused",
        "staff_present", "staff_absens", "total_staff", "length_of_stay",
    ]
    return df[keep].copy()

@task(name="fetch-features-from-feast")
def fetch_features(entity_df: pd.DataFrame) -> pd.DataFrame:
    """Pull historical features from Feast offline store.
    The resulting feature frame is identical to what the API sees at serving time.
    """
    try:
        feats = get_training_df(entity_df)
        # Re-attach target columns Feast doesn't know about
        for col in [
            "patient_churn", "service", "risk_score", "risk_level", "health_score",
            "staff_morale", "staff_attendance_rate",
        ]:
            if col in entity_df.columns:
                feats[col] = entity_df[col].values
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"Feast offline store unavailable, proceeding with raw entity frame: {e}")
        feats = entity_df.copy()
    return feats

# ═══════════ Shared split helper ═══════════
@task(name="split")
def split(X, y, stratify=True):
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if stratify else None)

@task(name="train-multimodel")
def train(X_train, y_train, X_test, y_test, task_type):
    return train_multimodel(X_train, y_train, X_test, y_test, task=task_type)

# ═══════════ Concept tasks ═══════════
@task(name="train-integrity")
def train_integrity(feats: pd.DataFrame):
    """Integrity audit — rule-based label, features pulled from Feast."""
    feats = feats.copy()
    feats["integrity_flagged"] = (
        (feats["patients_refused"] > feats["patients_request"]) |
        (feats["staff_present"] > feats["total_staff"]) |
        (feats["length_of_stay"] < 0)
    ).astype(int)

    available_features = [col for col in FEATURE_NAMES if col in feats.columns]
    X_train, X_test, y_train, y_test = split(feats[available_features], feats["integrity_flagged"])
    res = train(X_train, y_train, X_test, y_test, task_type="classification")
    log_and_register("integrity_audit", res, task="classification", features=available_features)

@task(name="train-service-performance")
def train_service_performance(feats: pd.DataFrame):
    """Service performance — regression on health_score."""
    features = [
        *FEATURE_NAMES,
        "health_score"
    ]
    X_train, X_test, y_train, y_test = split(feats[features], feats["health_score"], stratify=False)
    res = train(X_train, y_train, X_test, y_test, task_type="regression")
    log_and_register("service_performance", res, task="regression", features=features)

@task(name="train-staff")
def train_staff(feats: pd.DataFrame):
    """Two regression targets: staff_morale and staff_attendance_rate."""
    features = FEATURE_NAMES
    for target in ["staff_morale", "staff_attendance_rate"]:
        X_train, X_test, y_train, y_test = split(feats[features], feats[target], stratify=False)
        res = train(X_train, y_train, X_test, y_test, task_type="regression")
        log_and_register(f"staff_{target}", res, task="regression", features=features)

@task(name="train-risk")
def train_risk(feats: pd.DataFrame):
    """risk_score (regression) + risk_level (classification)."""
    features = FEATURE_NAMES

    # Regression
    X_train, X_test, y_train, y_test = split(feats[features], feats["risk_score"], stratify=False)
    res_reg = train(X_train, y_train, X_test, y_test, task_type="regression")
    log_and_register("risk_score", res_reg, task="regression", features=features)

    # Classification — string labels, handled by LabelEncoder inside train_multimodel
    y_cls = feats["risk_level"].astype(str)

    X_train, X_test, y_train, y_test = split(feats[features], y_cls, stratify=True)
    res_clf = train(X_train, y_train, X_test, y_test, task_type="classification")
    log_and_register("risk_level", res_clf, task="classification", features=features)

@task(name="train-churn")
def train_churn(feats: pd.DataFrame):
    """Binary churn classification."""
    features = FEATURE_NAMES
    X_train, X_test, y_train, y_test = split(
        feats[features], feats["patient_churn"].astype(int), stratify=True
    )
    res = train(X_train, y_train, X_test, y_test, task_type="classification")
    log_and_register("patient_churn", res, task="classification", features=features)

from app.flows._metrics import TRAIN_RUN

# ═══════════ Flow entrypoint ═══════════
@flow(name="hospital-training-flow", log_prints=True)
def training_flow():
    TRAIN_RUN.inc()
    # Fetch features
    entity_df = load_entity_frame()
    feats     = fetch_features(entity_df)

    # Train all 5 concepts
    train_integrity(feats)
    train_service_performance(feats)
    train_staff(feats)
    train_risk(feats)
    train_churn(feats)

    print("✅ All 5 concepts trained & registered (features from Feast).")

if __name__ == "__main__":
    training_flow()