import logging
import pandas as pd
from functools import lru_cache
from feast import FeatureStore

logger = logging.getLogger(__name__)

# Feature list using by feast
FEATURE_SERVICE = "patient_features"

FEATURES = [
    f"{FEATURE_SERVICE}:age",
    f"{FEATURE_SERVICE}:satisfaction",
    f"{FEATURE_SERVICE}:length_of_stay",
    f"{FEATURE_SERVICE}:available_beds",
    f"{FEATURE_SERVICE}:patients_request",
    f"{FEATURE_SERVICE}:patients_admitted",
    f"{FEATURE_SERVICE}:patients_refused",
    f"{FEATURE_SERVICE}:staff_morale",
    f"{FEATURE_SERVICE}:staff_present",
    f"{FEATURE_SERVICE}:total_staff",
    f"{FEATURE_SERVICE}:staff_absent",
    f"{FEATURE_SERVICE}:capacity_pressure",
    f"{FEATURE_SERVICE}:admission_rate_row",
    f"{FEATURE_SERVICE}:refusal_rate_row",
    f"{FEATURE_SERVICE}:staff_attendance_rate",
    f"{FEATURE_SERVICE}:service",
]

# Plain names in the same order (for DataFrame columns)
FEATURE_NAMES = [f.split(":")[1] for f in FEATURES]

@lru_cache(maxsize=1)
def get_store() -> FeatureStore:
    logger.info("Initializing Feast FeatureStore...")
    return FeatureStore(repo_path="app/feast")

# ====================================
# OFFLINE - used by Prefect training flow
# ===================================
def get_training_df(entity_df: pd.DataFrame) -> pd.DataFrame:
    """entity_df MUST contain: patient_id, event_timestamp
    Returns: one row per (patient_id, event_timestamp) with all features columns"""
    entity_df = entity_df.rename(
        columns={"arrival_date": "event_timestamp"}
    ).copy()
    entity_df["event_timestamp"] = pd.to_datetime(entity_df["event_timestamp"])

    df = get_store().get_historical_features(
        entity_df=entity_df[["patient_id", "event_timestamp"]],
        features=FEATURES,
    ).to_df()

    # Strip "patient_features:" prefix from columns
    df.columns = [c.split(":")[-1] for c in df.columns]
    return df

# ====================================
# ONLINE - used by FastAPI inference endpoint Production
# ====================================
def get_online_row(patient_id: str) -> dict:
    """Return a single feature dict for one patient"""
    resp = get_store().get_online_features(
        features=FEATURES,
        entity_rows=[{"patient_id": patient_id}],
    ).to_dict()

    # resp is {feature_name: [value]}, flatten
    return {k.split(":")[-1]: v[0] for k, v in resp.items() if k != "patient_id"}

def get_online_batch(patient_ids: list[str]) -> pd.DataFrame:
    """Bulk fetch - useful for scoring dashboards"""
    resp = get_store().get_online_features(
        features=FEATURES,
        entity_rows=[{"patient_id": pid} for pid in patient_ids],
    ).to_dict()

    # Convert dict-of-lists → DataFrame, strip prefixes
    flat = {k.split(":")[-1]: v for k, v in resp.items()}
    return pd.DataFrame(flat)