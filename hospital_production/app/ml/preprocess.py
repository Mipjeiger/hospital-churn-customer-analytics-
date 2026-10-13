import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

"""Shared feature engineering and preprocessing functions for the hospital healthcare project."""

DROP_COLUMNS = ["patient_id","name","arrival_date","departure_date","month","week","age_bin"]

def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Perform feature engineering on the input DataFrame."""
    df = df.copy()

    # Fix typo on staff
    if "staff_absens" in df.columns and "staff_absent" not in df.columns:

        df["staff_absent"] = df["staff_absens"]
        df["arrival_date"] = pd.to_datetime(df["arrival_date"], errors='coerce')
        df["departure_date"] = pd.to_datetime(df["departure_date"], errors='coerce')

        # Featured date columns
        df["arrival_year"] = df["arrival_date"].dt.year
        df["arrival_month"] = df["arrival_date"].dt.month
        df["arrival_week"] = df["arrival_date"].dt.isocalendar().week.astype(int)
        df["arrival_dow"] = df["arrival_date"].dt.dayofweek
        df["is_weekend"] = df["arrival_dow"].isin([5, 6]).astype(int)

        # Featured hospital management
        df["capacity_pressure"] = df["patients_request"] / (df["available_beds"] + 1)
        df["admission_rate_row"] = df["patients_admitted"] / df["patients_request"].replace(0, np.nan)
        df["refusal_rate_row"] = df["patients_refused"] / df["patients_request"].replace(0, np.nan)
        df["staff_attendance_rate"] = df["staff_present"] / df["total_staff"].replace(0, np.nan)

        los_min, los_max = df["length_of_stay"].min(), df["length_of_stay"].max()

        # Estimate risk score on the feature engineering 
        df["risk_score"] = (0.50*((100-df["satisfaction"])/100)
                      + 0.25*((df["length_of_stay"]-los_min)/max(los_max-los_min,1))
                      + 0.20*df["refusal_rate_row"].fillna(0).clip(0,1)
                      + 0.15*((100-df["staff_morale"])/100)) * 100
        df["risk_level"] = pd.cut(df["risk_score"], bins=[-1, 30, 60, 100], labels=["Low", "Medium", "High"])

        # Service grouping
        svc = df.groupby("service").agg(
            avg_sat=("satisfaction", "mean"),
            churn=("patient_churn", "mean"),
            adm_rate=("admission_rate_row", "mean"),
            att=("staff_attendance_rate", "mean"))
        svc["health_score"] = (0.35*svc.avg_sat/100 + 0.25*(1-svc.churn)
                               + 0.20*svc.adm_rate + 0.20*svc.att) * 100
        return df.merge(svc["health_score"], on="service", how="left")

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build a preprocessing pipeline for the input DataFrame."""
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])