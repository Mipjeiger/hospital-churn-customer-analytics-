import json
import pandas as pd
from app.api.config import settings
from app.ml.registry import load_concept

# --- Load dataset ----
df = pd.read_parquet(settings.DATA_PATH)

# -- Helper: pick best submodel from a MinIO bundle ---
def best_pipeline(concept: str, task: str):
    bundle = load_concept(concept)
    key = "roc_auc" if task == "classification" else "r2"
    _, best = max(bundle["models"].items(), key=lambda x: x[1]["metrics"].get(key, 0) or 0)

    return best["pipeline"], bundle["features"]

# ===== Tool functions =====
def age_stats():
    return {
        "mean": float(df["age"].mean()),
        "std": float(df["age"].std()),
        "min": int(df["age"].min()),
        "max": int(df["age"].max())
    }

def churn_rate(service: str = None):
    sub = df if service is None else df[df["service"] == service]

    if sub.empty:
        return {"error": f"No data for service '{service}'"}

    return {
        "service":    service or "all",
        "n_patients": int(len(sub)),
        "churn_rate": round(float(sub.patient_churn.mean()), 4),
    }

def service_summary(service: str):
    sub = df[df["service"] == service]

    if sub.empty:
        return {"error": f"No data for service '{service}'"}

    return {
        "service":           service,
        "n_patients":        int(len(sub)),
        "avg_satisfaction":  round(float(sub.satisfaction.mean()), 2),
        "avg_staff_morale":  round(float(sub.staff_morale.mean()), 2),
        "avg_los":           round(float(sub.length_of_stay.mean()), 2),
        "churn_rate":        round(float(sub.patient_churn.mean()), 4),
    }

def bed_availability(month: int = None):
    sub = df if month is None else df[df["month"] == month]
    return {
        "month":                  month or "all",
        "avg_available_beds":     round(float(sub.available_beds.mean()), 2),
        "avg_patients_admitted":  round(float(sub.patients_admitted.mean()), 2),
        "avg_refused":            round(float(sub.patients_refused.mean()), 2),
    }

def predict_patient_churn(patient_id: str):
    """Predict patient churn by id patient"""
    row = df[df["patient_id"] == patient_id]
    if row.empty:
        return {"error": f"No data for patient_id '{patient_id}'"}

    try:
        pipe, features = best_pipeline("churn", "classification")
    except Exception as e:
        return {"error": f"Failed to load churn model: {e}"}

    X = row[features]
    prob = float(pipe.predict_proba(X)[0, 1]) if hasattr(pipe, "predict_proba") else pipe.predict_proba(X)[:, 1]
    return {
        "patient_id":      patient_id,
        "predicted_churn": int(prob >= 0.5),
        "probability":     round(prob, 4),
    }

def service_health(service: str):
    sub = df[df["service"] == service]
    if sub.empty:
        return {"error": f"No data for service '{service}'"}

    return {
        "service":      service,
        "health_score": round(float(sub.health_score.mean()), 2)
                        if "health_score" in sub.columns else None,
        "n_patients":   int(len(sub)),
    }

def high_risk_patients(top_n: int = 10):
    if "risk_score" not in df.columns:
        return {"error": "risk_score column not found in dataset"}

    top = df.nlargest(top_n, "risk_score")[
        ["patient_id", "service", "age", "risk_score", "risk_level"]
    ]
    return {"top_patients": top.to_dict(orient="records")}

# ═══════════ JSON schemas for the LLM ═══════════
TOOLS = [
    {"type": "function", "function": {
        "name": "get_age_stats",
        "description": "Get age statistics of all patients",
        "parameters": {"type": "object", "properties": {}, "required": []}}},

    {"type": "function", "function": {
        "name": "get_churn_rate",
        "description": "Churn rate overall or for a specific service",
        "parameters": {"type": "object", "properties": {
            "service": {"type": "string",
                        "description": "Optional service name (e.g. surgery)"}},
            "required": []}}},

    {"type": "function", "function": {
        "name": "get_service_summary",
        "description": "Summary metrics of a hospital service",
        "parameters": {"type": "object", "properties": {
            "service": {"type": "string"}}, "required": ["service"]}}},

    {"type": "function", "function": {
        "name": "get_bed_availability",
        "description": "Bed availability, admitted & refused stats",
        "parameters": {"type": "object", "properties": {
            "month": {"type": "integer"}}, "required": []}}},

    {"type": "function", "function": {
        "name": "predict_patient_churn",
        "description": "Predict churn for one patient by ID",
        "parameters": {"type": "object", "properties": {
            "patient_id": {"type": "string"}}, "required": ["patient_id"]}}},

    {"type": "function", "function": {
        "name": "get_service_health",
        "description": "Get composite health score for a service",
        "parameters": {"type": "object", "properties": {
            "service": {"type": "string"}}, "required": ["service"]}}},

    {"type": "function", "function": {
        "name": "get_high_risk_patients",
        "description": "Return the top N highest operational-risk patients",
        "parameters": {"type": "object", "properties": {
            "top_n": {"type": "integer", "default": 10}}, "required": []}}},
]

# ═══════════ Name → callable map ═══════════
TOOL_MAP = {
    "get_age_stats":           age_stats,
    "get_churn_rate":          churn_rate,
    "get_service_summary":     service_summary,
    "get_bed_availability":    bed_availability,
    "predict_patient_churn":   predict_patient_churn,
    "get_service_health":      service_health,
    "get_high_risk_patients":  high_risk_patients,
}