import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json as _json
from plotly.subplots import make_subplots
import sys
from pathlib import Path
from sqlalchemy import create_engine, text

# ──────────────────────────────────────────────
# Path setup
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
STREAMLIT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != STREAMLIT_DIR]

from app.api.config import settings
from app.streamlit.metrics import start_metrics_server, update_dashboard_metrics
from app.ml.registry import load_concept
from app.ml.minio_store import _client

# ── NEW: import the two new UI modules
from app.streamlit import predict_ui, charts

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
PALETTE = ["#0F2A44", "#0E7C7B", "#2A9D8F", "#E6A817", "#D64545", "#6B7C93", "#8AB4B5", "#C4D7D1"]
SERVICE_COLORS = {"emergency": "#0F2A44", "surgery": "#0E7C7B", "general_medicine": "#2A9D8F", "ICU": "#D64545"}
RISK_COLORS = {"Low": "#0E7C7B", "Medium": "#E6A817", "High": "#D64545", "Critical": "#9B1C1C"}
CAPACITY_COLORS = {"Low": "#0E7C7B", "Moderate": "#E6A817", "High": "#E67E22", "Critical": "#D64545"}
CHURN_COLORS = {"Retained": "#0E7C7B", "Churned": "#D64545"}
AGE_ORDER = ["0–12", "13–18", "19–30", "31–45", "46–60", "61+"]
SATISFACTION_ORDER = ["0–20", "21–40", "41–60", "61–80", "81–100"]
RISK_ORDER = ["Low", "Medium", "High"]
CAPACITY_ORDER = ["Low", "Moderate", "High", "Critical"]

MODEL_CONCEPTS = [
    "patient_churn", "risk_score", "risk_level", "service_performance",
    "integrity_audit", "staff_staff_morale", "staff_staff_attendance_rate",
]

NAV_PAGES = [
    "Overview", "Patient Analytics", "Patient Management",
    "Churn Analytics", "Hospital Operations", "Staff Analytics",
    "Service Performance", "Operational Risk", "Model Performance",
    "Management Insights", "Data Explorer",
]

# ──────────────────────────────────────────────
# Page config & styles
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Hospital Care Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 1.2rem; padding-bottom: 1rem; }
h1 { font-weight: 700; color: #0F2A44; letter-spacing: -0.02em; }
h2 { color: #0F2A44; font-weight: 700; font-size: 1.35rem; margin-top: 1.6rem; border-bottom: 1px solid #E8EEF3; padding-bottom: 0.4rem; }
h3 { color: #1a3a5c; font-weight: 600; font-size: 1.05rem; margin-top: 1rem; }
.kpi-card { background: white; border: 1px solid #E6EBF0; border-radius: 12px; padding: 16px 18px; box-shadow: 0 1px 3px rgba(15,42,68,0.06); }
.kpi-label { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.07em; color: #6B7C93; font-weight: 600; }
.kpi-value { font-size: 1.55rem; font-weight: 700; color: #0F2A44; margin-top: 4px; }
.kpi-sub { font-size: 0.78rem; color: #8A9BB0; margin-top: 2px; }
.section-desc { color: #5A6D85; font-size: 0.9rem; margin-top: -6px; margin-bottom: 12px; }
.insight-box { background: #F0F7F7; border-left: 4px solid #0E7C7B; border-radius: 8px; padding: 10px 14px; margin: 6px 0; font-size: 0.88rem; color: #1a3a5c; }
.insight-warn { background: #FFF7E6; border-left-color: #E6A817; }
.insight-bad  { background: #FFF0F0; border-left-color: #D64545; }
[data-testid="stSidebar"] { background: #0F2A44; }
[data-testid="stSidebar"] * { color: #E8EEF3; }
[data-testid="stSidebar"] .stRadio label { color: #E8EEF3 !important; }
hr { border-color: #E8EEF3; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Metrics server
# ──────────────────────────────────────────────
@st.cache_resource
def initialize_metrics_server() -> None:
    start_metrics_server()

initialize_metrics_server()


# ──────────────────────────────────────────────
# Data access
# ──────────────────────────────────────────────
@st.cache_resource
def get_database_engine():
    return create_engine(
        f"postgresql+psycopg2://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}"
        f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}",
        pool_pre_ping=True,
    )


def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive date, age, satisfaction, capacity and risk features."""
    for col in ("arrival_date", "departure_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if "staff_absens" in df.columns and "staff_absent" not in df.columns:
        df["staff_absent"] = df["staff_absens"]

    if "arrival_date" in df.columns:
        dt = df["arrival_date"].dt
        df["arrival_year"] = dt.year
        df["arrival_month"] = dt.month
        df["arrival_month_name"] = dt.strftime("%b")
        df["arrival_week"] = dt.isocalendar().week.astype(int)
        df["arrival_day"] = dt.day
        df["arrival_day_of_week"] = dt.day_name()
        df["arrival_dow_num"] = dt.dayofweek
        df["is_weekend"] = df["arrival_dow_num"].isin([5, 6])

    df["age_group"] = pd.cut(df["age"], bins=[-1, 12, 18, 30, 45, 60, 120],
                             labels=AGE_ORDER, include_lowest=True)
    df["satisfaction_bin"] = pd.cut(df["satisfaction"], bins=[-1, 20, 40, 60, 80, 100],
                                    labels=SATISFACTION_ORDER, include_lowest=True)

    df["capacity_pressure"] = df["patients_request"] / (df["available_beds"] + 1)
    df["staff_attendance_rate"] = df["staff_present"] / df["total_staff"].replace(0, np.nan).fillna(0.0)
    df["admission_rate_row"] = df["patients_admitted"] / df["patients_request"].replace(0, np.nan).fillna(0.0)
    df["refusal_rate_row"] = df["patients_refused"] / df["patients_request"].replace(0, np.nan).fillna(0.0)
    df["capacity_pressure_level"] = pd.cut(
        df["capacity_pressure"], bins=[-1, 1, 2, 3, 1000],
        labels=CAPACITY_ORDER, include_lowest=True,
    )

    # Sanitize the derived feature columns to ensure no nofinities leak into model featurs
    feature_cols = [
        "capacity_pressure", "staff_attendance_rate", "admission_rate_row",
        "refusal_rate_row", "satisfaction", "patient_churn"
    ]
    for c in feature_cols:
        if c in df.columns:
            df[c] = df[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    df["health_score"] = (
        0.35 * (df["satisfaction"] / 100)
        + 0.25 * (1 - df["patient_churn"])
        + 0.20 * df["admission_rate_row"].fillna(0)
        + 0.20 * df["staff_attendance_rate"].fillna(0)
    ) * 100

    df["risk_score"] = (
        (df["patients_request"] <= df["patients_request"].mean()).astype(float)
        + (df["patients_refused"] >= df["patients_refused"].mean()).astype(float)
        + (df["patients_admitted"] <= df["patients_admitted"].mean()).astype(float)
        + (df["satisfaction"] <= df["satisfaction"].mean()).astype(float)
    ) / 4.0
    df["risk_level"] = pd.cut(df["risk_score"], bins=[-1, 0.25, 0.5, 0.75, 1],
                              labels=["Low", "Medium", "High", "Critical"], include_lowest=True)
    
    return df

@st.cache_data(show_spinner="Loading hospital dataset...")
def load_data() -> pd.DataFrame:
    schema, table = settings.POSTGRES_SCHEMA, settings.POSTGRES_TABLE
    with get_database_engine().connect() as conn:
        df = pd.read_sql(text(f"SELECT * FROM {schema}.{table}"), conn)
    return _add_features(df)

# ──────────────────────────────────────────────
# Model loading & prediction
# ──────────────────────────────────────────────
@st.cache_resource
def load_bundle_cached(concept: str):
    return load_concept(concept)

@st.cache_data
def load_manifest_cached(concept: str) -> dict:
    c = _client()
    r = c.get_object(settings.MINIO_BUCKET, f"manifest/{concept}.json")
    try:
        return _json.loads(r.read().decode())
    finally:
        r.close()
        r.release_conn()


@st.cache_data(show_spinner=False)
def predict_with_bundle(concept: str, submodel: str, df_hash: str, df: pd.DataFrame) -> pd.DataFrame:
    bundle = load_bundle_cached(concept)
    entry = bundle["models"][submodel]
    features = bundle["features"]

    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features for {concept}: {missing}")

    X = df[features]
    preds = entry["pipeline"].predict(X)

    le = entry.get("label_encoder")
    if le is not None:
        try:
            preds = le.inverse_transform(preds)
        except Exception:
            pass

    out = df[["patient_id"]].copy()
    out[f"{concept}_pred"] = preds
    out[f"{concept}_submodel"] = submodel

    if hasattr(entry["pipeline"], "predict_proba"):
        proba = entry["pipeline"].predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] == 2:
            out[f"{concept}_prob"] = proba[:, 1]
        elif proba.ndim == 2 and proba.shape[1] > 2:
            for i in range(proba.shape[1]):
                out[f"{concept}_prob_class_{i}"] = proba[:, i]
    return out


# ──────────────────────────────────────────────
# Formatting helpers
# ──────────────────────────────────────────────
def safe_div(num, den):
    try:
        return 0.0 if den == 0 or pd.isna(den) or pd.isna(num) else num / den
    except Exception:
        return 0.0

def fmt_pct(x): return f"{x*100:.1f}%" if pd.notna(x) else "—"
def fmt_num(x): return f"{x:,.0f}" if pd.notna(x) else "—"
def fmt_float(x, d=1): return f"{x:.{d}f}" if pd.notna(x) else "—"

def render_kpi(label, value, sub=""):
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )

def render_section_header(title, desc=""):
    st.markdown(f"### {title}")
    if desc:
        st.markdown(f'<div class="section-desc">{desc}</div>', unsafe_allow_html=True)

def render_insight(text, kind="info"):
    cls = "insight-box" + (f" insight-{kind}" if kind != "info" else "")
    st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)

def model_badge(concept: str, df: pd.DataFrame):
    col = f"{concept}_submodel"
    if col in df.columns and df[col].notna().any():
        used = df[col].dropna().unique()
        st.caption(f"🧠 Using: **{concept}** → {', '.join(map(str, used))}")

def empty_guard(df: pd.DataFrame) -> bool:
    if len(df) == 0:
        st.warning("No records match the current filters. Adjust the date, service, age, or churn filters in the sidebar.")
        return True
    return False

def styled_chart(fig, height=300, **kwargs):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), **kwargs)
    st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────
# Sidebar: model manifests & selection
# ──────────────────────────────────────────────
def load_manifests() -> dict[str, dict]:
    manifests = {}
    for concept in MODEL_CONCEPTS:
        try:
            manifests[concept] = load_manifest_cached(concept)
        except Exception:
            pass
    return manifests

def build_model_selection(manifests: dict[str, dict]) -> dict[str, str]:
    """Render per-concept submodel selectors; return {concept: chosen_submodel}."""
    choices = {}
    for concept, m in manifests.items():
        task = m["task"]
        metric_key = "roc_auc" if task == "classification" else "r2"
        submodels, metrics = m["submodels"], m["metrics"]
        best = max(submodels, key=lambda s: metrics.get(s, {}).get(metric_key, -1) or -1)

        labels = {
            (f"{s} ({metric_key}={metrics[s].get(metric_key):.3f})"
             if metrics.get(s, {}).get(metric_key) is not None else s): s
            for s in submodels
        }
        default_label = next(lbl for lbl, name in labels.items() if name == best)
        options = list(labels.keys())

        with st.expander(concept, expanded=False):
            pick = st.selectbox(
                "submodel", options,
                index=options.index(default_label),
                key=f"submodel_{concept}",
                label_visibility="collapsed",
            )
            choices[concept] = labels[pick]
    return choices

def attach_predictions(df: pd.DataFrame, choices: dict[str, str]) -> tuple[pd.DataFrame, dict[str, str]]:
    errors = {}
    frames = []
    for concept, submodel in choices.items():
        try:
            h = f"{concept}_{submodel}_{len(df)}"
            frames.append(predict_with_bundle(concept, submodel, h, df))
        except Exception as e:
            errors[concept] = str(e)
    for pf in frames:
        df = df.merge(pf, on="patient_id", how="left")
    return df, errors

def attach_router_prediction(concept: str, submodel: str, source_df: pd.DataFrame) -> pd.DataFrame | None:
    """Run a selected model against the database rows after Predict is clicked."""
    try:
        prediction_key = f"{concept}_{submodel}_{len(source_df)}"
        prediction_df = predict_with_bundle(
            concept=concept,
            submodel=submodel,
            df_hash=prediction_key,
            df=source_df
        )

        return source_df.merge(
            prediction_df,
            on="patient_id",
            how="left",
        )

    except Exception as e:
        st.error(f"Prediction failed for {concept} with submodel {submodel}: {e}")
        return source_df

def render_prediction_required_message(concept: str) -> None:
    st.info(
        f"Click **Predict** to calculate the `{concept}` model results for the database records and display the graph. "
        "If you have already submitted a form, the chart will update accordingly."
    )

# ──────────────────────────────────────────────
# Load data & build sidebar
# ──────────────────────────────────────────────
try:
    df_raw = load_data()
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

with st.sidebar:
    st.markdown("## 🏥 Hospital Care Analytics")
    st.markdown(
        '<div style="color:#8AB4B5; font-size:0.82rem; margin-top:-8px; margin-bottom:16px;">'
        'Operational & Patient Intelligence</div>',
        unsafe_allow_html=True,
    )

    nav = st.radio("Navigation", NAV_PAGES, label_visibility="collapsed")
    st.markdown("---")
    st.markdown("### Filters")

    min_d = df_raw["arrival_date"].min().date()
    max_d = df_raw["arrival_date"].max().date()
    d_range = st.date_input("Arrival date range", value=(min_d, max_d), min_value=min_d, max_value=max_d)
    start_d, end_d = d_range if isinstance(d_range, tuple) and len(d_range) == 2 else (min_d, max_d)

    all_services = sorted(df_raw["service"].dropna().unique().tolist())
    sel_services = st.multiselect("Service", all_services, default=all_services)

    age_min, age_max = int(df_raw["age"].min()), int(df_raw["age"].max())
    age_range = st.slider("Age range", age_min, age_max, (age_min, age_max))
    churn_opt = st.selectbox("Patient churn", ["All", "Churned", "Retained"])

    st.markdown("---")
    st.caption("Operational and analytical use only. No clinical diagnosis or medical advice.")

    st.markdown("---")
    st.markdown("### 🧠 Models")
    manifests = load_manifests()
    if manifests:
        st.caption(f"{len(manifests)}/{len(MODEL_CONCEPTS)} models loaded.")
        st.markdown(
            "".join(f'<div style="font-size:0.78rem;color:#8AB4B5;">✅ {c}</div>' for c in manifests),
            unsafe_allow_html=True,
        )
    else:
        st.warning("No manifests available.")

    st.markdown("### 🎛️ Model Selection")
    choices = build_model_selection(manifests)

# Apply filters
df = df_raw.copy()
mask = (df["arrival_date"].dt.date >= start_d) & (df["arrival_date"].dt.date <= end_d)
df = df[mask]
df = df[df["service"].isin(sel_services)] if sel_services else df.iloc[0:0]
df = df[(df["age"] >= age_range[0]) & (df["age"] <= age_range[1])]
if churn_opt == "Churned":
    df = df[df["patient_churn"] == 1]
elif churn_opt == "Retained":
    df = df[df["patient_churn"] == 0]

empty = len(df) == 0
update_dashboard_metrics(
    row_count=len(df),
    churn_rate=float(df["patient_churn"].mean()) if not empty else 0.0,
    average_satisfaction=float(df["satisfaction"].mean()) if not empty else 0.0,
)

churn_rate = df["patient_churn"].mean() if not empty else 0.0
admission_rate = safe_div(df["patients_admitted"].sum(), df["patients_request"].sum())
refusal_rate = safe_div(df["patients_refused"].sum(), df["patients_request"].sum())
staff_att = safe_div(df["staff_present"].sum(), df["total_staff"].sum())


# ──────────────────────────────────────────────
# Pages
# ──────────────────────────────────────────────
def page_overview():
    st.markdown("# Hospital Care Analytics")
    st.markdown(
        '<div class="section-desc">Monitor patient experience, hospital capacity, staff performance, '
        'and patient churn — filtered data updates every panel.</div>',
        unsafe_allow_html=True,
    )

    row1 = st.columns(5)
    kpis1 = [
        ("Total Patients", fmt_num(len(df)), f"of {len(df_raw):,} total"),
        ("Churned", fmt_num(df["patient_churn"].sum()) if not empty else "0", fmt_pct(churn_rate) + " churn rate"),
        ("Avg Satisfaction", fmt_float(df["satisfaction"].mean()) if not empty else "—", "/ 100"),
        ("Avg Length of Stay", fmt_float(df["length_of_stay"].mean()) + " d" if not empty else "—", ""),
        ("Avg Staff Morale", fmt_float(df["staff_morale"].mean()) if not empty else "—", "/ 100"),
    ]
    for col, (label, value, sub) in zip(row1, kpis1):
        with col:
            render_kpi(label, value, sub)

    row2 = st.columns(5)
    kpis2 = [
        ("Admission Rate", fmt_pct(admission_rate) if not empty else "—", f"{fmt_num(df['patients_admitted'].sum())} admitted" if not empty else ""),
        ("Refusal Rate", fmt_pct(refusal_rate) if not empty else "—", f"{fmt_num(df['patients_refused'].sum())} refused" if not empty else ""),
        ("Available Beds (avg)", fmt_float(df["available_beds"].mean()) if not empty else "—", "per day"),
        ("Staff Attendance", fmt_pct(staff_att) if not empty else "—", f"{fmt_num(df['staff_present'].sum())} / {fmt_num(df['total_staff'].sum())}"),
        ("Capacity Pressure (avg)", fmt_float(df["capacity_pressure"].mean(), 2) if not empty else "—", ""),
    ]
    for col, (label, value, sub) in zip(row2, kpis2):
        with col:
            render_kpi(label, value, sub)

    if "patient_churn_prob" in df.columns:
        c1, c2 = st.columns(2)
        with c1:
            render_kpi("Predicted Churn (avg)", f"{df['patient_churn_prob'].mean():.1%}", "")
        with c2:
            render_kpi("Predicted Churned (≥0.5)", fmt_num((df["patient_churn_prob"] > 0.5).sum()),
                       f"vs {fmt_num(df['patient_churn'].sum())} observed")

    if empty_guard(df):
        return

    left, right = st.columns(2)
    with left:
        render_section_header("Patient Volume Over Time", "Daily arrivals")
        daily = df.groupby(df["arrival_date"].dt.date).size().reset_index(name="patients")
        daily.columns = ["date", "patients"]
        fig = px.line(daily, x="date", y="patients")
        fig.update_traces(line_color="#0F2A44", line_width=2)
        styled_chart(fig, height=300, xaxis_title="", yaxis_title="Patients")
    with right:
        render_section_header("Churn Trend", "Monthly churn rate")
        monthly = (df.groupby(df["arrival_date"].dt.to_period("M"))
                   .agg(patients=("patient_id", "count"), churned=("patient_churn", "sum"))
                   .reset_index())
        monthly["arrival_date"] = monthly["arrival_date"].astype(str)
        monthly["churn_rate"] = monthly["churned"] / monthly["patients"]
        fig = px.line(monthly, x="arrival_date", y="churn_rate", markers=True)
        fig.update_traces(line_color="#D64545")
        styled_chart(fig, height=300, xaxis_title="", yaxis_title="Churn rate", yaxis_tickformat=".0%")

    c1, c2 = st.columns(2)
    with c1:
        render_section_header("Patients by Service")
        svc = df["service"].value_counts().reset_index()
        svc.columns = ["service", "patients"]
        fig = px.bar(svc, x="service", y="patients", color="service",
                     color_discrete_map=SERVICE_COLORS, text="patients")
        fig.update_traces(textposition="outside")
        styled_chart(fig, height=300, showlegend=False, xaxis_title="", yaxis_title="Patients")
    with c2:
        render_section_header("Churn Rate by Service")
        svc2 = (df.groupby("service")
                .agg(patients=("patient_id", "count"), churn_rate=("patient_churn", "mean"))
                .reset_index().sort_values("churn_rate", ascending=False))
        fig = px.bar(svc2, x="service", y="churn_rate", color="service",
                     color_discrete_map=SERVICE_COLORS, text_auto=".1%")
        styled_chart(fig, height=300, showlegend=False, xaxis_title="", yaxis_title="Churn rate", yaxis_tickformat=".0%")

    c1, c2 = st.columns(2)
    with c1:
        render_section_header("Satisfaction vs Churn")
        sat = df.groupby("patient_churn")["satisfaction"].mean().reset_index()
        sat["status"] = sat["patient_churn"].map({0: "Retained", 1: "Churned"})
        fig = px.bar(sat, x="status", y="satisfaction", color="status",
                     color_discrete_map=CHURN_COLORS, text_auto=".1f")
        styled_chart(fig, height=300, showlegend=False, xaxis_title="", yaxis_title="Avg satisfaction")
    with c2:
        render_section_header("Capacity Trend", "Available beds over time")
        cap = (df.groupby(df["arrival_date"].dt.to_period("M"))
               .agg(beds=("available_beds", "mean")).reset_index())
        cap["arrival_date"] = cap["arrival_date"].astype(str)
        fig = px.area(cap, x="arrival_date", y="beds")
        fig.update_traces(line_color="#0E7C7B", fillcolor="rgba(14,124,123,0.15)")
        styled_chart(fig, height=300, xaxis_title="", yaxis_title="Avg available beds")

    render_section_header("Patient Requests vs Admissions", "Monthly totals")
    req = (df.groupby(df["arrival_date"].dt.to_period("M"))
           .agg(requests=("patients_request", "sum"), admitted=("patients_admitted", "sum"),
                refused=("patients_refused", "sum")).reset_index())
    req["arrival_date"] = req["arrival_date"].astype(str)
    fig = go.Figure()
    fig.add_bar(x=req["arrival_date"], y=req["requests"], name="Requests", marker_color="#0F2A44")
    fig.add_bar(x=req["arrival_date"], y=req["admitted"], name="Admitted", marker_color="#0E7C7B")
    fig.add_bar(x=req["arrival_date"], y=req["refused"], name="Refused", marker_color="#D64545")
    styled_chart(fig, height=320, barmode="group", xaxis_title="", yaxis_title="Patients",
                 legend=dict(orientation="h", y=1.02))

    render_section_header("Key Insights")
    svc_churn = df.groupby("service")["patient_churn"].mean()
    if len(svc_churn) > 1:
        top_svc, top_val = svc_churn.idxmax(), svc_churn.max()
        if top_val > churn_rate + 0.03:
            render_insight(f"<b>{top_svc}</b> has the highest churn rate at {top_val:.1%}, "
                           f"{(top_val - churn_rate) * 100:.1f} pp above the filtered average ({churn_rate:.1%}).")
        else:
            render_insight("No service shows a churn rate more than 3 pp above the filtered average.")

    if df["patient_churn"].nunique() > 1:
        gap = (df[df["patient_churn"] == 1]["satisfaction"].mean()
               - df[df["patient_churn"] == 0]["satisfaction"].mean())
        if abs(gap) >= 1:
            render_insight(f"Churned patients have {'lower' if gap < 0 else 'higher'} satisfaction "
                           f"({gap:+.1f} pts vs retained).")

    avg_cp = df["capacity_pressure"].mean()
    if avg_cp >= 2:
        render_insight(f"Capacity pressure is elevated (avg {avg_cp:.2f} — High).", kind="warn")
    elif avg_cp >= 1:
        render_insight(f"Capacity pressure is moderate (avg {avg_cp:.2f}).")

    if staff_att < 0.75:
        render_insight(f"Staff attendance is low at {staff_att:.1%}. Check rostering and absenteeism drivers.", kind="warn")


def page_patient_analytics():
    st.markdown("# Patient Analytics")
    st.markdown('<div class="section-desc">Population structure, age groups, length of stay, satisfaction and service mix.</div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: render_kpi("Total Patients", fmt_num(len(df)))
    with c2: render_kpi("Average Age", fmt_float(df["age"].mean()) + " yrs" if not empty else "—")
    with c3: render_kpi("Avg Length of Stay", fmt_float(df["length_of_stay"].mean()) + " d" if not empty else "—")
    with c4: render_kpi("Avg Satisfaction", fmt_float(df["satisfaction"].mean()) if not empty else "—")
    with c5: render_kpi("Churn Rate", fmt_pct(churn_rate) if not empty else "—")

    if empty_guard(df):
        return

    left, right = st.columns(2)
    with left:
        render_section_header("Age Distribution")
        fig = px.histogram(df, x="age", nbins=30, color_discrete_sequence=["#0F2A44"])
        styled_chart(fig, height=300, xaxis_title="Age", yaxis_title="Patients")
    with right:
        render_section_header("Age Group Analysis")
        ag = (df.groupby("age_group", observed=True)
              .agg(patients=("patient_id", "count"), churn_rate=("patient_churn", "mean"),
                   avg_sat=("satisfaction", "mean")).reset_index())
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_bar(x=ag["age_group"].astype(str), y=ag["patients"], name="Patients", marker_color="#0F2A44")
        fig.add_trace(go.Scatter(x=ag["age_group"].astype(str), y=ag["churn_rate"],
                                 name="Churn rate", mode="lines+markers",
                                 line=dict(color="#D64545", width=3)), secondary_y=True)
        fig.update_layout(height=320, legend=dict(orientation="h", y=1.02))
        fig.update_yaxes(title_text="Patients", secondary_y=False)
        fig.update_yaxes(title_text="Churn rate", tickformat=".0%", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(ag.style.format({"churn_rate": "{:.1%}", "avg_sat": "{:.1f}"}),
                     use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        render_section_header("Length of Stay — Histogram")
        fig = px.histogram(df, x="length_of_stay", nbins=14, color_discrete_sequence=["#0E7C7B"])
        styled_chart(fig, height=300, xaxis_title="Length of stay (days)", yaxis_title="Patients")
    with c2:
        render_section_header("Length of Stay — Box Plot by Churn")
        fig = px.box(df, x=df["patient_churn"].map({0: "Retained", 1: "Churned"}),
                     y="length_of_stay", color=df["patient_churn"].map({0: "Retained", 1: "Churned"}),
                     color_discrete_map=CHURN_COLORS)
        styled_chart(fig, height=300, showlegend=False, xaxis_title="", yaxis_title="Length of stay (days)")

    c1, c2 = st.columns(2)
    with c1:
        render_section_header("Satisfaction Distribution")
        fig = px.histogram(df, x="satisfaction", nbins=20, color_discrete_sequence=["#2A9D8F"])
        styled_chart(fig, height=300, xaxis_title="Satisfaction", yaxis_title="Patients")
    with c2:
        render_section_header("Service Distribution")
        svc = df["service"].value_counts().reset_index()
        svc.columns = ["service", "patients"]
        fig = px.pie(svc, names="service", values="patients", hole=0.55,
                     color="service", color_discrete_map=SERVICE_COLORS)
        styled_chart(fig, height=300)

    # ── NEW: manual prediction form + bulk charts via charts.py
    render_section_header(
        "Patient Churn Prediction",
        "Click Predict to run the patient-churn model against the database records.",
    )

    patient_prediction = None

    if "patient_churn" in choices:
        patient_prediction = predict_ui.prediction_form(
            concept="patient_churn",
            submodel=choices["patient_churn"],
            defaults={
                "age": 45,
                "satisfaction": 70,
                "length_of_stay": 5,
                "available_beds": 20,
                "patients_request": 40,
                "patients_admitted": 30,
                "patients_refused": 10,
                "staff_morale": 70,
                "staff_present": 25,
                "total_staff": 30,
                "staff_absent": 5,
                "capacity_pressure": 40 / 21,
                "admission_rate_row": 30 / 40,
                "refusal_rate_row": 10 / 40,
                "staff_attendance_rate": 25 / 30,
                "service": "surgery",
            },
            form_key="form_churn_pa",
            services=sorted(df_raw["service"].dropna().unique().tolist()),
        )

    if patient_prediction is None:
        render_prediction_required_message("patient_churn")
    else:
        patient_prediction_df = attach_router_prediction(
            concept="patient_churn",
            submodel=choices["patient_churn"],
            source_df=df
        )

        if patient_prediction_df is not None:
            render_section_header(
                "Database Churn Predictions",
                "The graph is based on patient records loaded from PostgreSQL.",
            )

            model_badge(
                "patient_churn",
                patient_prediction_df,
            )

            c1, c2 = st.columns(2)

            with c1:
                charts.churn_distribution(
                    patient_prediction_df,
                    CHURN_COLORS,
                )

            with c2:
                charts.churn_confusion(
                    patient_prediction_df,
                )

    render_section_header("Bulk Predictions on Filtered Data")
    model_badge("patient_churn", df)
    c1, c2 = st.columns(2)
    with c1:
        charts.churn_distribution(df, CHURN_COLORS)
    with c2:
        charts.churn_confusion(df)
    m = manifests.get("patient_churn")
    if m and "patient_churn_submodel" in df.columns and df["patient_churn_submodel"].notna().any():
        used = df["patient_churn_submodel"].dropna().iloc[0]
        met = m["metrics"].get(used, {})
        st.caption(f"Submodel **{used}** — accuracy {met.get('accuracy', 0):.3f}, "
                   f"F1 {met.get('f1', 0):.3f}, ROC-AUC {met.get('roc_auc', 0):.3f}")

    render_section_header("Key Insights")
    ag2 = df.groupby("age_group", observed=True)["patient_churn"].mean()
    if len(ag2) > 0:
        spread = ag2.max() - ag2.min()
        if spread >= 0.05:
            render_insight(f"Churn varies by age group (spread {spread:.1%}). "
                           f"Highest: {ag2.idxmax()} ({ag2.max():.1%}), lowest: {ag2.idxmin()} ({ag2.min():.1%}).")
    if df["patient_churn"].nunique() > 1:
        los_gap = (df[df["patient_churn"] == 1]["length_of_stay"].mean()
                   - df[df["patient_churn"] == 0]["length_of_stay"].mean())
        if abs(los_gap) >= 0.5:
            render_insight(f"Length of stay differs by {los_gap:+.1f} days between churned and retained patients.")


def page_patient_management():
    st.markdown("# Patient Management")
    st.markdown('<div class="section-desc">Search, filter, and prioritise patients. '
                'Risk score is a heuristic — not a clinical prediction.</div>', unsafe_allow_html=True)
    st.info("Operational risk score is a heuristic for analytical prioritisation. Not a clinical prediction.")

    if empty_guard(df):
        return
    
    render_section_header(
        "Risk-Level Prediction",
        "Click Predict to run the risk-level model against the database records."
    )

    risk_prediction = None

    if "risk_level" in choices:
        risk_prediction = predict_ui.prediction_form(
            concept="risk_level",
            submodel=choices["risk_level"],
            defaults={
                "age": 55,
                "satisfaction": 60,
                "length_of_stay": 7,
                "available_beds": 15,
                "patients_request": 50,
                "patients_admitted": 35,
                "patients_refused": 15,
                "staff_morale": 65,
                "staff_present": 22,
                "total_staff": 30,
                "staff_absent": 8,
                "capacity_pressure": 50 / 16,
                "admission_rate_row": 35 / 50,
                "refusal_rate_row": 15 / 50,
                "staff_attendance_rate": 22 / 30,
                "service": "emergency",
            },
            form_key="form_risk_pm",
            services=sorted(df_raw["service"].dropna().unique().tolist()),
        )

    if risk_prediction is None:
        render_prediction_required_message("risk_level")
    else:
        risk_prediction_df = attach_router_prediction(
            concept="risk_level",
            submodel=choices["risk_level"],
            source_df=df,
        )

        if risk_prediction_df is not None:
            st.subheader("Database Risk-Level Predictions")

            risk_counts = (
                risk_prediction_df["risk_level_pred"]
                .dropna()
                .astype(str)
                .str.title()
                .value_counts()
                .reindex(RISK_ORDER, fill_value=0)
                .rename_axis("risk_level")
                .reset_index(name="patients")
            )

            fig = px.bar(
                risk_counts,
                x="risk_level",
                y="patients",
                color="risk_level",
                color_discrete_map=RISK_COLORS,
                text="patients",
            )

            fig.update_traces(textposition="outside")

            styled_chart(
                fig,
                height=320,
                showlegend=False,
                xaxis_title="Predicted risk level",
                yaxis_title="Patients",
            )

    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    with c1: q = st.text_input("Search patient (ID or name)", placeholder="e.g. PAT-… or Richard")
    with c2: f_service = st.multiselect("Service", sorted(df["service"].unique().tolist()), default=[])
    with c3: f_churn = st.selectbox("Churn status", ["All", "Churned", "Retained"])
    with c4: f_risk = st.selectbox("Risk level", ["All", "Low", "Medium", "High"])

    c5, c6, c7 = st.columns(3)
    with c5: f_sat = st.slider("Satisfaction range", 0, 100, (0, 100))
    with c6:
        los_min, los_max = int(df["length_of_stay"].min()), int(df["length_of_stay"].max())
        f_los = st.slider("Length of stay (days)", los_min, los_max, (los_min, los_max))
    with c7:
        f_age = st.slider("Age", int(df["age"].min()), int(df["age"].max()),
                          (int(df["age"].min()), int(df["age"].max())))

    tdf = df.copy()
    if q:
        ql = q.lower()
        tdf = tdf[tdf["patient_id"].str.lower().str.contains(ql, na=False)
                  | tdf["name"].str.lower().str.contains(ql, na=False)]
    if f_service:
        tdf = tdf[tdf["service"].isin(f_service)]
    if f_churn == "Churned":
        tdf = tdf[tdf["patient_churn"] == 1]
    elif f_churn == "Retained":
        tdf = tdf[tdf["patient_churn"] == 0]
    if f_risk != "All":
        tdf = tdf[tdf["risk_level"] == f_risk]
    tdf = tdf[(tdf["satisfaction"] >= f_sat[0]) & (tdf["satisfaction"] <= f_sat[1])]
    tdf = tdf[(tdf["length_of_stay"] >= f_los[0]) & (tdf["length_of_stay"] <= f_los[1])]
    tdf = tdf[(tdf["age"] >= f_age[0]) & (tdf["age"] <= f_age[1])].sort_values("risk_score", ascending=False)

    st.markdown(f"**{len(tdf):,} patients** match the table filters (sorted by risk score).")

    show_cols = ["patient_id", "name", "age", "service", "arrival_date", "departure_date",
                 "length_of_stay", "satisfaction", "patient_churn", "risk_score", "risk_level",
                 "capacity_pressure_level"]
    labels = ["Patient ID", "Name", "Age", "Service", "Arrival", "Departure", "LOS",
              "Satisfaction", "Churn", "Risk Score (heuristic)", "Risk Level (heuristic)", "Capacity"]

    extra = [(c, lbl) for c, lbl in [("risk_level_pred", "Risk Level (model)"),
                                     ("patient_churn_prob", "Churn Prob (model)")] if c in tdf.columns]
    for c, lbl in extra:
        show_cols.append(c)
        labels.append(lbl)

    pretty = tdf[[c for c in show_cols if c in tdf.columns]].copy()
    pretty.columns = labels[:len(pretty.columns)]
    pretty["Churn"] = pretty["Churn"].map({0: "Retained", 1: "Churned"})
    pretty["Risk Score (heuristic)"] = pretty["Risk Score (heuristic)"].round(1)
    st.dataframe(pretty, use_container_width=True, height=420, hide_index=True)

    model_badge("risk_level", df)
    model_badge("patient_churn", df)

    csv = pretty.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered patients (CSV)", csv, file_name="filtered_patients.csv", mime="text/csv")

    render_section_header("Patients Requiring Attention")
    att_mask = tdf["patient_churn"] == 1
    if "risk_level_pred" in tdf.columns:
        att_mask |= tdf["risk_level_pred"].astype(str).str.lower() == "high"
    attention = tdf[att_mask].sort_values(
        "patient_churn_prob" if "patient_churn_prob" in tdf.columns else "risk_score",
        ascending=False,
    ).head(20)

    att_cols = [c for c in ["patient_id", "name", "age", "service", "satisfaction", "length_of_stay",
                             "patient_churn", "patient_churn_prob", "risk_level_pred"] if c in attention.columns]
    att = attention[att_cols].copy()
    att.columns = ["Patient ID", "Name", "Age", "Service", "Satisfaction", "LOS",
                   "Observed Churn", "Predicted Churn Prob", "Predicted Risk"][:len(att.columns)]
    if "Observed Churn" in att.columns:
        att["Observed Churn"] = att["Observed Churn"].map({0: "Retained", 1: "Churned"})
    st.dataframe(att, use_container_width=True, hide_index=True)


def page_churn_analytics():
    st.markdown("# Churn Analytics")
    st.markdown('<div class="section-desc">Observed patient churn — analyse drivers and high-risk cohorts.</div>',
                unsafe_allow_html=True)

    cols = st.columns(6)
    with cols[0]: render_kpi("Churn Rate", fmt_pct(churn_rate) if not empty else "—")
    with cols[1]: render_kpi("Churned", fmt_num(df["patient_churn"].sum()) if not empty else "—")
    with cols[2]: render_kpi("Retained", fmt_num((df["patient_churn"] == 0).sum()) if not empty else "—")
    if not empty:
        svc_churn = df.groupby("service")["patient_churn"].mean()
        top_svc = svc_churn.idxmax() if len(svc_churn) > 0 else "—"
        with cols[3]: render_kpi("Highest Churn Service", top_svc, fmt_pct(svc_churn.max()) if len(svc_churn) > 0 else "—")
        with cols[4]: render_kpi("Avg Sat (Churned)",
                                 fmt_float(df[df["patient_churn"] == 1]["satisfaction"].mean())
                                 if (df["patient_churn"] == 1).any() else "—")
        with cols[5]: render_kpi("Avg LOS (Churned)",
                                 fmt_float(df[df["patient_churn"] == 1]["length_of_stay"].mean()) + " d"
                                 if (df["patient_churn"] == 1).any() else "—")

    if empty_guard(df):
        return

    render_section_header(
        "Churn Prediction",
        "Click Predict to calculate model predictions for database records.",
    )

    churn_prediction = None

    if "patient_churn" in choices:
        churn_prediction = predict_ui.prediction_form(
            concept="patient_churn",
            submodel=choices["patient_churn"],
            defaults={
                "age": 35,
                "satisfaction": 55,
                "length_of_stay": 4,
                "available_beds": 25,
                "patients_request": 30,
                "patients_admitted": 22,
                "patients_refused": 8,
                "staff_morale": 60,
                "staff_present": 20,
                "total_staff": 28,
                "staff_absent": 8,
                "capacity_pressure": 30 / 26,
                "admission_rate_row": 22 / 30,
                "refusal_rate_row": 8 / 30,
                "staff_attendance_rate": 20 / 28,
                "service": "general_medicine",
            },
            form_key="form_churn_ca",
            services=sorted(df_raw["service"].dropna().unique().tolist()),
        )

    if churn_prediction is None:
        render_prediction_required_message("patient_churn")
    else:
        churn_prediction_df = attach_router_prediction(
            concept="patient_churn",
            submodel=choices["patient_churn"],
            source_df=df,
        )

        if churn_prediction_df is not None:
            c1, c2 = st.columns(2)

            with c1:
                render_section_header(
                    "Predicted Churn Distribution",
                    "Model result for the filtered database records.",
                )
                charts.churn_distribution(
                    churn_prediction_df,
                    CHURN_COLORS,
                )

            with c2:
                render_section_header(
                    "Predicted Churn Confusion",
                    "Observed database churn compared with model predictions.",
                )
                charts.churn_confusion(
                    churn_prediction_df,
                )
    
    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        render_section_header("Churn Distribution")
        cnt = df["patient_churn"].value_counts().reset_index()
        cnt.columns = ["churn", "n"]
        cnt["label"] = cnt["churn"].map({0: "Retained", 1: "Churned"})
        fig = px.pie(cnt, names="label", values="n", hole=0.62,
                     color="label", color_discrete_map=CHURN_COLORS)
        fig.update_traces(textinfo="percent+value", textposition="inside")
        styled_chart(fig, height=300)
    with c2:
        render_section_header("Churn by Service")
        svc = (df.groupby("service")
               .agg(patients=("patient_id", "count"), churn_rate=("patient_churn", "mean"))
               .reset_index().sort_values("churn_rate", ascending=False))
        fig = px.bar(svc, x="service", y="churn_rate", color="service",
                     color_discrete_map=SERVICE_COLORS, text_auto=".1%")
        fig.update_traces(textposition="outside")
        styled_chart(fig, height=300, showlegend=False, yaxis_tickformat=".0%",
                     xaxis_title="", yaxis_title="Churn rate")

    c1, c2 = st.columns(2)
    with c1:
        render_section_header("Churn by Age Group")
        ag = (df.groupby("age_group", observed=True)
              .agg(patients=("patient_id", "count"), churn_rate=("patient_churn", "mean"))
              .reset_index())
        fig = px.bar(ag, x=ag["age_group"].astype(str), y="churn_rate", text_auto=".1%",
                     color_discrete_sequence=["#0F2A44"])
        fig.update_traces(textposition="outside")
        styled_chart(fig, height=300, yaxis_tickformat=".0%",
                     xaxis_title="Age group", yaxis_title="Churn rate")
        st.caption("Sample sizes: " + ", ".join(f"{r['age_group']} n={r['patients']}" for _, r in ag.iterrows()))
    with c2:
        render_section_header("Churn by Satisfaction Band")
        sb = (df.groupby("satisfaction_bin", observed=True)
              .agg(patients=("patient_id", "count"), churn_rate=("patient_churn", "mean"))
              .reset_index())
        fig = px.bar(sb, x=sb["satisfaction_bin"].astype(str), y="churn_rate", text_auto=".1%",
                     color_discrete_sequence=["#2A9D8F"])
        fig.update_traces(textposition="outside")
        styled_chart(fig, height=300, yaxis_tickformat=".0%",
                     xaxis_title="Satisfaction band", yaxis_title="Churn rate")
        st.caption(", ".join(f"{r['satisfaction_bin']} n={r['patients']}" for _, r in sb.iterrows()))

    c1, c2 = st.columns(2)
    with c1:
        render_section_header("Churn vs Length of Stay")
        fig = px.box(df, x=df["patient_churn"].map({0: "Retained", 1: "Churned"}),
                     y="length_of_stay", color=df["patient_churn"].map({0: "Retained", 1: "Churned"}),
                     color_discrete_map=CHURN_COLORS)
        styled_chart(fig, height=320, showlegend=False, xaxis_title="", yaxis_title="Length of stay (days)")
    with c2:
        render_section_header("Churn vs Satisfaction — Scatter")
        sample = df.sample(min(2000, len(df)), random_state=42) if len(df) > 2000 else df
        fig = px.scatter(sample, x="satisfaction", y="length_of_stay",
                         color=sample["patient_churn"].map({0: "Retained", 1: "Churned"}),
                         color_discrete_map=CHURN_COLORS, opacity=0.45)
        styled_chart(fig, height=320, xaxis_title="Satisfaction", yaxis_title="Length of stay (days)")

    render_section_header("Churn Heatmap — Service × Age Group")
    heat = (df.groupby(["service", "age_group"], observed=True)
            .agg(churn_rate=("patient_churn", "mean"), n=("patient_id", "count")).reset_index())
    pivot = heat.pivot(index="service", columns="age_group", values="churn_rate").reindex(columns=AGE_ORDER)
    pivot_n = heat.pivot(index="service", columns="age_group", values="n").reindex(columns=AGE_ORDER).fillna(0)

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values, x=AGE_ORDER, y=pivot.index.astype(str),
        colorscale=[[0, "#E6F4EA"], [0.5, "#FDE8A0"], [1, "#D64545"]],
        colorbar=dict(title="Churn rate", tickformat=".0%"),
        hovertemplate="Service: %{y}<br>Age: %{x}<br>Churn: %{z:.1%}<extra></extra>",
        zmin=0, zmax=1,
    ))
    annotations = []
    for i, svc in enumerate(pivot.index):
        for j, ag in enumerate(AGE_ORDER):
            v, n = pivot.iloc[i, j], pivot_n.iloc[i, j]
            if pd.notna(v):
                annotations.append(dict(x=ag, y=svc, text=f"{v:.0%}<br>n={int(n)}",
                                        showarrow=False, font=dict(size=10, color="black" if v < 0.6 else "white")))
    styled_chart(fig, height=360, annotations=annotations, xaxis_title="Age group", yaxis_title="Service")

    # ── NEW: bulk model predictions at the bottom
    render_section_header("Bulk Predictions — Model vs Observed")
    model_badge("patient_churn", df)
    c1, c2 = st.columns(2)
    with c1:
        charts.churn_distribution(df, CHURN_COLORS)
    with c2:
        charts.churn_confusion(df)


def page_hospital_operations():
    st.markdown("# Hospital Operations")
    st.markdown('<div class="section-desc">Demand, admissions, refusals, and capacity pressure.</div>',
                unsafe_allow_html=True)

    cols = st.columns(6)
    with cols[0]: render_kpi("Patient Requests", fmt_num(df["patients_request"].sum()) if not empty else "—")
    with cols[1]: render_kpi("Admitted", fmt_num(df["patients_admitted"].sum()) if not empty else "—")
    with cols[2]: render_kpi("Refused", fmt_num(df["patients_refused"].sum()) if not empty else "—")
    with cols[3]: render_kpi("Admission Rate", fmt_pct(admission_rate) if not empty else "—")
    with cols[4]: render_kpi("Refusal Rate", fmt_pct(refusal_rate) if not empty else "—")
    with cols[5]: render_kpi("Avg Available Beds", fmt_float(df["available_beds"].mean()) if not empty else "—")

    if empty_guard(df):
        return

    render_section_header(
        "Risk-Score Prediction",
        "Click Predict to run the risk-score model against database records.",
    )

    risk_score_prediction = None

    if "risk_score" in choices:
        risk_score_prediction = predict_ui.prediction_form(
            concept="risk_score",
            submodel=choices["risk_score"],
            defaults={
                "age": 50,
                "satisfaction": 65,
                "length_of_stay": 6,
                "available_beds": 18,
                "patients_request": 45,
                "patients_admitted": 32,
                "patients_refused": 13,
                "staff_morale": 68,
                "staff_present": 24,
                "total_staff": 30,
                "staff_absent": 6,
                "capacity_pressure": 45 / 19,
                "admission_rate_row": 32 / 45,
                "refusal_rate_row": 13 / 45,
                "staff_attendance_rate": 24 / 30,
                "service": "ICU",
            },
            form_key="form_risk_score_ops",
            services=sorted(df_raw["service"].dropna().unique().tolist()),
        )

    if risk_score_prediction is None:
        render_prediction_required_message("risk_score")
    else:
        risk_score_df = attach_router_prediction(
            concept="risk_score",
            submodel=choices["risk_score"],
            source_df=df,
        )

        if risk_score_df is not None:
            render_section_header(
                "Predicted Risk Score by Service",
                "Model result calculated from the filtered database records.",
            )

            model_badge("risk_score", risk_score_df)

            charts.service_prediction_bar(
                risk_score_df,
                "risk_score",
                SERVICE_COLORS,
            )

    render_section_header("Model-Predicted Risk Score by Service")
    model_badge("risk_score", df)

    render_section_header("Requests vs Admissions — Monthly")
    mon = (df.groupby(df["arrival_date"].dt.to_period("M"))
           .agg(requests=("patients_request", "sum"), admitted=("patients_admitted", "sum"),
                refused=("patients_refused", "sum")).reset_index())
    mon["arrival_date"] = mon["arrival_date"].astype(str)
    fig = go.Figure()
    fig.add_bar(x=mon["arrival_date"], y=mon["requests"], name="Requests", marker_color="#0F2A44")
    fig.add_bar(x=mon["arrival_date"], y=mon["admitted"], name="Admitted", marker_color="#0E7C7B")
    fig.add_bar(x=mon["arrival_date"], y=mon["refused"], name="Refused", marker_color="#D64545")
    styled_chart(fig, height=320, barmode="group", xaxis_title="", yaxis_title="Patients",
                 legend=dict(orientation="h", y=1.02))

    c1, c2 = st.columns(2)
    with c1:
        render_section_header("Admission Rate by Service")
        svc = (df.groupby("service")
               .agg(req=("patients_request", "sum"), adm=("patients_admitted", "sum"))
               .reset_index())
        svc["admission_rate"] = svc["adm"] / svc["req"]
        svc = svc.sort_values("admission_rate", ascending=False)
        fig = px.bar(svc, x="service", y="admission_rate", color="service",
                     color_discrete_map=SERVICE_COLORS, text_auto=".1%")
        fig.update_traces(textposition="outside")
        styled_chart(fig, height=300, showlegend=False, yaxis_tickformat=".0%",
                     xaxis_title="", yaxis_title="Admission rate")
    with c2:
        render_section_header("Refusal Rate by Service")
        svc2 = (df.groupby("service")
                .agg(req=("patients_request", "sum"), ref=("patients_refused", "sum"))
                .reset_index())
        svc2["refusal_rate"] = svc2["ref"] / svc2["req"]
        svc2 = svc2.sort_values("refusal_rate", ascending=False)
        fig = px.bar(svc2, x="service", y="refusal_rate", color="service",
                     color_discrete_map=SERVICE_COLORS, text_auto=".1%")
        fig.update_traces(textposition="outside")
        styled_chart(fig, height=300, showlegend=False, yaxis_tickformat=".0%",
                     xaxis_title="", yaxis_title="Refusal rate")

    c1, c2 = st.columns(2)
    with c1:
        render_section_header("Available Beds Over Time")
        beds = (df.groupby(df["arrival_date"].dt.to_period("M"))
                .agg(beds=("available_beds", "mean")).reset_index())
        beds["arrival_date"] = beds["arrival_date"].astype(str)
        fig = px.line(beds, x="arrival_date", y="beds", markers=True)
        fig.update_traces(line_color="#0E7C7B")
        styled_chart(fig, height=300, xaxis_title="", yaxis_title="Avg available beds")
    with c2:
        render_section_header("Demand vs Capacity — Monthly Avg")
        dc = (df.groupby(df["arrival_date"].dt.to_period("M"))
              .agg(req=("patients_request", "mean"), beds=("available_beds", "mean"),
                   adm=("patients_admitted", "mean")).reset_index())
        dc["arrival_date"] = dc["arrival_date"].astype(str)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dc["arrival_date"], y=dc["req"], name="Requests",
                                 mode="lines+markers", line=dict(color="#0F2A44")))
        fig.add_trace(go.Scatter(x=dc["arrival_date"], y=dc["beds"], name="Beds",
                                 mode="lines+markers", line=dict(color="#0E7C7B")))
        fig.add_trace(go.Scatter(x=dc["arrival_date"], y=dc["adm"], name="Admitted",
                                 mode="lines+markers", line=dict(color="#2A9D8F", dash="dash")))
        styled_chart(fig, height=300, xaxis_title="", yaxis_title="Patients / Beds")

    render_section_header("Capacity Pressure Distribution")
    cpd = df["capacity_pressure_level"].value_counts().reindex(CAPACITY_ORDER).reset_index()
    cpd.columns = ["level", "count"]
    cpd = cpd.dropna()
    fig = px.bar(cpd, x="level", y="count", color="level",
                 color_discrete_map=CAPACITY_COLORS, text="count")
    fig.update_traces(textposition="outside")
    styled_chart(fig, height=300, showlegend=False, xaxis_title="Pressure level", yaxis_title="Records")

    render_section_header("Key Insights")
    if refusal_rate > 0.4:
        render_insight(f"Refusal rate is high at {refusal_rate:.1%} — demand exceeds capacity.", kind="bad")
    elif refusal_rate > 0.25:
        render_insight(f"Refusal rate is {refusal_rate:.1%} — moderate capacity strain.", kind="warn")

    svc_ref = (df.groupby("service")
               .agg(req=("patients_request", "sum"), ref=("patients_refused", "sum")))
    svc_ref["rate"] = svc_ref["ref"] / svc_ref["req"]
    worst = svc_ref["rate"].idxmax()
    if svc_ref.loc[worst, "rate"] > refusal_rate + 0.05:
        render_insight(f"<b>{worst}</b> has the highest refusal rate ({svc_ref.loc[worst, 'rate']:.1%}) — investigate service-specific bottlenecks.")

def page_staff_analytics():
    st.markdown("# Staff Analytics")
    st.markdown('<div class="section-desc">Staffing levels, morale, attendance and associations with patient outcomes. '
                'Correlation ≠ causation.</div>', unsafe_allow_html=True)

    cols = st.columns(5)
    with cols[0]: render_kpi("Total Staff (sum)", fmt_num(df["total_staff"].sum()) if not empty else "—")
    with cols[1]: render_kpi("Staff Present", fmt_num(df["staff_present"].sum()) if not empty else "—")
    with cols[2]: render_kpi("Staff Absent", fmt_num(df["staff_absent"].sum()) if not empty else "—")
    with cols[3]: render_kpi("Attendance Rate", fmt_pct(staff_att) if not empty else "—")
    with cols[4]: render_kpi("Avg Staff Morale", fmt_float(df["staff_morale"].mean()) if not empty else "—", "/ 100")

    if empty_guard(df):
        return

    # Staff model predictions features
    render_section_header(
        "Staff Model Predictions",
        "Submit either form to calculate predictions for the database records.",
    )

    c1, c2 = st.columns(2)

    with c1:
        morale_prediction = None

        if "staff_staff_morale" in choices:
            morale_prediction = predict_ui.prediction_form(
                concept="staff_staff_morale",
                submodel=choices["staff_staff_morale"],
                defaults={
                    "age": 40,
                    "satisfaction": 70,
                    "length_of_stay": 5,
                    "available_beds": 20,
                    "patients_request": 35,
                    "patients_admitted": 28,
                    "patients_refused": 7,
                    "staff_morale": 70,
                    "staff_present": 24,
                    "total_staff": 30,
                    "staff_absent": 6,
                    "capacity_pressure": 35 / 21,
                    "admission_rate_row": 28 / 35,
                    "refusal_rate_row": 7 / 35,
                    "staff_attendance_rate": 24 / 30,
                    "service": "surgery",
                },
                form_key="form_morale",
                services=sorted(df_raw["service"].dropna().unique().tolist()),
            )

        if morale_prediction is None:
            render_prediction_required_message("staff_staff_morale")
        else:
            inp = morale_prediction.get("_input", {})
            service_val = inp.get("service", "unknown")
            pred_val = morale_prediction.get("prediction")

            # Build an updated DataFrame by inserting the single prediction into the full dataset
            updated_df = df.copy()
            pred_col = "staff_staff_morale_pred"
            if pred_col not in updated_df.columns:
                updated_df[pred_col] = pd.NA
            mask = updated_df["service"] == service_val
            updated_df.loc[mask, pred_col] = pred_val
            model_badge("staff_staff_morale", pd.DataFrame(updated_df))
            charts.service_prediction_bar(
                pd.DataFrame(updated_df),
                "staff_staff_morale",
                SERVICE_COLORS,
            )

    with c2:
        attendance_prediction = None

        if "staff_staff_attendance_rate" in choices:
            attendance_prediction = predict_ui.prediction_form(
                concept="staff_staff_attendance_rate",
                submodel=choices["staff_staff_attendance_rate"],
                defaults={
                    "age": 40,
                    "satisfaction": 70,
                    "length_of_stay": 5,
                    "available_beds": 20,
                    "patients_request": 35,
                    "patients_admitted": 28,
                    "patients_refused": 7,
                    "staff_morale": 70,
                    "staff_present": 24,
                    "total_staff": 30,
                    "staff_absent": 6,
                    "capacity_pressure": 35 / 21,
                    "admission_rate_row": 28 / 35,
                    "refusal_rate_row": 7 / 35,
                    "staff_attendance_rate": 24 / 30,
                    "service": "surgery",
                },
                form_key="form_attendance",
                services=sorted(df_raw["service"].dropna().unique().tolist()),
            )

        # Build an updated DataFrame by inserting the single attendance prediction into the full dataset
        if attendance_prediction is not None:
            inp = attendance_prediction.get("_input", {})
            service_val = inp.get("service", "unknown")
            pred_val = attendance_prediction.get("prediction")
            updated_att_df = df.copy()
            pred_col = "staff_staff_attendance_rate_pred"
            if pred_col not in updated_att_df.columns:
                updated_att_df[pred_col] = pd.NA
            mask = updated_att_df["service"] == service_val
            updated_att_df.loc[mask, pred_col] = pred_val
            model_badge("staff_staff_attendance_rate", pd.DataFrame(updated_att_df))
            charts.service_prediction_bar(
                pd.DataFrame(updated_att_df),
                "staff_staff_attendance_rate",
                SERVICE_COLORS,
            )
        else:
            render_prediction_required_message("staff_staff_attendance_rate")
    
    render_section_header("Bulk Predictions — Observed vs Predicted")
    model_badge("staff_staff_morale", df)

    c1, c2 = st.columns(2)
    with c1:
        render_section_header("Staff Presence Over Time")
        sp = (df.groupby(df["arrival_date"].dt.to_period("M"))
              .agg(present=("staff_present", "mean"), absent=("staff_absent", "mean")).reset_index())
        sp["arrival_date"] = sp["arrival_date"].astype(str)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sp["arrival_date"], y=sp["present"], name="Present",
                                 mode="lines+markers", line=dict(color="#0E7C7B")))
        fig.add_trace(go.Scatter(x=sp["arrival_date"], y=sp["absent"], name="Absent",
                                 mode="lines+markers", line=dict(color="#D64545")))
        styled_chart(fig, height=300, xaxis_title="", yaxis_title="Staff (avg per record)")
    with c2:
        render_section_header("Staff Absence Distribution")
        fig = px.histogram(df, x="staff_absent", nbins=30, color_discrete_sequence=["#D64545"])
        styled_chart(fig, height=300, xaxis_title="Staff absent", yaxis_title="Records")

    c1, c2 = st.columns(2)
    with c1:
        render_section_header("Staff Morale Trend")
        mo = (df.groupby(df["arrival_date"].dt.to_period("M"))
              .agg(morale=("staff_morale", "mean")).reset_index())
        mo["arrival_date"] = mo["arrival_date"].astype(str)
        fig = px.line(mo, x="arrival_date", y="morale", markers=True)
        fig.update_traces(line_color="#0F2A44")
        styled_chart(fig, height=300, xaxis_title="", yaxis_title="Morale")
    with c2:
        render_section_header("Morale vs Patient Satisfaction")
        sample = df.sample(min(2000, len(df)), random_state=7) if len(df) > 2000 else df
        corr = df["staff_morale"].corr(df["satisfaction"])
        fig = px.scatter(sample, x="staff_morale", y="satisfaction", trendline="ols",
                         trendline_color_override="#D64545", color_discrete_sequence=["#0E7C7B"],
                         opacity=0.4)
        fig.update_layout(height=300, xaxis_title="Staff morale", yaxis_title="Patient satisfaction",
                          title=dict(text=f"r = {corr:.2f}", font=dict(size=12)))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Correlation ≠ causation.")
    
    render_section_header("Staff Absence vs Patient Churn")
    df["_abs_bin"] = pd.cut(df["staff_absent"], bins=[-1, 2, 8, 20, 100],
                            labels=["0–2", "3–8", "9–20", "21+"])
    ab = (df.groupby("_abs_bin", observed=True)
          .agg(patients=("patient_id", "count"), churn_rate=("patient_churn", "mean")).reset_index())
    fig = px.bar(ab, x="_abs_bin", y="churn_rate", text_auto=".1%", color_discrete_sequence=["#0F2A44"])
    fig.update_traces(textposition="outside")
    styled_chart(fig, height=300, yaxis_tickformat=".0%",
                 xaxis_title="Staff absent (per record)", yaxis_title="Churn rate")
    st.caption(", ".join(f"{r['_abs_bin']} n={r['patients']}" for _, r in ab.iterrows()))

    render_section_header("Correlation Heatmap")
    cols = [c for c in ["satisfaction", "patient_satisfaction", "length_of_stay", "staff_morale",
                        "staff_absent", "staff_present", "patients_refused", "patients_admitted",
                        "available_beds", "patients_request", "patient_churn"] if c in df.columns]
    corr_mat = df[cols].corr(numeric_only=True)
    fig = go.Figure(data=go.Heatmap(
        z=corr_mat.values, x=corr_mat.columns, y=corr_mat.columns,
        colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
        text=np.round(corr_mat.values, 2), texttemplate="%{text}",
    ))
    styled_chart(fig, height=420, xaxis_tickangle=-30)
    st.caption("Correlation ≠ causation.")

    render_section_header("Key Insights")
    corr_ms = df["staff_morale"].corr(df["satisfaction"])
    if abs(corr_ms) >= 0.3:
        render_insight(f"Staff morale and patient satisfaction show a "
                       f"{'positive' if corr_ms > 0 else 'negative'} association (r={corr_ms:.2f}).")


def page_service_performance():
    st.markdown("# Service Performance")
    st.markdown('<div class="section-desc">Comparative performance with a transparent Service Health Score.</div>',
                unsafe_allow_html=True)
    if empty_guard(df):
        return

    render_section_header(
        "Service-Performance Prediction",
        "Click Predict to run the service-performance model against database records.",
    )

    service_prediction = None

    if "service_performance" in choices:
        service_prediction = predict_ui.prediction_form(
            concept="service_performance",
            submodel=choices["service_performance"],
            defaults={
                "age": 45,
                "satisfaction": 75,
                "length_of_stay": 4,
                "available_beds": 22,
                "patients_request": 40,
                "patients_admitted": 32,
                "patients_refused": 8,
                "staff_morale": 72,
                "staff_present": 26,
                "total_staff": 30,
                "staff_absent": 4,
                "capacity_pressure": 40 / 23,
                "admission_rate_row": 32 / 40,
                "refusal_rate_row": 8 / 40,
                "staff_attendance_rate": 26 / 30,
                "service": "general_medicine",
            },
            form_key="form_service_perf_page",
            services=sorted(df_raw["service"].dropna().unique().tolist()),
        )

    if service_prediction is None:
        render_prediction_required_message("service_performance")
    else:
        service_prediction_df = attach_router_prediction(
            concept="service_performance",
            submodel=choices["service_performance"],
            source_df=df,
        )

        if service_prediction_df is not None:
            model_badge(
                "service_performance",
                service_prediction_df,
            )

            charts.service_prediction_bar(
                service_prediction_df,
                "service_performance",
                SERVICE_COLORS,
            )

    render_section_header("Model-Predicted Service Health")
    model_badge("service_performance", df)

    tbl = (df.groupby("service")
           .agg(patients=("patient_id", "count"),
                churn_rate=("patient_churn", "mean"),
                avg_satisfaction=("satisfaction", "mean"),
                avg_los=("length_of_stay", "mean"),
                requests=("patients_request", "sum"),
                admitted=("patients_admitted", "sum"),
                refused=("patients_refused", "sum"),
                avg_beds=("available_beds", "mean"),
                avg_morale=("staff_morale", "mean"),
                avg_present=("staff_present", "mean"),
                avg_total=("total_staff", "mean"),
                avg_cp=("capacity_pressure", "mean"))
           .reset_index())
    tbl["admission_rate"] = tbl["admitted"] / tbl["requests"]
    tbl["refusal_rate"] = tbl["refused"] / tbl["requests"]
    tbl["attendance"] = tbl["avg_present"] / tbl["avg_total"]
    tbl["health_score"] = (
        0.35 * (tbl["avg_satisfaction"] / 100)
        + 0.25 * (1 - tbl["churn_rate"])
        + 0.20 * tbl["admission_rate"].fillna(0)
        + 0.20 * tbl["attendance"].fillna(0)
    ) * 100

    tbl_show = tbl[["service", "patients", "churn_rate", "avg_satisfaction", "avg_los",
                     "requests", "admitted", "refused", "admission_rate", "refusal_rate",
                     "avg_beds", "avg_morale", "attendance", "avg_cp", "health_score"]].copy()
    tbl_show.columns = ["Service", "Patients", "Churn Rate", "Avg Satisfaction", "Avg LOS",
                         "Requests", "Admitted", "Refused", "Admission Rate", "Refusal Rate",
                         "Avg Beds", "Staff Morale", "Attendance", "Avg Capacity Pressure", "Health Score"]
    st.dataframe(
        tbl_show.style.format({
            "Churn Rate": "{:.1%}", "Avg Satisfaction": "{:.1f}", "Avg LOS": "{:.1f}",
            "Admission Rate": "{:.1%}", "Refusal Rate": "{:.1%}", "Avg Beds": "{:.1f}",
            "Staff Morale": "{:.1f}", "Attendance": "{:.1%}", "Avg Capacity Pressure": "{:.2f}",
            "Health Score": "{:.1f}",
        }).background_gradient(subset=["Health Score"], cmap="Greens"),
        use_container_width=True, hide_index=True,
    )
    with st.expander("How is the Service Health Score calculated?"):
        st.markdown("""
        **Service Health Score (0–100, higher is better)**:
        - 35% Patient Satisfaction
        - 25% Low Churn (1 − churn_rate)
        - 20% Admission Performance (admitted / requests)
        - 20% Staff Attendance (present / total_staff)

        Score = 100 × (0.35·sat + 0.25·(1−churn) + 0.20·admission_rate + 0.20·attendance)

        A transparent operational composite — not a clinical quality score.
        """)

    render_section_header("Rankings")
    r1, r2, r3, r4, r5 = st.columns(5)
    top_churn = tbl.loc[tbl["churn_rate"].idxmax()]
    with r1: render_kpi("Highest Churn", top_churn["service"], f"{top_churn['churn_rate']:.1%}")
    low_sat = tbl.loc[tbl["avg_satisfaction"].idxmin()]
    with r2: render_kpi("Lowest Satisfaction", low_sat["service"], f"{low_sat['avg_satisfaction']:.1f}")
    high_ref = tbl.loc[tbl["refusal_rate"].idxmax()]
    with r3: render_kpi("Highest Refusal Rate", high_ref["service"], f"{high_ref['refusal_rate']:.1%}")
    high_cp = tbl.loc[tbl["avg_cp"].idxmax()]
    with r4: render_kpi("Highest Capacity Pressure", high_cp["service"], f"{high_cp['avg_cp']:.2f}")
    low_att = tbl.loc[tbl["attendance"].idxmin()]
    with r5: render_kpi("Lowest Attendance", low_att["service"], f"{low_att['attendance']:.1%}")

    render_section_header("Service Health Score Ranking")
    sorted_tbl = tbl.sort_values("health_score", ascending=True)
    fig = px.bar(sorted_tbl, x="health_score", y="service", orientation="h",
                 color="health_score", color_continuous_scale="Greens",
                 text=sorted_tbl["health_score"].apply(lambda x: f"{x:.1f}"))
    fig.update_traces(textposition="outside")
    styled_chart(fig, height=300, showlegend=False, xaxis_range=[0, 100],
                 xaxis_title="Health Score (0–100)", yaxis_title="")

    render_section_header("Key Insights")
    best = tbl.loc[tbl["health_score"].idxmax(), "service"]
    worst = tbl.loc[tbl["health_score"].idxmin(), "service"]
    render_insight(f"<b>{best}</b> leads on the composite Health Score; <b>{worst}</b> trails — "
                   f"prioritise the drivers for the lowest-ranked service.")
    churn_gap = tbl["churn_rate"].max() - tbl["churn_rate"].min()
    if churn_gap >= 0.05:
        render_insight(f"Churn varies {churn_gap:.1%} across services — service-level intervention is warranted.")


def page_operational_risk():
    st.markdown("# Operational Risk Monitoring")
    st.markdown('<div class="section-desc">Heuristic risk and capacity pressure monitoring.</div>',
                unsafe_allow_html=True)
    st.info("Operational risk score is a heuristic — not a clinical prediction.")

    if empty_guard(df):
        return

    render_section_header(
        "Risk-Level Prediction",
        "Click Predict to run the risk-level classifier against database records.",
    )

    risk_level_prediction = None

    if "risk_level" in choices:
        risk_level_prediction = predict_ui.prediction_form(
            concept="risk_level",
            submodel=choices["risk_level"],
            defaults={
                "age": 50,
                "satisfaction": 55,
                "length_of_stay": 8,
                "available_beds": 10,
                "patients_request": 55,
                "patients_admitted": 35,
                "patients_refused": 20,
                "staff_morale": 55,
                "staff_present": 18,
                "total_staff": 30,
                "staff_absent": 12,
                "capacity_pressure": 55 / 11,
                "admission_rate_row": 35 / 55,
                "refusal_rate_row": 20 / 55,
                "staff_attendance_rate": 18 / 30,
                "service": "emergency",
            },
            form_key="form_risk_ops",
            services=sorted(df_raw["service"].dropna().unique().tolist()),
        )

    if risk_level_prediction is None:
        render_prediction_required_message("risk_level")
    else:
        risk_level_df = attach_router_prediction(
            concept="risk_level",
            submodel=choices["risk_level"],
            source_df=df,
        )

        if risk_level_df is not None:
            render_section_header(
                "Predicted Risk-Level Distribution",
                "Model result calculated from the filtered database records.",
            )

            risk_counts = (
                risk_level_df["risk_level_pred"]
                .dropna()
                .astype(str)
                .str.title()
                .value_counts()
                .reindex(RISK_ORDER, fill_value=0)
                .rename_axis("risk_level")
                .reset_index(name="patients")
            )

            fig = px.bar(
                risk_counts,
                x="risk_level",
                y="patients",
                color="risk_level",
                color_discrete_map=RISK_COLORS,
                text="patients",
            )

            fig.update_traces(textposition="outside")

            styled_chart(
                fig,
                height=320,
                showlegend=False,
                xaxis_title="Predicted risk level",
                yaxis_title="Patients",
            )
    
    st.markdown("---")

    cols = st.columns(4)
    with cols[0]: render_kpi("Avg Risk Score", fmt_float(df["risk_score"].mean(), 1) + " /100")
    with cols[1]: render_kpi("High Risk", fmt_num((df["risk_level"] == "High").sum()),
                             fmt_pct(safe_div((df["risk_level"] == "High").sum(), len(df))))
    with cols[2]: render_kpi("Medium Risk", fmt_num((df["risk_level"] == "Medium").sum()),
                             fmt_pct(safe_div((df["risk_level"] == "Medium").sum(), len(df))))
    with cols[3]: render_kpi("Critical Capacity", fmt_num((df["capacity_pressure_level"] == "Critical").sum()),
                             fmt_pct(safe_div((df["capacity_pressure_level"] == "Critical").sum(), len(df))))

    with st.expander("How is the Operational Risk Score calculated?"):
        st.markdown("""
        **Risk Score (0–100)**:
        - 40% Dissatisfaction risk: (100 − satisfaction) / 100
        - 25% Length-of-stay risk: (LOS − min) / (max − min)
        - 20% Refusal pressure: patients_refused / patients_request (clipped 0–1)
        - 15% Low morale risk: (100 − staff_morale) / 100

        **Levels**: Low ≤30, Medium 31–60, High >60.
        **Capacity Pressure** = patients_request / (available_beds + 1).
        Levels: Low <1, Moderate 1–2, High 2–4, Critical >4.

        Thresholds are operational heuristics, not clinical standards.
        """)

    c1, c2 = st.columns(2)
    with c1:
        render_section_header("Risk Level Distribution")
        rc = df["risk_level"].value_counts().reindex(RISK_ORDER).reset_index()
        rc.columns = ["level", "count"]
        rc = rc.dropna()
        fig = px.bar(rc, x="level", y="count", color="level",
                     color_discrete_map=RISK_COLORS, text="count")
        fig.update_traces(textposition="outside")
        styled_chart(fig, height=300, showlegend=False, xaxis_title="Risk level", yaxis_title="Patients")
    with c2:
        render_section_header("Risk by Service")
        rs = (df.groupby("service")["risk_score"].mean().reset_index()
              .sort_values("risk_score", ascending=False))
        fig = px.bar(rs, x="service", y="risk_score", color="service",
                     color_discrete_map=SERVICE_COLORS, text_auto=".1f")
        fig.update_traces(textposition="outside")
        styled_chart(fig, height=300, showlegend=False, xaxis_title="", yaxis_title="Avg risk score")

    c1, c2 = st.columns(2)
    with c1:
        render_section_header("Risk vs Satisfaction")
        sample = df.sample(min(2000, len(df)), random_state=11) if len(df) > 2000 else df
        fig = px.scatter(sample, x="satisfaction", y="risk_score", color="risk_level",
                         color_discrete_map=RISK_COLORS, opacity=0.5)
        styled_chart(fig, height=320, xaxis_title="Satisfaction", yaxis_title="Risk score")
    with c2:
        render_section_header("Capacity Pressure — Service Avg")
        cp = (df.groupby("service")["capacity_pressure"].mean().reset_index()
              .sort_values("capacity_pressure", ascending=False))
        fig = px.bar(cp, x="service", y="capacity_pressure", color="service",
                     color_discrete_map=SERVICE_COLORS, text_auto=".2f")
        fig.update_traces(textposition="outside")
        styled_chart(fig, height=320, showlegend=False, xaxis_title="", yaxis_title="Avg capacity pressure")

    render_section_header("Highest-Risk Patients")
    top = df.sort_values("risk_score", ascending=False).head(15)[
        ["patient_id", "name", "age", "service", "satisfaction", "length_of_stay",
         "patients_refused", "staff_morale", "risk_score", "risk_level", "capacity_pressure_level"]
    ]
    top.columns = ["Patient ID", "Name", "Age", "Service", "Satisfaction", "LOS",
                   "Refused", "Morale", "Risk Score", "Risk Level", "Capacity"]
    top["Risk Score"] = top["Risk Score"].round(1)
    st.dataframe(top, use_container_width=True, hide_index=True)


def page_model_performance():
    st.markdown("# Model Performance")
    st.markdown('<div class="section-desc">Trained submodels per concept with evaluation metrics — '
                'sourced from manifests stored in MinIO.</div>', unsafe_allow_html=True)

    if not manifests:
        st.info("No manifests found. Train models first.")
        return

    render_section_header("Summary", "Best submodel per concept by primary metric")
    rows = []
    for concept, m in manifests.items():
        task = m["task"]
        metric_key = "roc_auc" if task == "classification" else "r2"
        metrics = m["metrics"]
        best_name, best_metrics = max(metrics.items(), key=lambda kv: kv[1].get(metric_key, -1) or -1)
        rows.append({
            "Concept": concept, "Task": task, "Submodels": len(metrics),
            "Best Submodel": best_name,
            "Primary Metric": f"{metric_key} = {best_metrics.get(metric_key):.4f}",
            "Feature Count": len(m["features"]),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    render_section_header("Per-Concept Detail")
    for concept, m in manifests.items():
        with st.expander(f"{concept} - {len(m['submodels'])} submodels", expanded=False):
            df_metrics = pd.DataFrame(m["metrics"]).T.reset_index().rename(columns={"index": "submodel"})
            fmt = {c: "{:.4f}" for c in df_metrics.columns if c != "submodel"}
            st.dataframe(df_metrics.style.format(fmt), use_container_width=True, hide_index=True)

            c1, c2 = st.columns([3, 2])
            with c1:
                melted = df_metrics.melt(id_vars="submodel", var_name="metric", value_name="score")
                fig = px.bar(melted, x="submodel", y="score", color="metric", barmode="group",
                             title=f"{concept} - metrics by submodel")
                styled_chart(fig, height=320, xaxis_title="", yaxis_title="Score")
            with c2:
                st.markdown("**Features used**")
                st.code("\n".join(m["features"]), language="text")

    # ── NEW: use charts.prediction_coverage_table
    render_section_header("Prediction Coverage")
    charts.prediction_coverage_table(df, MODEL_CONCEPTS)

def page_management_insights():
    st.markdown("# Management Insights")
    st.markdown('<div class="section-desc">Auto-generated from the current filtered dataset. '
                'Groups with n&lt;30 flagged.</div>', unsafe_allow_html=True)
    if empty_guard(df):
        return

    insights = []

    if df["patient_churn"].nunique() > 1:
        s_c = df[df["patient_churn"] == 1]["satisfaction"].mean()
        s_r = df[df["patient_churn"] == 0]["satisfaction"].mean()
        if pd.notna(s_c) and pd.notna(s_r) and abs(s_c - s_r) >= 2:
            insights.append(("Patient Experience & Churn",
                             f"Churned patients average {s_c:.1f} satisfaction vs {s_r:.1f} for retained "
                             f"(Δ {s_c - s_r:+.1f}). Association, not causation."))

    avg_cp = df["capacity_pressure"].mean()
    crit_share = safe_div((df["capacity_pressure_level"] == "Critical").sum(), len(df))
    if avg_cp >= 2 or crit_share > 0.08:
        insights.append(("Capacity & Access",
                         f"Capacity pressure averages {avg_cp:.2f} with {crit_share:.1%} at Critical level."))
    elif avg_cp >= 1:
        insights.append(("Capacity & Access", f"Capacity pressure is moderate ({avg_cp:.2f})."))

    svc_churn = df.groupby("service")["patient_churn"].mean()
    if len(svc_churn) > 1:
        gap = svc_churn.max() - svc_churn.min()
        if gap >= 0.05:
            insights.append(("Service Variation",
                             f"Churn ranges from {svc_churn.min():.1%} to {svc_churn.max():.1%} "
                             f"(gap {gap:.1%}). <b>{svc_churn.idxmax()}</b> is the outlier."))

    morale = df["staff_morale"].mean()
    corr_ms = df["staff_morale"].corr(df["satisfaction"])
    if morale < 65:
        insights.append(("Workforce",
                         f"Avg staff morale is {morale:.1f}/100 — below 65. "
                         f"Correlates with patient experience (r={corr_ms:.2f})."))
    elif morale < 75:
        insights.append(("Workforce", f"Staff morale averages {morale:.1f}/100 — moderate."))

    if staff_att < 0.7:
        insights.append(("Attendance", f"Staff attendance is {staff_att:.1%} — below 70%."))

    if refusal_rate > 0.35:
        insights.append(("Patient Flow",
                         f"Refusal rate is {refusal_rate:.1%} — more than one in three requests refused."))

    hr = safe_div((df["risk_level"] == "High").sum(), len(df))
    if hr > 0.15:
        insights.append(("Risk Concentration", f"{hr:.1%} are High operational risk."))

    if "patient_churn_prob" in df.columns:
        high_pool = df[df["patient_churn_prob"] > 0.7]
        if len(high_pool) > 0:
            used = df["patient_churn_submodel"].dropna().iloc[0]
            insights.append(("Model Signal",
                             f"{len(high_pool)} patients ({len(high_pool)/len(df):.1%}) "
                             f"have predicted churn probability > 0.7 using **{used}**. "
                             f"Prioritise for outreach — predicted, not confirmed."))

    if not insights:
        render_insight("No meaningful differences detected in the current filtered data. "
                       "Try broadening the date or service filters.")
    else:
        for title, body in insights:
            st.markdown(f"**{title}**")
            render_insight(body)
            st.markdown("")

    st.markdown("### At-a-Glance — Filtered KPIs")
    kpi_df = pd.DataFrame([{
        "Patients": len(df), "Churn Rate": f"{churn_rate:.1%}",
        "Avg Satisfaction": f"{df['satisfaction'].mean():.1f}",
        "Avg LOS (days)": f"{df['length_of_stay'].mean():.1f}",
        "Admission Rate": f"{admission_rate:.1%}",
        "Refusal Rate": f"{refusal_rate:.1%}",
        "Avg Beds": f"{df['available_beds'].mean():.1f}",
        "Staff Attendance": f"{staff_att:.1%}",
        "Staff Morale": f"{df['staff_morale'].mean():.1f}",
        "Avg Risk Score": f"{df['risk_score'].mean():.1f}",
        "Avg Capacity Pressure": f"{df['capacity_pressure'].mean():.2f}",
    }])
    st.dataframe(kpi_df, use_container_width=True, hide_index=True)


def page_data_explorer():
    st.markdown("# Data Explorer")
    st.markdown('<div class="section-desc">Data quality, schema and preview. Download respects global filters.</div>',
                unsafe_allow_html=True)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1: render_kpi("Rows (filtered)", fmt_num(len(df)), f"raw {len(df_raw):,}")
    with c2: render_kpi("Columns", str(df_raw.shape[1]))
    with c3: render_kpi("Missing Values", fmt_num(df_raw.isna().sum().sum()))
    with c4: render_kpi("Duplicate IDs", fmt_num(df_raw.duplicated(subset=["patient_id"]).sum()))
    with c5: render_kpi("Duplicate Rows", fmt_num(df_raw.duplicated().sum()))
    with c6: render_kpi("Services", str(df_raw["service"].nunique()))

    c7, c8 = st.columns(2)
    with c7: render_kpi("Date Range (raw)",
                        f"{df_raw['arrival_date'].min().date()} → {df_raw['arrival_date'].max().date()}")
    with c8: render_kpi("Churn (raw)",
                        f"{(df_raw['patient_churn'] == 1).sum():,} / {(df_raw['patient_churn'] == 0).sum():,}",
                        f"{df_raw['patient_churn'].mean():.1%} churn rate")

    render_section_header("Data Validation")
    issues = []
    if df_raw.duplicated(subset=["patient_id"]).sum() > 0:
        issues.append(("Duplicate patient_id records.", "warn"))
    if (df_raw["patients_refused"] > df_raw["patients_request"]).sum() > 0:
        issues.append(("Operational inconsistency: patients_refused > patients_request.", "warn"))
    mismatch = ((df_raw["staff_present"] + df_raw["staff_absent"]).round() != df_raw["total_staff"]).sum()
    if mismatch > 0:
        issues.append((f"Staffing inconsistency in {mismatch:,} records — likely synthetic/aggregated fields.", "warn"))
    if (df_raw["departure_date"] < df_raw["arrival_date"]).sum() > 0:
        issues.append(("Invalid dates: departure < arrival.", "warn"))
    if not set(df_raw["patient_churn"].unique()).issubset({0, 1}):
        issues.append(("Invalid churn values detected.", "warn"))
    if df_raw.isna().sum().sum() == 0:
        issues.append(("No missing values — dataset is complete.", "info"))

    for msg, kind in issues:
        render_insight(msg, kind=kind if kind != "info" else "info")
    st.caption("Records are not modified. Warnings are surfaced for analytical awareness.")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Dataset Preview", "Data Types", "Missing Values", "Descriptive Statistics", "Model Columns"])

    with tab1:
        st.dataframe(df.head(200), use_container_width=True, hide_index=True, height=400)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download filtered dataset (CSV)", csv,
                           file_name="hospital_filtered.csv", mime="text/csv")
        st.caption(f"Showing first 200 of {len(df):,} rows.")
    with tab2:
        dtypes = pd.DataFrame({"column": df_raw.columns, "dtype": df_raw.dtypes.astype(str).values})
        st.dataframe(dtypes, use_container_width=True, hide_index=True)
    with tab3:
        miss = df_raw.isna().sum().reset_index()
        miss.columns = ["column", "missing"]
        miss["missing_pct"] = miss["missing"] / len(df_raw) * 100
        st.dataframe(miss.style.format({"missing_pct": "{:.2f}%"}), use_container_width=True, hide_index=True)
    with tab4:
        st.markdown("**Numeric summary (raw dataset)**")
        st.dataframe(df_raw.describe().T, use_container_width=True)
        st.markdown("**Correlation (numeric)**")
        num_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
        corr = df_raw[num_cols].corr(numeric_only=True)
        fig = go.Figure(data=go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns, colorscale="RdBu",
            zmid=0, zmin=-1, zmax=1, text=np.round(corr.values, 2), texttemplate="%{text}",
        ))
        styled_chart(fig, height=500, xaxis_tickangle=-30)
        st.caption("Correlation ≠ causation.")
    with tab5:
        model_cols = [c for c in df.columns
                      if c.endswith("_pred") or c.endswith("_prob")
                      or "_prob_class_" in c or c.endswith("_submodel")]
        if model_cols:
            st.dataframe(df[model_cols].head(50), use_container_width=True, hide_index=True)
        else:
            st.info("No model columns — check the Models panel in the sidebar for errors.")

        rows = []
        for c in MODEL_CONCEPTS:
            col = f"{c}_submodel"
            if col in df.columns and df[col].notna().any():
                rows.append({"Concept": c, "Submodel": df[col].dropna().iloc[0]})
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ──────────────────────────────────────────────
# Router
# ──────────────────────────────────────────────
PAGE_FUNCTIONS = {
    "Overview": page_overview,
    "Patient Analytics": page_patient_analytics,
    "Patient Management": page_patient_management,
    "Churn Analytics": page_churn_analytics,
    "Hospital Operations": page_hospital_operations,
    "Staff Analytics": page_staff_analytics,
    "Service Performance": page_service_performance,
    "Operational Risk": page_operational_risk,
    "Model Performance": page_model_performance,
    "Management Insights": page_management_insights,
    "Data Explorer": page_data_explorer,
}

PAGE_FUNCTIONS[nav]()

# Footer
st.markdown("---")
st.caption("Hospital Care Analytics — Operational dashboard. "
           "Data source: fix_final_hospital_database.parquet. Metrics computed live from filtered data.")