import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
import json as _json
from plotly.subplots import make_subplots
import pathlib
import sys
from pathlib import Path
from sqlalchemy import create_engine, text

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STREAMLIT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path[:] = [path for path in sys.path if Path(path or ".").resolve() != STREAMLIT_DIR]

from app.api.config import settings
from app.streamlit.metrics import start_metrics_server, update_dashboard_metrics
from app.ml.registry import load_concept
from app.ml.minio_store import _client

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Hospital Care Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def initialize_metrics_server() -> None:
    start_metrics_server()

initialize_metrics_server()

# ──────────────────────────────────────────────
# Styles
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 1.2rem; padding-bottom: 1rem; }
h1 { font-weight: 700; color: #0F2A44; letter-spacing: -0.02em; }
h2 { color: #0F2A44; font-weight: 700; font-size: 1.35rem; margin-top: 1.6rem; border-bottom: 1px solid #E8EEF3; padding-bottom: 0.4rem; }
h3 { color: #1a3a5c; font-weight: 600; font-size: 1.05rem; margin-top: 1rem; }
.kpi-card {
  background: white; border: 1px solid #E6EBF0; border-radius: 12px;
  padding: 16px 18px; box-shadow: 0 1px 3px rgba(15,42,68,0.06);
}
.kpi-label { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.07em; color: #6B7C93; font-weight: 600; }
.kpi-value { font-size: 1.55rem; font-weight: 700; color: #0F2A44; margin-top: 4px; }
.kpi-sub { font-size: 0.78rem; color: #8A9BB0; margin-top: 2px; }
.section-desc { color: #5A6D85; font-size: 0.9rem; margin-top: -6px; margin-bottom: 12px; }
.insight-box {
  background: #F0F7F7; border-left: 4px solid #0E7C7B; border-radius: 8px;
  padding: 10px 14px; margin: 6px 0; font-size: 0.88rem; color: #1a3a5c;
}
.insight-warn { background: #FFF7E6; border-left-color: #E6A817; }
.insight-bad  { background: #FFF0F0; border-left-color: #D64545; }
.badge-low { background:#E6F4EA; color:#1B6B2E; padding:2px 8px; border-radius:999px; font-size:0.75rem; font-weight:600; }
.badge-med { background:#FFF3CD; color:#7A5A00; padding:2px 8px; border-radius:999px; font-size:0.75rem; font-weight:600; }
.badge-high{ background:#FCE8E8; color:#9B1C1C; padding:2px 8px; border-radius:999px; font-size:0.75rem; font-weight:600; }
.badge-crit{ background:#9B1C1C; color:white; padding:2px 8px; border-radius:999px; font-size:0.75rem; font-weight:600; }
[data-testid="stSidebar"] { background: #0F2A44; }
[data-testid="stSidebar"] * { color: #E8EEF3; }
[data-testid="stSidebar"] .stRadio label { color: #E8EEF3 !important; }
hr { border-color: #E8EEF3; }
</style>
""", unsafe_allow_html=True)

PALETTE = ["#0F2A44", "#0E7C7B", "#2A9D8F", "#E6A817", "#D64545", "#6B7C93", "#8AB4B5", "#C4D7D1"]
SERVICE_COLORS = {"emergency":"#0F2A44","surgery":"#0E7C7B","general_medicine":"#2A9D8F","ICU":"#D64545"}

# ──────────────────────────────────────────────
# Data loading — robust path
# ──────────────────────────────────────────────
def resolve_parquet_path():
    candidates = [
        pathlib.Path(__file__).resolve().parent.parent / "database" / "fix_final_hospital_database.parquet",
        pathlib.Path(__file__).resolve().parent / "database" / "fix_final_hospital_database.parquet",
        pathlib.Path.cwd() / "develop" / "database" / "fix_final_hospital_database.parquet",
        pathlib.Path.cwd() / "database" / "fix_final_hospital_database.parquet",
        pathlib.Path("develop/database/fix_final_hospital_database.parquet"),
        pathlib.Path("database/fix_final_hospital_database.parquet"),
        pathlib.Path(__file__).resolve().parents[2] / "develop" / "database" / "fix_final_hospital_database.parquet",
    ]
    for p in candidates:
        if p.exists():
            return p
    # fallback: search
    for p in pathlib.Path.cwd().rglob("fix_final_hospital_database.parquet"):
        return p
    return candidates[0]

@st.cache_resource
def get_database_engine():
    return create_engine(
        (
            "postgresql+psycopg2://"
            f"{settings.POSTGRES_USER}:"
            f"{settings.POSTGRES_PASSWORD}@"
            f"{settings.POSTGRES_HOST}:"
            f"{settings.POSTGRES_PORT}/"
            f"{settings.POSTGRES_DB}"
        ),
        pool_pre_ping=True,
    )

@st.cache_data(show_spinner="Loading hospital dataset...")
def load_data():
    schema = settings.POSTGRES_SCHEMA
    table = settings.POSTGRES_TABLE

    query = text(f"""SELECT * FROM {schema}.{table}""")

    with get_database_engine().connect() as conn:
        df = pd.read_sql(query, conn)

    # Normalize column names: staff_absens typo -> staff_absent (keep both)
    if "staff_absens" in df.columns and "staff_absent" not in df.columns:
        df["staff_absent"] = df["staff_absens"]

    # Ensure datetime
    for c in ["arrival_date","departure_date"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    
    # ========================================
    # Add new features to enchance analysis and insights
    # =======================================
    if "arrival_date" in df.columns:
        df["arrival_year"] = df["arrival_date"].dt.year
        df["arrival_month"] = df["arrival_date"].dt.month
        df["arrival_month_name"] = df["arrival_date"].dt.strftime("%b")
        df["arrival_week"] = df["arrival_date"].dt.isocalendar().week.astype(int)
        df["arrival_day"] = df["arrival_date"].dt.day
        df["arrival_day_of_week"] = df["arrival_date"].dt.day_name()
        df["arrival_dow_num"] = df["arrival_date"].dt.dayofweek
        df["is_weekend"] = df["arrival_dow_num"].isin([5,6])
    # Age groups
    df["age_group"] = pd.cut(df["age"], bins=[-1,12,18,30,45,60,120],
                             labels=["0–12","13–18","19–30","31–45","46–60","61+"], include_lowest=True)
    df["satisfaction_bin"] = pd.cut(df["satisfaction"], bins=[-1,20,40,60,80,100],
                                    labels=["0–20","21–40","41–60","61–80","81–100"], include_lowest=True)
    
    df["capacity_pressure"] = df["patients_request"] / (df["available_beds"] + 1)
    df["staff_attendance_rate"] = (df["staff_present"] / df["total_staff"].replace(0, np.nan).fillna(0.0))
    df["admission_rate_row"] = (df["patients_admitted"] / df["patients_request"].replace(0, np.nan).fillna(0.0))
    df["refusal_rate_row"] = (df["patients_refused"] / df["patients_request"].replace(0, np.nan).fillna(0.0))
    df["capacity_pressure_level"] = pd.cut(df["capacity_pressure"], 
                                           bins=[-1,1,2,3,1000], labels=["Low", "Moderate", "High", "Very High"], include_lowest=True)
    df["risk_score"] = (
            (df["patients_request"] <= df["patients_request"].mean()).astype(float)
            + (df["patients_refused"] >= df["patients_refused"].mean()).astype(float)
            + (df["patients_admitted"] <= df["patients_admitted"].mean()).astype(float)
            + (df["satisfaction"] <= df["satisfaction"].mean()).astype(float)
    ) / 4.0
    df["risk_level"] = pd.cut(df["risk_score"], 
                              bins=[-1, 0.25, 0.5, 0.75, 1], labels=["Low", "Medium", "High", "Critical"], include_lowest=True)

    return df

# ──────────────────────────────────────────────
# Model bundles + manifest from MinIO
# ──────────────────────────────────────────────
@st.cache_resource
def load_bundle_cached(concept: str):
    """Load a trained bundle once per Streamlit session."""
    return load_concept(concept)

@st.cache_data
def load_manifest_cached(concept: str) -> dict:
    """Load a manifest JSON from MinIO (metrics, submodels, features.)"""
    c = _client()
    r = c.get_object(settings.MINIO_BUCKET, f"manifest/{concept}.json")
    try:
        return _json.loads(r.read().decode())
    finally:
        r.close()
        r.release_conn()

@st.cache_data(show_spinner=False)
def predict_with_bundle(concept: str, submodel: str, df_hash: str, df: pd.DataFrame) -> pd.DataFrame:
    """Run the best submodel of a concept bundle against df.
    'df_hash' as a cache key the frame iteself is passed for content."""
    bundle = load_bundle_cached(concept)
    models = bundle["models"]

    if submodel not in models:
        raise ValueError(f"Submodel '{submodel}' not found in bundle for concept '{concept}'. Available: {list(models.keys())}")

    entry = models[submodel]
    pipe = entry["pipeline"]
    features = bundle["features"]

    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing required features for {concept}: {missing}")

    # Split into X and predict
    X = df[features]
    preds = pipe.predict(X)

    # Label encoding
    le = entry.get("label_encoder")
    if le is not None:
        try:
            preds = le.inverse_transform(preds)
        except Exception:
            pass # Keep raw preds if inverse fails

    out = df[["patient_id"]].copy()
    out[f"{concept}_pred"] = preds
    out[f"{concept}_submodel"] = submodel

    # Get attribution predictions if available
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X)

        if proba.ndim == 2 and proba.shape[1] == 2:
            out[f"{concept}_prob"] = proba[:, 1]

        elif proba.ndim == 2 and proba.shape[1] > 2:
            for i in range(proba.shape[1]):
                out[f"{concept}_prob_class_{i}"] = proba[:, i]

    return out

# Load dataframe by the function
try:
    df_raw = load_data()
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

# Attach model predictions to the raw frame for every page can use it
MODEL_CONCEPTS = [
    "patient_churn",
    "risk_score",
    "risk_level",
    "service_performance",
    "integrity_audit",
    "staff_staff_morale",
    "staff_staff_attendance_rate",
]

model_errors = {}

# ──────────────────────────────────────────────
# Helpers — safe metrics
# ──────────────────────────────────────────────
def safe_div(num, den):
    try:
        if den == 0 or pd.isna(den) or pd.isna(num):
            return 0.0
        return num / den
    except:
        return 0.0

def calculate_churn_rate(df):
    if len(df)==0: return 0.0
    return df["patient_churn"].mean() if "patient_churn" in df.columns else 0.0

def calculate_admission_rate(df):
    r = df["patients_request"].sum()
    a = df["patients_admitted"].sum()
    return safe_div(a, r)

def calculate_refusal_rate(df):
    r = df["patients_request"].sum()
    f = df["patients_refused"].sum()
    return safe_div(f, r)

def calculate_staff_attendance_rate(df):
    present = df["staff_present"].sum()
    total = df["total_staff"].sum()
    return safe_div(present, total)

def render_kpi(label, value, sub=""):
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>', unsafe_allow_html=True)

def render_section_header(title, desc=""):
    st.markdown(f"### {title}")
    if desc:
        st.markdown(f'<div class="section-desc">{desc}</div>', unsafe_allow_html=True)

def render_insight(text, kind="info"):
    cls = "insight-box"
    if kind=="warn": cls += " insight-warn"
    elif kind=="bad": cls += " insight-bad"
    st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)

def model_badge(concept: str):
    """Small text showing which submodel produced the columns for a concept."""
    col = f"{concept}_submodel"
    if col in df.columns and df[col].notna().any():
        used = df[col].dropna().unique()
        st.caption(f"🧠 Using: **{concept}** → {', '.join(map(str, used))}")

def fmt_pct(x): return f"{x*100:.1f}%" if pd.notna(x) else "—"
def fmt_num(x): return f"{x:,.0f}" if pd.notna(x) else "—"
def fmt_float(x, d=1): return f"{x:.{d}f}" if pd.notna(x) else "—"

# ──────────────────────────────────────────────
# Sidebar — Navigation + Global Filters
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 Hospital Care Analytics")
    st.markdown('<div style="color:#8AB4B5; font-size:0.82rem; margin-top:-8px; margin-bottom:16px;">Operational & Patient Intelligence</div>', unsafe_allow_html=True)

    nav = st.radio(
        "Navigation",
        ["🏥 Overview","👥 Patient Analytics","🧑‍⚕️ Patient Management",
        "📉 Churn Analytics","🛏️ Hospital Operations","👨‍⚕️ Staff Analytics",
        "📊 Service Performance","⚠️ Operational Risk","🧠 Model Performance",   
        "💡 Management Insights","🔎 Data Explorer"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("### Filters")

    # Date range
    min_d = df_raw["arrival_date"].min().date()
    max_d = df_raw["arrival_date"].max().date()
    d_range = st.date_input("Arrival date range", value=(min_d, max_d), min_value=min_d, max_value=max_d)
    if isinstance(d_range, tuple) and len(d_range)==2:
        start_d, end_d = d_range
    else:
        start_d, end_d = min_d, max_d

    # Service
    all_services = sorted(df_raw["service"].dropna().unique().tolist())
    sel_services = st.multiselect("Service", all_services, default=all_services)
    
    # Age
    age_min, age_max = int(df_raw["age"].min()), int(df_raw["age"].max())
    age_range = st.slider("Age range", age_min, age_max, (age_min, age_max))

    # Churn
    churn_opt = st.selectbox("Patient churn", ["All","Churned","Retained"])
    st.markdown("---")
    st.caption("This dashboard is intended for operational and analytical purposes. It does not provide clinical diagnosis or medical advice.")

    # --- Add models panel on the sidebar ---
    st.markdown("---")
    st.markdown("### 🧠 Models")

    try:
        concepts_available = [c for c in MODEL_CONCEPTS if c not in model_errors]
        if concepts_available:
            st.caption(f"{len(concepts_available)}/{len(MODEL_CONCEPTS)} models loaded successfully.")
            st.markdown(
            "".join(f'<div style="font-size:0.78rem;color:#8AB4B5;">✅ {c}</div>'
                    for c in concepts_available),
            unsafe_allow_html=True
        )

        if model_errors:
            with st.expander(f"⚠️ {len(model_errors)} failed", expanded=False):
                for c, e in model_errors.items():
                    st.caption(f"**{c}** — {e[:120]}")

    except Exception as e:
        st.warning(f"❌ Model panel error: {e}")

    # --- Submodel selector ---
    st.markdown("### 🎛️ Model Selection")

    # Per-concept model choice (defaults to best metric)
    model_choice: dict[str, str] = {}
    model_manifests: dict[str, dict] = {}

    for concept in MODEL_CONCEPTS:
        if concept in model_errors:
            continue # Skip concepts with errors

        try:
            m = load_manifest_cached(concept)
            model_manifests[concept] = m
        except Exception:
            continue # Skip manifest load errors

        task = m["task"]
        metric_key = "roc_auc" if task=="classification" else "r2"

        submodels = m["submodels"]
        metrics = m["metrics"]
        best_by_metric = max(submodels, key=lambda s: metrics.get(s, {}).get(metric_key, -1) or -1)

        # Human-readable label with primary metric
        options = []
        label_to_name = {}
        for s in submodels:
            val = metrics.get(s, {}).get(metric_key)
            pretty = f"{s} ({metric_key}={val:.3f})" if val is not None else s
            options.append(pretty)
            label_to_name[pretty] = s

        # Default label is the best by metric
        default_label = next(lbl for lbl, n in label_to_name.items() if n == best_by_metric)
        with st.expander(concept, expanded=False):
            pick = st.selectbox(
                "submodel",
                options,
                index=options.index(default_label),
                key=f"submodel_{concept}",
                label_visibility="collapsed"
            )
            model_choice[concept] = label_to_name[pick]

    # --- Compute predictions with user-selected submodels ---
    pred_frames = {}
    for concept, submodel in model_choice.items():
        if concept in model_errors:
            continue

        try:
            h = f"{concept}_{submodel}_{len(df_raw)}"
            pred_frames[concept] = predict_with_bundle(concept, submodel, h, df_raw)

        except Exception as e:
            model_errors[concept] = str(e)

    # Attach to df_raw in every pages
    for concept, pf in pred_frames.items():
        df_raw = df_raw.merge(pf, on="patient_id", how="left")

# ──────────────────────────────────────────────
# Apply global filters
# ──────────────────────────────────────────────
df = df_raw.copy()
mask = (df["arrival_date"].dt.date >= start_d) & (df["arrival_date"].dt.date <= end_d)

df = df[mask]
if sel_services:
    df = df[df["service"].isin(sel_services)]
else:
    df = df.iloc[0:0]

df = df[(df["age"] >= age_range[0]) & (df["age"] <= age_range[1])]
if churn_opt == "Churned":
    df = df[df["patient_churn"]==1]
elif churn_opt == "Retained":
    df = df[df["patient_churn"]==0]

empty = len(df)==0

update_dashboard_metrics(
    row_count=len(df),
    churn_rate=float(df["patient_churn"].mean()) if not empty else 0.0,
    average_satisfaction=float(df["satisfaction"].mean()) if not empty else 0.0,
)

def empty_guard():
    if empty:
        st.warning("No records match the current filters. Adjust the date, service, age, or churn filters in the sidebar.")
        return True
    return False

# Shared aggregates for insights
churn_rate = calculate_churn_rate(df) if not empty else 0
admission_rate = calculate_admission_rate(df) if not empty else 0
refusal_rate = calculate_refusal_rate(df) if not empty else 0
staff_att = calculate_staff_attendance_rate(df) if not empty else 0

# ──────────────────────────────────────────────
# Page: Overview
# ──────────────────────────────────────────────
if nav == "🏥 Overview":
    st.markdown("# Hospital Care Analytics")
    st.markdown('<div class="section-desc">Monitor patient experience, hospital capacity, staff performance, and patient churn — filtered data updates every panel.</div>', unsafe_allow_html=True)

    # KPI row 1
    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: render_kpi("Total Patients", fmt_num(len(df)), f"of {len(df_raw):,} total")
    with c2: render_kpi("Churned", fmt_num(df["patient_churn"].sum()) if not empty else "0", fmt_pct(churn_rate) + " churn rate")
    with c3: render_kpi("Avg Satisfaction", fmt_float(df["satisfaction"].mean(),1) if not empty else "—", "/ 100")
    with c4: render_kpi("Avg Length of Stay", fmt_float(df["length_of_stay"].mean(),1)+" d" if not empty else "—", f"range {int(df_raw['length_of_stay'].min())}–{int(df_raw['length_of_stay'].max())} d")
    with c5: render_kpi("Avg Staff Morale", fmt_float(df["staff_morale"].mean(),1) if not empty else "—", "/ 100")
    c6,c7,c8,c9,c10 = st.columns(5)
    with c6: render_kpi("Admission Rate", fmt_pct(admission_rate) if not empty else "—", f"{fmt_num(df['patients_admitted'].sum())} admitted" if not empty else "")
    with c7: render_kpi("Refusal Rate", fmt_pct(refusal_rate) if not empty else "—", f"{fmt_num(df['patients_refused'].sum())} refused" if not empty else "")
    with c8: render_kpi("Available Beds (avg)", fmt_float(df["available_beds"].mean(),1) if not empty else "—", "per day")
    with c9: render_kpi("Staff Attendance", fmt_pct(staff_att) if not empty else "—", f"{fmt_num(df['staff_present'].sum())} / {fmt_num(df['total_staff'].sum())}")
    with c10: render_kpi("Capacity Pressure (avg)", fmt_float(df["capacity_pressure"].mean(),2) if not empty else "—", df["capacity_pressure_level"].mode().iloc[0] if not empty and len(df["capacity_pressure_level"].mode())>0 else "")

    # Model derived KPI
    if "patient_churn_prob" in df.columns:
        c11, c12 = st.columns(2)
        with c11:
            render_kpi(
                "Predicted Churn (avg)",
                f"{df['patient_churn_prob'].mean():.1%}",
                f"using {df['patient_churn_submodel'].dropna().iloc[0] if df['patient_churn_submodel'].notna().any() else '—'}",
            )
        with c12:
            pred_pos = (df["patient_churn_prob"] > 0.5).sum()
            render_kpi(
                "Predicted Churned (≥0.5)",
                fmt_num(pred_pos),
                f"vs {fmt_num(df['patient_churn'].sum())} observed",
            )

    if not empty_guard():
        # Row: Patient Volume Over Time + Churn Trend
        left, right = st.columns(2)
        with left:
            render_section_header("Patient Volume Over Time", "Daily arrivals (arrival_date)")
            daily = df.groupby(df["arrival_date"].dt.date).size().reset_index(name="patients")
            daily.columns = ["date","patients"]
            fig = px.line(daily, x="date", y="patients", markers=False)
            fig.update_traces(line_color="#0F2A44", line_width=2)
            fig.update_layout(height=300, margin=dict(l=10,r=10,t=10,b=10), xaxis_title="", yaxis_title="Patients")
            st.plotly_chart(fig, use_container_width=True)
        with right:
            render_section_header("Churn Trend", "Monthly churn rate")
            monthly = df.groupby(df["arrival_date"].dt.to_period("M")).agg(patients=("patient_id","count"), churned=("patient_churn","sum")).reset_index()
            monthly["arrival_date"] = monthly["arrival_date"].astype(str)
            monthly["churn_rate"] = monthly["churned"]/monthly["patients"]
            fig = px.line(monthly, x="arrival_date", y="churn_rate", markers=True)
            fig.update_traces(line_color="#D64545")
            fig.update_layout(height=300, margin=dict(l=10,r=10,t=10,b=10), xaxis_title="", yaxis_title="Churn rate", yaxis_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

        c1,c2 = st.columns(2)
        with c1:
            render_section_header("Patients by Service")
            svc = df["service"].value_counts().reset_index()
            svc.columns=["service","patients"]
            fig = px.bar(svc, x="service", y="patients", color="service", color_discrete_map=SERVICE_COLORS, text="patients")
            fig.update_traces(textposition="outside")
            fig.update_layout(height=300, showlegend=False, xaxis_title="", yaxis_title="Patients")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            render_section_header("Churn Rate by Service")
            svc2 = df.groupby("service").agg(patients=("patient_id","count"), churn_rate=("patient_churn","mean")).reset_index().sort_values("churn_rate", ascending=False)
            fig = px.bar(svc2, x="service", y="churn_rate", color="service", color_discrete_map=SERVICE_COLORS, text_auto=".1%")
            fig.update_layout(height=300, showlegend=False, xaxis_title="", yaxis_title="Churn rate", yaxis_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

        c1,c2 = st.columns(2)
        with c1:
            render_section_header("Satisfaction vs Churn", "Average satisfaction for churned vs retained")
            sat_churn = df.groupby("patient_churn")["satisfaction"].mean().reset_index()
            sat_churn["status"] = sat_churn["patient_churn"].map({0:"Retained",1:"Churned"})
            fig = px.bar(sat_churn, x="status", y="satisfaction", color="status", color_discrete_map={"Retained":"#0E7C7B","Churned":"#D64545"}, text_auto=".1f")
            fig.update_layout(height=300, showlegend=False, xaxis_title="", yaxis_title="Avg satisfaction")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            render_section_header("Capacity Trend", "Available beds over time (monthly avg)")
            cap = df.groupby(df["arrival_date"].dt.to_period("M")).agg(beds=("available_beds","mean")).reset_index()
            cap["arrival_date"] = cap["arrival_date"].astype(str)
            fig = px.area(cap, x="arrival_date", y="beds")
            fig.update_traces(line_color="#0E7C7B", fillcolor="rgba(14,124,123,0.15)")
            fig.update_layout(height=300, xaxis_title="", yaxis_title="Avg available beds")
            st.plotly_chart(fig, use_container_width=True)

        render_section_header("Patient Requests vs Admissions", "Monthly totals")
        req = df.groupby(df["arrival_date"].dt.to_period("M")).agg(requests=("patients_request","sum"), admitted=("patients_admitted","sum"), refused=("patients_refused","sum")).reset_index()
        req["arrival_date"] = req["arrival_date"].astype(str)
        fig = go.Figure()
        fig.add_bar(x=req["arrival_date"], y=req["requests"], name="Requests", marker_color="#0F2A44")
        fig.add_bar(x=req["arrival_date"], y=req["admitted"], name="Admitted", marker_color="#0E7C7B")
        fig.add_bar(x=req["arrival_date"], y=req["refused"], name="Refused", marker_color="#D64545")
        fig.update_layout(barmode="group", height=320, xaxis_title="", yaxis_title="Patients", legend=dict(orientation="h", y=1.02))
        st.plotly_chart(fig, use_container_width=True)

        # Key Insights
        render_section_header("Key Insights", "Auto-generated from the current filtered dataset")
        if not empty:
            insights=[]
            # highest churn service
            svc_churn = df.groupby("service")["patient_churn"].mean()
            if len(svc_churn)>1:
                top_svc = svc_churn.idxmax()
                top_val = svc_churn.max()
                avg = churn_rate
                if top_val > avg + 0.03:
                    insights.append(f"<b>{top_svc}</b> has the highest churn rate at {top_val:.1%}, { (top_val-avg)*100:.1f} pp above the filtered average ({avg:.1%}).")
                else:
                    insights.append("No service shows a churn rate more than 3 pp above the filtered average.")
            # satisfaction gap
            gap = df[df["patient_churn"]==1]["satisfaction"].mean() - df[df["patient_churn"]==0]["satisfaction"].mean() if df["patient_churn"].nunique()>1 else 0
            if abs(gap) >= 1:
                direction = "lower" if gap<0 else "higher"
                insights.append(f"Churned patients have {direction} satisfaction on average ({gap:+.1f} pts vs retained).")
            else:
                insights.append("No meaningful satisfaction gap was detected between churned and retained patients in the current filter.")
            # capacity
            avg_cp = df["capacity_pressure"].mean()
            if avg_cp >= 2:
                insights.append(f"Capacity pressure is elevated (avg {avg_cp:.2f} — High). Demand frequently exceeds available beds.")
            elif avg_cp >=1:
                insights.append(f"Capacity pressure is moderate (avg {avg_cp:.2f}). Monitor peak demand days.")
            else:
                insights.append(f"Capacity pressure is low on average ({avg_cp:.2f}).")
            # staff
            if staff_att < 0.75:
                insights.append(f"Staff attendance is low at {staff_att:.1%}. Check rostering and absenteeism drivers.")
            for ins in insights:
                render_insight(ins)

# ──────────────────────────────────────────────
# Page: Patient Analytics
# ──────────────────────────────────────────────
elif nav == "👥 Patient Analytics":
    st.markdown("# Patient Analytics")
    st.markdown('<div class="section-desc">Population structure, age groups, length of stay, satisfaction and service mix.</div>', unsafe_allow_html=True)
    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: render_kpi("Total Patients", fmt_num(len(df)))
    with c2: render_kpi("Average Age", fmt_float(df["age"].mean(),1)+" yrs" if not empty else "—")
    with c3: render_kpi("Avg Length of Stay", fmt_float(df["length_of_stay"].mean(),1)+" d" if not empty else "—")
    with c4: render_kpi("Avg Satisfaction", fmt_float(df["satisfaction"].mean(),1) if not empty else "—")
    with c5: render_kpi("Churn Rate", fmt_pct(churn_rate) if not empty else "—")
    if not empty_guard():
        left, right = st.columns(2)
        with left:
            render_section_header("Age Distribution")
            fig = px.histogram(df, x="age", nbins=30, color_discrete_sequence=["#0F2A44"])
            fig.update_layout(height=300, xaxis_title="Age", yaxis_title="Patients")
            st.plotly_chart(fig, use_container_width=True)
        with right:
            render_section_header("Age Group Analysis", "Count, churn and satisfaction by age band")
            ag = df.groupby("age_group", observed=True).agg(patients=("patient_id","count"), churn_rate=("patient_churn","mean"), avg_sat=("satisfaction","mean")).reset_index()
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_bar(x=ag["age_group"].astype(str), y=ag["patients"], name="Patients", marker_color="#0F2A44")
            fig.add_trace(go.Scatter(x=ag["age_group"].astype(str), y=ag["churn_rate"], name="Churn rate", mode="lines+markers", line=dict(color="#D64545", width=3)), secondary_y=True)
            fig.update_layout(height=320, legend=dict(orientation="h", y=1.02))
            fig.update_yaxes(title_text="Patients", secondary_y=False)
            fig.update_yaxes(title_text="Churn rate", tickformat=".0%", secondary_y=True)
            fig.update_xaxes(title_text="Age group")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(ag.style.format({"churn_rate":"{:.1%}", "avg_sat":"{:.1f}"}), use_container_width=True, hide_index=True)
        c1,c2 = st.columns(2)
        with c1:
            render_section_header("Length of Stay — Histogram")
            fig = px.histogram(df, x="length_of_stay", nbins=14, color_discrete_sequence=["#0E7C7B"])
            fig.update_layout(height=300, xaxis_title="Length of stay (days)", yaxis_title="Patients")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            render_section_header("Length of Stay — Box Plot by Churn")
            fig = px.box(df, x=df["patient_churn"].map({0:"Retained",1:"Churned"}), y="length_of_stay", color=df["patient_churn"].map({0:"Retained",1:"Churned"}), color_discrete_map={"Retained":"#0E7C7B","Churned":"#D64545"})
            fig.update_layout(height=300, showlegend=False, xaxis_title="", yaxis_title="Length of stay (days)")
            st.plotly_chart(fig, use_container_width=True)
        c1,c2 = st.columns(2)
        with c1:
            render_section_header("Satisfaction Distribution")
            fig = px.histogram(df, x="satisfaction", nbins=20, color_discrete_sequence=["#2A9D8F"])
            fig.update_layout(height=300, xaxis_title="Satisfaction", yaxis_title="Patients")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            render_section_header("Service Distribution")
            svc = df["service"].value_counts().reset_index()
            svc.columns=["service","patients"]
            fig = px.pie(svc, names="service", values="patients", hole=0.55, color="service", color_discrete_map=SERVICE_COLORS)
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

        # Churn insights by selected submodel of the patient is churned
        if "patient_churn_prob" in df.columns:
            render_section_header("Model Churn vs Observed",
                                "Predicted probability from the selected submodel, compared to observed churn")
            model_badge("patient_churn")

            c1, c2 = st.columns(2)
            with c1:
                fig = px.histogram(df, x="patient_churn_prob", nbins=40,
                                color_discrete_sequence=["#0E7C7B"])
                fig.update_layout(height=300, xaxis_title="Predicted churn probability",
                                yaxis_title="Patients")
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                cm = pd.crosstab(
                    df["patient_churn"].map({0: "Actual Retained", 1: "Actual Churned"}),
                    (df["patient_churn_prob"] > 0.5).map({False: "Pred Retained", True: "Pred Churned"}),
                )
                fig = px.imshow(cm.values, text_auto=True, aspect="auto",
                                x=cm.columns.tolist(), y=cm.index.tolist(),
                                color_continuous_scale="Blues")
                fig.update_layout(height=300, xaxis_title="", yaxis_title="",
                                title=dict(text="Confusion matrix (threshold=0.5)",
                                            font=dict(size=12)))
                st.plotly_chart(fig, use_container_width=True)

            # ROC-ish summary from the manifest
            m = model_manifests.get("patient_churn")
            if m:
                used = df["patient_churn_submodel"].dropna().iloc[0]
                met = m["metrics"].get(used, {})
                st.caption(f"Submodel **{used}** — accuracy {met.get('accuracy', 0):.3f}, "
                        f"F1 {met.get('f1', 0):.3f}, ROC-AUC {met.get('roc_auc', 0):.3f} "
                        f"(from manifest)")

        # Insights
        render_section_header("Key Insights")
        ag2 = df.groupby("age_group", observed=True)["patient_churn"].mean()
        if len(ag2)>0:
            spread = ag2.max() - ag2.min()
            if spread >= 0.05:
                render_insight(f"Churn varies by age group (spread {spread:.1%}). Highest: {ag2.idxmax()} ({ag2.max():.1%}), lowest: {ag2.idxmin()} ({ag2.min():.1%}).")
            else:
                render_insight("No meaningful churn difference across age groups in the current filter (spread < 5 pp).")
        los_gap = df[df["patient_churn"]==1]["length_of_stay"].mean() - df[df["patient_churn"]==0]["length_of_stay"].mean() if df["patient_churn"].nunique()>1 else 0
        if abs(los_gap) >= 0.5:
            render_insight(f"Length of stay differs by {los_gap:+.1f} days between churned and retained patients.")
        else:
            render_insight("Length-of-stay difference between churned and retained patients is negligible (< 0.5 days).")

# ──────────────────────────────────────────────
# Page: Patient Management
# ──────────────────────────────────────────────
elif nav == "🧑‍⚕️ Patient Management":
    st.markdown("# Patient Management")
    st.markdown('<div class="section-desc">Search, filter, and prioritise patients. Risk score is a heuristic for operational prioritisation — not a clinical prediction.</div>', unsafe_allow_html=True)
    st.info("This operational risk score is a heuristic for analytical prioritisation. It is not a clinical prediction or medical assessment.")
    if not empty_guard():
        # Filters row
        c1,c2,c3,c4 = st.columns(4)
        with c1:
            q = st.text_input("Search patient (ID or name)", placeholder="e.g. PAT-… or Richard")
        with c2:
            f_service = st.multiselect("Service (table filter)", sorted(df["service"].unique().tolist()), default=[])
        with c3:
            f_churn = st.selectbox("Churn status (table)", ["All","Churned","Retained"])
        with c4:
            f_risk = st.selectbox("Risk level", ["All","Low","Medium","High"])
        c5,c6,c7 = st.columns(3)
        with c5:
            f_sat = st.slider("Satisfaction range", 0, 100, (0,100))
        with c6:
            los_min, los_max = int(df["length_of_stay"].min()), int(df["length_of_stay"].max())
            f_los = st.slider("Length of stay (days)", los_min, los_max, (los_min, los_max))
        with c7:
            f_age = st.slider("Age (table)", int(df["age"].min()), int(df["age"].max()), (int(df["age"].min()), int(df["age"].max())))

        # tdf = filtered df for table display
        tdf = df.copy()
        if q:
            ql = q.lower()
            tdf = tdf[tdf["patient_id"].str.lower().str.contains(ql, na=False) | tdf["name"].str.lower().str.contains(ql, na=False)]
        if f_service:
            tdf = tdf[tdf["service"].isin(f_service)]
        if f_churn == "Churned":
            tdf = tdf[tdf["patient_churn"]==1]
        elif f_churn == "Retained":
            tdf = tdf[tdf["patient_churn"]==0]
        if f_risk != "All":
            tdf = tdf[tdf["risk_level"]==f_risk]
        
        tdf = tdf[(tdf["satisfaction"]>=f_sat[0]) & (tdf["satisfaction"]<=f_sat[1])]
        tdf = tdf[(tdf["length_of_stay"]>=f_los[0]) & (tdf["length_of_stay"]<=f_los[1])]
        tdf = tdf[(tdf["age"]>=f_age[0]) & (tdf["age"]<=f_age[1])]
        tdf = tdf.sort_values("risk_score", ascending=False)

        st.markdown(f"**{len(tdf):,} patients** match the table filters (sorted by risk score).")

        # Add base columns and base labels for helper features
        base_cols = [
            "patient_id","name","age","service","arrival_date","departure_date",
            "length_of_stay","satisfaction","patient_churn",
            "risk_score","risk_level",
            "capacity_pressure_level",
        ]
        base_labels = [
            "Patient ID","Name","Age","Service","Arrival","Departure",
            "LOS","Satisfaction","Churn",
            "Risk Score (heuristic)","Risk Level (heuristic)",
            "Capacity",
        ]

        # Optional model columns
        optional = [
            ("risk_level_pred",   "Risk Level (model)"),
            ("patient_churn_prob","Churn Prob (model)"),
        ]

        show_cols = [c for c in base_cols if c in tdf.columns]
        labels = [base_labels[i] for i, c in enumerate(base_cols) if c in tdf.columns]

        for col, label in optional:
            if col in tdf.columns:
                show_cols.append(col)
                labels.append(label)

        pretty = tdf[show_cols].copy()
        pretty.columns = labels
        pretty["Churn"] = pretty["Churn"].map({0:"Retained",1:"Churned"})
        pretty["Risk Score (heuristic)"] = pretty["Risk Score (heuristic)"].round(1)
        st.dataframe(pretty, use_container_width=True, height=420, hide_index=True)
        model_badge("risk_level")
        model_badge("patient_churn")

        csv = pretty.to_csv(index=False).encode("utf-8")
        st.download_button("Download filtered patients (CSV)", csv, file_name="filtered_patients.csv", mime="text/csv")

        render_section_header("Patients Requiring Attention", "Churned or High-risk patients, sorted by risk score")
        attention = (
            tdf[(tdf["patient_churn"]==1) | (tdf["risk_level_pred"].astype(str).str.lower()=="high")]
            .sort_values("patient_churn_prob", ascending=False)
            .head(20)
        )
        att_cols = ["patient_id","name","age","service","satisfaction","length_of_stay",
                    "patient_churn","patient_churn_prob","risk_level_pred"]
        att = attention[att_cols].copy()
        att.columns = ["Patient ID","Name","Age","Service","Satisfaction","LOS",
                    "Observed Churn","Predicted Churn Prob","Predicted Risk"]
        att["Observed Churn"] = att["Observed Churn"].map({0:"Retained",1:"Churned"})
        att["Predicted Churn Prob"] = att["Predicted Churn Prob"].round(3)
        st.dataframe(att, use_container_width=True, hide_index=True)

        # Insights
        render_section_header("Key Insights")
        hr = len(tdf[tdf["risk_level"]=="High"])
        hr_pct = safe_div(hr, len(tdf))
        if hr_pct > 0.15:
            render_insight(f"{hr_pct:.1%} of the displayed patients are High risk ({hr} patients) — prioritise outreach and service review.", kind="bad")
        elif hr_pct > 0.05:
            render_insight(f"{hr} patients ({hr_pct:.1%}) are High risk.", kind="warn")
        else:
            render_insight("Few high-risk patients in the current view — risk is concentrated and manageable.")

# ──────────────────────────────────────────────
# Page: Churn Analytics
# ──────────────────────────────────────────────
elif nav == "📉 Churn Analytics":
    st.markdown("# Churn Analytics")
    st.markdown('<div class="section-desc">Observed patient churn — not a prediction. Analyse churn drivers and high-risk cohorts.</div>', unsafe_allow_html=True)
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    with c1: render_kpi("Churn Rate", fmt_pct(churn_rate) if not empty else "—")
    with c2: render_kpi("Churned", fmt_num(df["patient_churn"].sum()) if not empty else "—")
    with c3: render_kpi("Retained", fmt_num((df["patient_churn"]==0).sum()) if not empty else "—")
    if not empty:
        svc_churn = df.groupby("service")["patient_churn"].mean()
        top_svc = svc_churn.idxmax() if len(svc_churn)>0 else "—"
        top_val = svc_churn.max() if len(svc_churn)>0 else 0
        with c4: render_kpi("Highest Churn Service", top_svc, fmt_pct(top_val))
        with c5: render_kpi("Avg Sat (Churned)", fmt_float(df[df["patient_churn"]==1]["satisfaction"].mean(),1) if (df["patient_churn"]==1).any() else "—")
        with c6: render_kpi("Avg LOS (Churned)", fmt_float(df[df["patient_churn"]==1]["length_of_stay"].mean(),1)+" d" if (df["patient_churn"]==1).any() else "—")
    else:
        for c in [c4,c5,c6]: 
            with c: render_kpi("—","—")
    if not empty_guard():
        c1,c2 = st.columns(2)
        with c1:
            render_section_header("Churn Distribution")
            cnt = df["patient_churn"].value_counts().reset_index()
            cnt.columns=["churn","n"]
            cnt["label"] = cnt["churn"].map({0:"Retained",1:"Churned"})
            fig = px.pie(cnt, names="label", values="n", hole=0.62, color="label", color_discrete_map={"Retained":"#0E7C7B","Churned":"#D64545"})
            fig.update_layout(height=300, showlegend=True)
            fig.update_traces(textinfo="percent+value", textposition="inside")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            render_section_header("Churn by Service")
            svc = df.groupby("service").agg(patients=("patient_id","count"), churn_rate=("patient_churn","mean")).reset_index().sort_values("churn_rate", ascending=False)
            svc["label"] = svc["churn_rate"].apply(lambda x: f"{x:.1%} (n={svc.loc[svc['churn_rate']==x,'patients'].values[0]})" if len(svc[svc['churn_rate']==x])>0 else f"{x:.1%}")
            fig = px.bar(svc, x="service", y="churn_rate", color="service", color_discrete_map=SERVICE_COLORS, text=svc["churn_rate"].apply(lambda x: f"{x:.1%}"))
            fig.update_traces(textposition="outside")
            fig.update_layout(height=300, showlegend=False, yaxis_tickformat=".0%", xaxis_title="", yaxis_title="Churn rate")
            st.plotly_chart(fig, use_container_width=True)
        c1,c2 = st.columns(2)
        with c1:
            render_section_header("Churn by Age Group")
            ag = df.groupby("age_group", observed=True).agg(patients=("patient_id","count"), churn_rate=("patient_churn","mean")).reset_index()
            fig = px.bar(ag, x=ag["age_group"].astype(str), y="churn_rate", text=ag["churn_rate"].apply(lambda x: f"{x:.1%}"), color_discrete_sequence=["#0F2A44"])
            fig.update_traces(textposition="outside")
            fig.update_layout(height=300, yaxis_tickformat=".0%", xaxis_title="Age group", yaxis_title="Churn rate")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Sample sizes: " + ", ".join([f"{r['age_group']} n={r['patients']}" for _,r in ag.iterrows()]))
        with c2:
            render_section_header("Churn by Satisfaction Band")
            sb = df.groupby("satisfaction_bin", observed=True).agg(patients=("patient_id","count"), churn_rate=("patient_churn","mean")).reset_index()
            fig = px.bar(sb, x=sb["satisfaction_bin"].astype(str), y="churn_rate", text=sb["churn_rate"].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "—"), color_discrete_sequence=["#2A9D8F"])
            fig.update_traces(textposition="outside")
            fig.update_layout(height=300, yaxis_tickformat=".0%", xaxis_title="Satisfaction band", yaxis_title="Churn rate")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Groups with n < 30 should be interpreted cautiously. " + ", ".join([f"{r['satisfaction_bin']} n={r['patients']}" for _,r in sb.iterrows()]))
        c1,c2 = st.columns(2)
        with c1:
            render_section_header("Churn vs Length of Stay")
            fig = px.box(df, x=df["patient_churn"].map({0:"Retained",1:"Churned"}), y="length_of_stay", color=df["patient_churn"].map({0:"Retained",1:"Churned"}), color_discrete_map={"Retained":"#0E7C7B","Churned":"#D64545"})
            fig.update_layout(height=320, showlegend=False, xaxis_title="", yaxis_title="Length of stay (days)")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            render_section_header("Churn vs Satisfaction — Scatter")
            # sample for performance if large
            sample = df.sample(min(2000, len(df)), random_state=42) if len(df)>2000 else df
            fig = px.scatter(sample, x="satisfaction", y="length_of_stay", color=sample["patient_churn"].map({0:"Retained",1:"Churned"}), color_discrete_map={"Retained":"#0E7C7B","Churned":"#D64545"}, opacity=0.45)
            fig.update_layout(height=320, xaxis_title="Satisfaction", yaxis_title="Length of stay (days)", legend_title="Status")
            st.plotly_chart(fig, use_container_width=True)

        render_section_header("Churn Heatmap — Service × Age Group", "Cell = churn rate; hover shows sample size. Small groups (n<30) are hatched in the table below.")
        heat = df.groupby(["service","age_group"], observed=True).agg(churn_rate=("patient_churn","mean"), n=("patient_id","count")).reset_index()
        pivot = heat.pivot(index="service", columns="age_group", values="churn_rate")
        pivot_n = heat.pivot(index="service", columns="age_group", values="n")
        # Ensure consistent column order
        cols_order = ["0–12","13–18","19–30","31–45","46–60","61+"]
        for c in cols_order:
            if c not in pivot.columns: pivot[c]=np.nan
            if c not in pivot_n.columns: pivot_n[c]=0
        pivot = pivot[cols_order]
        pivot_n = pivot_n[cols_order]
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values, x=cols_order, y=pivot.index.astype(str),
            colorscale=[[0,"#E6F4EA"],[0.5,"#FDE8A0"],[1,"#D64545"]],
            colorbar=dict(title="Churn rate", tickformat=".0%"),
            hovertemplate="Service: %{y}<br>Age: %{x}<br>Churn: %{z:.1%}<extra></extra>",
            zmin=0, zmax=1
        ))
        # annotate n
        annotations=[]
        for i, svc in enumerate(pivot.index):
            for j, ag in enumerate(cols_order):
                v = pivot.iloc[i,j]
                n = pivot_n.iloc[i,j]
                if pd.notna(v):
                    annotations.append(dict(x=ag, y=svc, text=f"{v:.0%}<br>n={int(n)}", showarrow=False, font=dict(size=10, color="black" if v<0.6 else "white")))
        fig.update_layout(height=360, annotations=annotations, xaxis_title="Age group", yaxis_title="Service")
        st.plotly_chart(fig, use_container_width=True)
        # table with caution flag
        heat["caution"] = heat["n"] < 30
        heat_display = heat.copy()
        heat_display["churn_rate"] = heat_display["churn_rate"].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "—")
        heat_display["flag"] = heat_display["caution"].apply(lambda x: "⚠️ n<30" if x else "—")
        heat_display = heat_display[["service","age_group","churn_rate","n","flag"]].sort_values(["service","age_group"])
        heat_display.columns=["Service","Age Group","Churn Rate","n","Note"]
        st.dataframe(heat_display, use_container_width=True, hide_index=True)

        # Insights
        render_section_header("Key Insights")
        # satisfaction band trend
        sb_valid = df.groupby("satisfaction_bin", observed=True)["patient_churn"].mean()
        if len(sb_valid.dropna()) >= 2:
            low_sat = sb_valid.get("41–60", np.nan)
            high_sat = sb_valid.get("81–100", np.nan)
            if pd.notna(low_sat) and pd.notna(high_sat) and abs(low_sat - high_sat) >= 0.05:
                render_insight(f"Churn is {low_sat:.1%} in the 41–60 satisfaction band vs {high_sat:.1%} at 81–100 — dissatisfaction is associated with higher churn (correlation ≠ causation).")
            else:
                render_insight("No strong churn gradient across satisfaction bands in the current filter.")
        # LOS
        los_c = df[df["patient_churn"]==1]["length_of_stay"].mean()
        los_r = df[df["patient_churn"]==0]["length_of_stay"].mean()
        if pd.notna(los_c) and pd.notna(los_r):
            if los_c < los_r - 0.5:
                render_insight(f"Churned patients have a shorter average stay ({los_c:.1f} d vs {los_r:.1f} d retained). Short stays may signal early disengagement — investigate discharge reasons, not as a causal claim.")
            elif los_c > los_r + 0.5:
                render_insight(f"Churned patients stay longer on average ({los_c:.1f} d vs {los_r:.1f} d). Prolonged stays may erode experience.")
            else:
                render_insight("Length-of-stay difference by churn status is negligible in the current view.")

# ──────────────────────────────────────────────
# Page: Hospital Operations
# ──────────────────────────────────────────────
elif nav == "🛏️ Hospital Operations":
    st.markdown("# Hospital Operations")
    st.markdown('<div class="section-desc">Demand, admissions, refusals, and capacity pressure.</div>', unsafe_allow_html=True)
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    with c1: render_kpi("Patient Requests", fmt_num(df["patients_request"].sum()) if not empty else "—")
    with c2: render_kpi("Admitted", fmt_num(df["patients_admitted"].sum()) if not empty else "—")
    with c3: render_kpi("Refused", fmt_num(df["patients_refused"].sum()) if not empty else "—")
    with c4: render_kpi("Admission Rate", fmt_pct(admission_rate) if not empty else "—")
    with c5: render_kpi("Refusal Rate", fmt_pct(refusal_rate) if not empty else "—")
    with c6: render_kpi("Avg Available Beds", fmt_float(df["available_beds"].mean(),1) if not empty else "—")
    
    if not empty_guard():    
        # Monthly requests vs admissions
        render_section_header("Requests vs Admissions — Monthly")
        mon = df.groupby(df["arrival_date"].dt.to_period("M")).agg(requests=("patients_request","sum"), admitted=("patients_admitted","sum"), refused=("patients_refused","sum")).reset_index()
        mon["arrival_date"] = mon["arrival_date"].astype(str)
        fig = go.Figure()
        fig.add_bar(x=mon["arrival_date"], y=mon["requests"], name="Requests", marker_color="#0F2A44")
        fig.add_bar(x=mon["arrival_date"], y=mon["admitted"], name="Admitted", marker_color="#0E7C7B")
        fig.add_bar(x=mon["arrival_date"], y=mon["refused"], name="Refused", marker_color="#D64545")
        fig.update_layout(barmode="group", height=320, xaxis_title="", yaxis_title="Patients", legend=dict(orientation="h", y=1.02))
        st.plotly_chart(fig, use_container_width=True)
        
        c1,c2 = st.columns(2)
        with c1:
            render_section_header("Admission Rate by Service")
            svc = df.groupby("service").agg(req=("patients_request","sum"), adm=("patients_admitted","sum")).reset_index()
            svc["admission_rate"] = svc["adm"]/svc["req"]
            svc = svc.sort_values("admission_rate", ascending=False)
            fig = px.bar(svc, x="service", y="admission_rate", color="service", color_discrete_map=SERVICE_COLORS, text=svc["admission_rate"].apply(lambda x: f"{x:.1%}"))
            fig.update_traces(textposition="outside")
            fig.update_layout(height=300, showlegend=False, yaxis_tickformat=".0%", xaxis_title="", yaxis_title="Admission rate")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            render_section_header("Refusal Rate by Service")
            svc["refusal_rate"] = 1 - svc["admission_rate"]
            # also compute directly from refused/req to account for rounding
            svc2 = df.groupby("service").agg(req=("patients_request","sum"), ref=("patients_refused","sum")).reset_index()
            svc2["refusal_rate"] = svc2["ref"]/svc2["req"]
            svc2 = svc2.sort_values("refusal_rate", ascending=False)
            fig = px.bar(svc2, x="service", y="refusal_rate", color="service", color_discrete_map=SERVICE_COLORS, text=svc2["refusal_rate"].apply(lambda x: f"{x:.1%}"))
            fig.update_traces(textposition="outside")
            fig.update_layout(height=300, showlegend=False, yaxis_tickformat=".0%", xaxis_title="", yaxis_title="Refusal rate")
            st.plotly_chart(fig, use_container_width=True)
        
        c1,c2 = st.columns(2)
        with c1:
            render_section_header("Available Beds Over Time")
            beds = df.groupby(df["arrival_date"].dt.to_period("M")).agg(beds=("available_beds","mean")).reset_index()
            beds["arrival_date"] = beds["arrival_date"].astype(str)
            fig = px.line(beds, x="arrival_date", y="beds", markers=True)
            fig.update_traces(line_color="#0E7C7B")
            fig.update_layout(height=300, xaxis_title="", yaxis_title="Avg available beds")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            render_section_header("Demand vs Capacity — Monthly Avg")
            dc = df.groupby(df["arrival_date"].dt.to_period("M")).agg(req=("patients_request","mean"), beds=("available_beds","mean"), adm=("patients_admitted","mean")).reset_index()
            dc["arrival_date"] = dc["arrival_date"].astype(str)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dc["arrival_date"], y=dc["req"], name="Requests (avg)", mode="lines+markers", line=dict(color="#0F2A44")))
            fig.add_trace(go.Scatter(x=dc["arrival_date"], y=dc["beds"], name="Available beds (avg)", mode="lines+markers", line=dict(color="#0E7C7B")))
            fig.add_trace(go.Scatter(x=dc["arrival_date"], y=dc["adm"], name="Admitted (avg)", mode="lines+markers", line=dict(color="#2A9D8F", dash="dash")))
            fig.update_layout(height=300, xaxis_title="", yaxis_title="Patients / Beds")
            st.plotly_chart(fig, use_container_width=True)
        
        # Capacity pressure distribution
        render_section_header("Capacity Pressure Distribution", "capacity_pressure = patients_request / (available_beds + 1) — Low <1, Moderate 1–2, High 2–4, Critical >4")
        cpd = df["capacity_pressure_level"].value_counts().reindex(["Low","Moderate","High","Critical"]).reset_index()
        cpd.columns=["level","count"]
        cpd = cpd.dropna()
        fig = px.bar(cpd, x="level", y="count", color="level", color_discrete_map={"Low":"#0E7C7B","Moderate":"#E6A817","High":"#E67E22","Critical":"#D64545"}, text="count")
        fig.update_traces(textposition="outside")
        fig.update_layout(height=300, showlegend=False, xaxis_title="Pressure level", yaxis_title="Records")
        st.plotly_chart(fig, use_container_width=True)

        render_section_header("Key Insights")
        if refusal_rate > 0.4:
            render_insight(f"Refusal rate is high at {refusal_rate:.1%} — demand exceeds capacity on many days. Review bed allocation and admission triage.", kind="bad")
        elif refusal_rate > 0.25:
            render_insight(f"Refusal rate is {refusal_rate:.1%} — moderate capacity strain.", kind="warn")
        else:
            render_insight(f"Refusal rate is {refusal_rate:.1%} — capacity is generally meeting demand.")
        
        # service with highest refusal
        svc_ref = df.groupby("service").agg(req=("patients_request","sum"), ref=("patients_refused","sum"))
        svc_ref["rate"] = svc_ref["ref"]/svc_ref["req"]
        worst = svc_ref["rate"].idxmax()
        if svc_ref.loc[worst,"rate"] > refusal_rate + 0.05:
            render_insight(f"<b>{worst}</b> has the highest refusal rate ({svc_ref.loc[worst,'rate']:.1%}) — investigate service-specific bottlenecks.")

        # Model informed service risk
        if "risk_score_pred" in df.columns:
            render_section_header("Predicted Risk Score by Service",
                                "Average predicted risk_score from the selected submodel")
            rs = df.groupby("service")["risk_score_pred"].mean().reset_index().sort_values("risk_score_pred", ascending=False)
            fig = px.bar(rs, x="service", y="risk_score_pred", color="service",
                        color_discrete_map=SERVICE_COLORS,
                        text=rs["risk_score_pred"].apply(lambda x: f"{x:.2f}"))
            fig.update_traces(textposition="outside")
            fig.update_layout(height=300, showlegend=False,
                            xaxis_title="", yaxis_title="Avg predicted risk score")
            st.plotly_chart(fig, use_container_width=True)
            model_badge("risk_score")

# ──────────────────────────────────────────────
# Page: Staff Analytics
# ──────────────────────────────────────────────
elif nav == "👨‍⚕️ Staff Analytics":
    st.markdown("# Staff Analytics")
    st.markdown('<div class="section-desc">Staffing levels, morale, attendance and associations with patient outcomes. Correlation indicates association and does not establish causation.</div>', unsafe_allow_html=True)
    
    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: render_kpi("Total Staff (sum)", fmt_num(df["total_staff"].sum()) if not empty else "—", "across all records")
    with c2: render_kpi("Staff Present", fmt_num(df["staff_present"].sum()) if not empty else "—")
    with c3: render_kpi("Staff Absent", fmt_num(df["staff_absent"].sum()) if not empty else "—")
    with c4: render_kpi("Attendance Rate", fmt_pct(staff_att) if not empty else "—")
    with c5: render_kpi("Avg Staff Morale", fmt_float(df["staff_morale"].mean(),1) if not empty else "—", "/ 100")
    
    if not empty_guard():
        c1,c2 = st.columns(2)
        with c1:
            render_section_header("Staff Presence Over Time (monthly avg)")
            sp = df.groupby(df["arrival_date"].dt.to_period("M")).agg(present=("staff_present","mean"), absent=("staff_absent","mean"), total=("total_staff","mean")).reset_index()
            sp["arrival_date"] = sp["arrival_date"].astype(str)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=sp["arrival_date"], y=sp["present"], name="Present", mode="lines+markers", line=dict(color="#0E7C7B")))
            fig.add_trace(go.Scatter(x=sp["arrival_date"], y=sp["absent"], name="Absent", mode="lines+markers", line=dict(color="#D64545")))
            fig.update_layout(height=300, xaxis_title="", yaxis_title="Staff (avg per record)")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            render_section_header("Staff Absence Distribution")
            fig = px.histogram(df, x="staff_absent", nbins=30, color_discrete_sequence=["#D64545"])
            fig.update_layout(height=300, xaxis_title="Staff absent", yaxis_title="Records")
            st.plotly_chart(fig, use_container_width=True)
        
        c1,c2 = st.columns(2)
        with c1:
            render_section_header("Staff Morale Trend (monthly avg)")
            mo = df.groupby(df["arrival_date"].dt.to_period("M")).agg(morale=("staff_morale","mean")).reset_index()
            mo["arrival_date"] = mo["arrival_date"].astype(str)
            fig = px.line(mo, x="arrival_date", y="morale", markers=True)
            fig.update_traces(line_color="#0F2A44")
            fig.update_layout(height=300, xaxis_title="", yaxis_title="Morale")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            render_section_header("Staff Morale vs Patient Satisfaction")
            sample = df.sample(min(2000,len(df)), random_state=7) if len(df)>2000 else df
            fig = px.scatter(sample, x="staff_morale", y="satisfaction", trendline="ols", trendline_color_override="#D64545", color_discrete_sequence=["#0E7C7B"], opacity=0.4)
            corr = df["staff_morale"].corr(df["satisfaction"])
            fig.update_layout(height=300, xaxis_title="Staff morale", yaxis_title="Patient satisfaction", title=dict(text=f"r = {corr:.2f}", font=dict(size=12)))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Correlation indicates association and does not establish causation.")
        render_section_header("Staff Absence vs Patient Churn")

        # bin absence into levels
        df["_abs_bin"] = pd.cut(df["staff_absent"], bins=[-1,2,8,20,100], labels=["0–2","3–8","9–20","21+"])
        ab = df.groupby("_abs_bin", observed=True).agg(patients=("patient_id","count"), churn_rate=("patient_churn","mean")).reset_index()
        fig = px.bar(ab, x="_abs_bin", y="churn_rate", text=ab["churn_rate"].apply(lambda x: f"{x:.1%}"), color_discrete_sequence=["#0F2A44"])
        fig.update_traces(textposition="outside")
        fig.update_layout(height=300, yaxis_tickformat=".0%", xaxis_title="Staff absent (per record)", yaxis_title="Churn rate")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Groups: " + ", ".join([f"{r['_abs_bin']} n={r['patients']}" for _,r in ab.iterrows()]))

        # Correlation heatmap (selected)
        render_section_header("Correlation Heatmap", "Selected operational variables")
        cols = [c for c in ["satisfaction","patient_satisfaction","length_of_stay","staff_morale","staff_absent","staff_present","patients_refused","patients_admitted","available_beds","patients_request","patient_churn"] if c in df.columns]
        corr_mat = df[cols].corr(numeric_only=True)
        fig = go.Figure(data=go.Heatmap(z=corr_mat.values, x=corr_mat.columns, y=corr_mat.columns, colorscale="RdBu", zmid=0, zmin=-1, zmax=1, text=np.round(corr_mat.values,2), texttemplate="%{text}", hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>"))
        fig.update_layout(height=420, xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Correlation indicates association and does not establish causation.")

        render_section_header("Key Insights")
        corr_morale_sat = df["staff_morale"].corr(df["satisfaction"])
        if abs(corr_morale_sat) >= 0.3:
            render_insight(f"Staff morale and patient satisfaction show a {'positive' if corr_morale_sat>0 else 'negative'} association (r={corr_morale_sat:.2f}). Investigate staffing and experience links — correlation ≠ causation.")
        else:
            render_insight(f"Staff morale and satisfaction are weakly associated (r={corr_morale_sat:.2f}) in the current filter.")
        corr_abs_churn = df["staff_absent"].corr(df["patient_churn"])
        if abs(corr_abs_churn) >= 0.2:
            render_insight(f"Staff absence and churn show r={corr_abs_churn:.2f} — monitor rostering during peaks.")
        else:
            render_insight("No strong association between staff absence and churn in the current filter.")

        # Predicted staff metrics
        if "staff_staff_morale_pred" in df.columns:
            render_section_header("Predicted vs Observed Staff Morale",
                                "From the selected submodel of staff_staff_morale")
            mo = df.groupby(df["arrival_date"].dt.to_period("M")).agg(
                observed=("staff_morale", "mean"),
                predicted=("staff_staff_morale_pred", "mean"),
            ).reset_index()
            mo["arrival_date"] = mo["arrival_date"].astype(str)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=mo["arrival_date"], y=mo["observed"], name="Observed",
                                    mode="lines+markers", line=dict(color="#0F2A44")))
            fig.add_trace(go.Scatter(x=mo["arrival_date"], y=mo["predicted"], name="Predicted",
                                    mode="lines+markers", line=dict(color="#D64545", dash="dash")))
            fig.update_layout(height=300, xaxis_title="", yaxis_title="Morale")
            st.plotly_chart(fig, use_container_width=True)
            model_badge("staff_staff_morale")

# ──────────────────────────────────────────────
# Page: Service Performance
# ──────────────────────────────────────────────
elif nav == "📊 Service Performance":
    st.markdown("# Service Performance")
    st.markdown('<div class="section-desc">Comparative performance across services with a transparent Service Health Score.</div>', unsafe_allow_html=True)
    if not empty_guard():
        tbl = df.groupby("service").agg(
            patients=("patient_id","count"),
            churn_rate=("patient_churn","mean"),
            avg_satisfaction=("satisfaction","mean"),
            avg_los=("length_of_stay","mean"),
            requests=("patients_request","sum"),
            admitted=("patients_admitted","sum"),
            refused=("patients_refused","sum"),
            avg_beds=("available_beds","mean"),
            avg_morale=("staff_morale","mean"),
            avg_present=("staff_present","mean"),
            avg_total=("total_staff","mean"),
        ).reset_index()
        tbl["admission_rate"] = tbl["admitted"]/tbl["requests"]
        tbl["refusal_rate"] = tbl["refused"]/tbl["requests"]
        tbl["attendance"] = tbl["avg_present"]/tbl["avg_total"]
        
        # Proper capacity pressure avg per service
        cp_svc = df.groupby("service")["capacity_pressure"].mean().reset_index().rename(columns={"capacity_pressure":"avg_cp"})
        tbl = tbl.merge(cp_svc, on="service", how="left")
        
        # Health score: 35% satisfaction (norm 0-100), 25% low churn (1-churn), 20% admission, 20% attendance — all 0-1 scaled then 0-100
        tbl["health_score"] = (0.35*(tbl["avg_satisfaction"]/100) + 0.25*(1-tbl["churn_rate"]) + 0.20*tbl["admission_rate"].fillna(0) + 0.20*tbl["attendance"].fillna(0))*100

        tbl_show = tbl[["service","patients","churn_rate","avg_satisfaction","avg_los","requests","admitted","refused","admission_rate","refusal_rate","avg_beds","avg_morale","attendance","avg_cp","health_score"]].copy()
        tbl_show.columns=["Service","Patients","Churn Rate","Avg Satisfaction","Avg LOS","Requests","Admitted","Refused","Admission Rate","Refusal Rate","Avg Beds","Staff Morale","Attendance","Avg Capacity Pressure","Health Score"]
        
        # Format
        st.dataframe(
            tbl_show.style.format({
                "Churn Rate":"{:.1%}", "Avg Satisfaction":"{:.1f}", "Avg LOS":"{:.1f}",
                "Admission Rate":"{:.1%}", "Refusal Rate":"{:.1%}", "Avg Beds":"{:.1f}",
                "Staff Morale":"{:.1f}", "Attendance":"{:.1%}", "Avg Capacity Pressure":"{:.2f}", "Health Score":"{:.1f}"
            }).background_gradient(subset=["Health Score"], cmap="Greens"),
            use_container_width=True, hide_index=True
        )
        with st.expander("How is the Service Health Score calculated?"):
            st.markdown("""
            Service Health Score (0–100, higher is better):
            - 35% Patient Satisfaction (satisfaction / 100)
            - 25% Low Churn (1 − churn_rate)
            - 20% Admission Performance (admitted / requests)
            - 20% Staff Attendance (present / total_staff)
            
            Score = 100 × (0.35·sat + 0.25·(1−churn) + 0.20·admission_rate + 0.20·attendance)
            
            All components are 0–1 normalised before weighting. This is a transparent operational composite — not a clinical quality score.
            """)

        # Rankings
        st.markdown("### Rankings")
        r1,r2,r3,r4,r5 = st.columns(5)
        def rank_card(col, title, series, fmt="{:.1%}", ascending=False):
            s = series.sort_values(ascending=ascending)
            top = s.index[0]; val = s.iloc[0]
            svc = tbl.loc[top,"service"]
            with col:
                render_kpi(title, f"{svc}", fmt.format(val) if isinstance(val,float) else str(val))
        rank_card(r1,"Highest Churn", tbl.set_index(tbl.index)["churn_rate"], "{:.1%}", ascending=False)
        
        # lowest satisfaction
        low_sat_idx = tbl["avg_satisfaction"].idxmin()
        with r2: render_kpi("Lowest Satisfaction", tbl.loc[low_sat_idx,"service"], f"{tbl.loc[low_sat_idx,'avg_satisfaction']:.1f}")
        high_ref_idx = tbl["refusal_rate"].idxmax()
        with r3: render_kpi("Highest Refusal Rate", tbl.loc[high_ref_idx,"service"], f"{tbl.loc[high_ref_idx,'refusal_rate']:.1%}")
        high_cp_idx = tbl["avg_cp"].idxmax()
        with r4: render_kpi("Highest Capacity Pressure", tbl.loc[high_cp_idx,"service"], f"{tbl.loc[high_cp_idx,'avg_cp']:.2f}")
        low_att_idx = tbl["attendance"].idxmin()
        with r5: render_kpi("Lowest Attendance", tbl.loc[low_att_idx,"service"], f"{tbl.loc[low_att_idx,'attendance']:.1%}")

        # Health score bar
        render_section_header("Service Health Score Ranking")
        tbl_sorted = tbl.sort_values("health_score", ascending=True)
        fig = px.bar(tbl_sorted, x="health_score", y="service", orientation="h", color="health_score", color_continuous_scale="Greens", text=tbl_sorted["health_score"].apply(lambda x: f"{x:.1f}"))
        fig.update_traces(textposition="outside")
        fig.update_layout(height=300, xaxis_title="Health Score (0–100)", yaxis_title="", showlegend=False, xaxis_range=[0,100])
        st.plotly_chart(fig, use_container_width=True)

        render_section_header("Key Insights")
        best = tbl.loc[tbl["health_score"].idxmax(),"service"]
        worst = tbl.loc[tbl["health_score"].idxmin(),"service"]
        render_insight(f"<b>{best}</b> leads on the composite Health Score; <b>{worst}</b> trails — prioritise the drivers (churn, satisfaction, refusals, attendance) for the lowest-ranked service.")
        
        # churn gap
        churn_gap = tbl["churn_rate"].max() - tbl["churn_rate"].min()
        if churn_gap >= 0.05:
            render_insight(f"Churn varies {churn_gap:.1%} across services — service-level intervention is warranted.")

        # Models predicted service performance
        if "service_performance_pred" in df.columns:
            svc_health = df.groupby("service")["service_performance_pred"].mean().reset_index()
            svc_health = svc_health.rename(columns={"service_performance_pred": "model_health"})
            svc_health = svc_health.sort_values("model_health", ascending=False)
            render_section_header("Model-Predicted Service Health",
                                "Average predicted service_performance from the selected submodel")
            fig = px.bar(svc_health, x="service", y="model_health", color="service",
                        color_discrete_map=SERVICE_COLORS,
                        text=svc_health["model_health"].apply(lambda x: f"{x:.2f}"))
            fig.update_traces(textposition="outside")
            fig.update_layout(height=300, showlegend=False,
                            xaxis_title="", yaxis_title="Predicted health")
            st.plotly_chart(fig, use_container_width=True)
            model_badge("service_performance")

# ──────────────────────────────────────────────
# Page: Operational Risk
# ──────────────────────────────────────────────
elif nav == "⚠️ Operational Risk":
    st.markdown("# Operational Risk Monitoring")
    st.markdown('<div class="section-desc">Heuristic risk and capacity pressure monitoring. All thresholds are documented and not clinical standards.</div>', unsafe_allow_html=True)
    st.info("This operational risk score is a heuristic for analytical prioritisation. It is not a clinical prediction or medical assessment.")
    if not empty_guard():
        c1,c2,c3,c4 = st.columns(4)
        with c1: render_kpi("Avg Risk Score", fmt_float(df["risk_score"].mean(),1)+" /100")
        with c2: render_kpi("High Risk", fmt_num((df["risk_level"]=="High").sum()), fmt_pct(safe_div((df["risk_level"]=="High").sum(), len(df))))
        with c3: render_kpi("Medium Risk", fmt_num((df["risk_level"]=="Medium").sum()), fmt_pct(safe_div((df["risk_level"]=="Medium").sum(), len(df))))
        with c4: render_kpi("Critical Capacity", fmt_num((df["capacity_pressure_level"]=="Critical").sum()), fmt_pct(safe_div((df["capacity_pressure_level"]=="Critical").sum(), len(df))))
        with st.expander("How is the Operational Risk Score calculated?", expanded=False):
            st.markdown("""
            Risk Score (0–100, higher = higher operational attention):
            - 40% Dissatisfaction risk: (100 − satisfaction) / 100
            - 25% Length-of-stay risk: (LOS − min LOS) / (max LOS − min LOS)
            - 20% Refusal pressure: patients_refused / patients_request (clipped 0–1)
            - 15% Low morale risk: (100 − staff_morale) / 100
            
            Score = 100 × (0.40·dissat + 0.25·los + 0.20·refusal + 0.15·morale)
            
            Levels: Low ≤30, Medium 31–60, High >60.
            
            Capacity Pressure = patients_request / (available_beds + 1). Levels: Low <1, Moderate 1–2, High 2–4, Critical >4. Thresholds are operational heuristics, not clinical standards.
            """)
        c1,c2 = st.columns(2)
        with c1:
            render_section_header("Risk Level Distribution")
            rc = df["risk_level"].value_counts().reindex(["Low","Medium","High"]).reset_index()
            rc.columns=["level","count"]
            fig = px.bar(rc, x="level", y="count", color="level", color_discrete_map={"Low":"#0E7C7B","Medium":"#E6A817","High":"#D64545"}, text="count")
            fig.update_traces(textposition="outside")
            fig.update_layout(height=300, showlegend=False, xaxis_title="Risk level", yaxis_title="Patients")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            render_section_header("Risk by Service")
            rs = df.groupby("service")["risk_score"].mean().reset_index().sort_values("risk_score", ascending=False)
            fig = px.bar(rs, x="service", y="risk_score", color="service", color_discrete_map=SERVICE_COLORS, text=rs["risk_score"].apply(lambda x: f"{x:.1f}"))
            fig.update_traces(textposition="outside")
            fig.update_layout(height=300, showlegend=False, xaxis_title="", yaxis_title="Avg risk score")
            st.plotly_chart(fig, use_container_width=True)
        c1,c2 = st.columns(2)
        with c1:
            render_section_header("Risk vs Satisfaction — Scatter")
            sample = df.sample(min(2000,len(df)), random_state=11) if len(df)>2000 else df
            fig = px.scatter(sample, x="satisfaction", y="risk_score", color="risk_level", color_discrete_map={"Low":"#0E7C7B","Medium":"#E6A817","High":"#D64545"}, opacity=0.5)
            fig.update_layout(height=320, xaxis_title="Satisfaction", yaxis_title="Risk score")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            render_section_header("Capacity Pressure — Service Avg")
            cp = df.groupby("service")["capacity_pressure"].mean().reset_index().sort_values("capacity_pressure", ascending=False)
            fig = px.bar(cp, x="service", y="capacity_pressure", color="service", color_discrete_map=SERVICE_COLORS, text=cp["capacity_pressure"].apply(lambda x: f"{x:.2f}"))
            fig.update_traces(textposition="outside")
            fig.update_layout(height=320, showlegend=False, xaxis_title="", yaxis_title="Avg capacity pressure")
            st.plotly_chart(fig, use_container_width=True)
        render_section_header("Highest-Risk Patients")
        top = df.sort_values("risk_score", ascending=False).head(15)[["patient_id","name","age","service","satisfaction","length_of_stay","patients_refused","staff_morale","risk_score","risk_level","capacity_pressure_level"]]
        top.columns=["Patient ID","Name","Age","Service","Satisfaction","LOS","Refused","Morale","Risk Score","Risk Level","Capacity"]
        top["Risk Score"] = top["Risk Score"].round(1)
        st.dataframe(top, use_container_width=True, hide_index=True)

        render_section_header("Key Insights")
        hr_rate = safe_div((df["risk_level"]=="High").sum(), len(df))
        if hr_rate > 0.2:
            render_insight(f"{hr_rate:.1%} are High risk — operational attention is broadly needed.", kind="bad")
        elif hr_rate > 0.08:
            render_insight(f"{hr_rate:.1%} are High risk — focus on the highest-scoring segment.", kind="warn")
        else:
            render_insight("High-risk share is contained — risk is concentrated in a small cohort.")
        cp_crit = safe_div((df["capacity_pressure_level"]=="Critical").sum(), len(df))
        if cp_crit > 0.1:
            render_insight(f"{cp_crit:.1%} of records are at Critical capacity pressure — urgent capacity review recommended.", kind="bad")

# ──────────────────────────────────────────────
# Page: Management Insights
# ──────────────────────────────────────────────
elif nav == "💡 Management Insights":
    st.markdown("# Management Insights")
    st.markdown('<div class="section-desc">Automatically generated from the current filtered dataset. No conclusions are drawn from groups with insufficient evidence (n&lt;30 flagged).</div>', unsafe_allow_html=True)
    if empty_guard():
        st.stop()
    insights=[]

    # 1. Churn vs satisfaction
    if df["patient_churn"].nunique()>1:
        s_c = df[df["patient_churn"]==1]["satisfaction"].mean()
        s_r = df[df["patient_churn"]==0]["satisfaction"].mean()
        if pd.notna(s_c) and pd.notna(s_r):
            diff = s_c - s_r
            if abs(diff) >= 2:
                insights.append(("Patient Experience & Churn", f"Churned patients average {s_c:.1f} satisfaction vs {s_r:.1f} for retained (Δ {diff:+.1f}). Lower experience is associated with churn — review exit feedback and service recovery (association, not causation)."))
            else:
                insights.append(("Patient Experience & Churn", "No meaningful satisfaction gap between churned and retained patients in the current view."))
    # 2. Capacity
    avg_cp = df["capacity_pressure"].mean()
    crit_share = safe_div((df["capacity_pressure_level"]=="Critical").sum(), len(df))
    if avg_cp >= 2 or crit_share > 0.08:
        insights.append(("Capacity & Access", f"Capacity pressure averages {avg_cp:.2f} with {crit_share:.1%} at Critical level. Demand exceeds bed availability — consider dynamic bed management and admission smoothing."))
    elif avg_cp >= 1:
        insights.append(("Capacity & Access", f"Capacity pressure is moderate ({avg_cp:.2f}). Monitor peak days; no systemic overload in this window."))
    else:
        insights.append(("Capacity & Access", "Capacity is sufficient on average — pressure is low."))
    
    # 3. Service outlier
    svc_churn = df.groupby("service")["patient_churn"].mean()
    if len(svc_churn)>1:
        gap = svc_churn.max() - svc_churn.min()
        if gap >= 0.05:
            insights.append(("Service Variation", f"Churn ranges from {svc_churn.min():.1%} to {svc_churn.max():.1%} across services (gap {gap:.1%}). <b>{svc_churn.idxmax()}</b> is the outlier to prioritise."))
        else:
            insights.append(("Service Variation", "Churn is consistent across services in the current filter (gap < 5 pp)."))
    
    # 4. Staff morale
    morale = df["staff_morale"].mean()
    corr_ms = df["staff_morale"].corr(df["satisfaction"])
    if morale < 65:
        insights.append(("Workforce", f"Average staff morale is {morale:.1f}/100 — below 65. Low morale coincides with patient experience risk (r={corr_ms:.2f}). Review workload and support."))
    elif morale < 75:
        insights.append(("Workforce", f"Staff morale averages {morale:.1f}/100 — moderate. Track trend alongside attendance."))
    else:
        insights.append(("Workforce", f"Staff morale is healthy at {morale:.1f}/100."))
    
    # 5. Attendance
    if staff_att < 0.7:
        insights.append(("Attendance", f"Staff attendance is {staff_att:.1%} — below 70%. Absenteeism may amplify capacity strain."))
    
    # 6. Refusal
    if refusal_rate > 0.35:
        insights.append(("Patient Flow", f"Refusal rate is {refusal_rate:.1%} — more than one in three requests are refused. Triage and capacity allocation need review."))
    
    # 7. LOS
    if df["patient_churn"].nunique()>1:
        los_c = df[df["patient_churn"]==1]["length_of_stay"].mean()
        los_r = df[df["patient_churn"]==0]["length_of_stay"].mean()
        if pd.notna(los_c) and pd.notna(los_r) and abs(los_c - los_r) >= 0.7:
            insights.append(("Length of Stay", f"LOS gap: churned {los_c:.1f} d vs retained {los_r:.1f} d. Examine early discharges and prolonged stays separately."))
    
    # 8. Risk concentration
    hr = safe_div((df["risk_level"]=="High").sum(), len(df))
    if hr > 0.15:
        insights.append(("Risk Concentration", f"{hr:.1%} are High operational risk. A focused case-review list is warranted."))

    if not insights:
        render_insight("No meaningful differences were detected in the current filtered data. Try broadening the date or service filters.")
    else:
        for title, body in insights:
            st.markdown(f"**{title}**")
            render_insight(body)
            st.markdown("")

    # Model-informed patient churn
    if "patient_churn_prob" in df.columns:
        high_risk_pool = df[df["patient_churn_prob"] > 0.7]
        if len(high_risk_pool) > 0:
            insights.append((
                "Model Signal",
                f"{len(high_risk_pool)} patients ({len(high_risk_pool)/len(df):.1%}) have predicted "
                f"churn probability > 0.7 using the **{df['patient_churn_submodel'].dropna().iloc[0]}** "
                f"submodel. Prioritise these for outreach — predicted, not confirmed.",
            ))

    # Summary table
    st.markdown("### At-a-Glance — Filtered KPIs")
    kpi_df = pd.DataFrame([{
        "Patients": len(df),
        "Churn Rate": f"{churn_rate:.1%}",
        "Avg Satisfaction": f"{df['satisfaction'].mean():.1f}",
        "Avg LOS (days)": f"{df['length_of_stay'].mean():.1f}",
        "Admission Rate": f"{admission_rate:.1%}",
        "Refusal Rate": f"{refusal_rate:.1%}",
        "Avg Beds": f"{df['available_beds'].mean():.1f}",
        "Staff Attendance": f"{staff_att:.1%}",
        "Staff Morale": f"{df['staff_morale'].mean():.1f}",
        "Avg Risk Score": f"{df['risk_score'].mean():.1f}",
        "Avg Capacity Pressure": f"{df['capacity_pressure'].mean():.2f}"
    }])
    st.dataframe(kpi_df, use_container_width=True, hide_index=True)

# ──────────────────────────────────────────────
# Page: Data Explorer
# ──────────────────────────────────────────────
elif nav == "🔎 Data Explorer":
    st.markdown("# Data Explorer")
    st.markdown('<div class="section-desc">Data quality, schema and preview. Download respects current global filters.</div>', unsafe_allow_html=True)
    
    # Quality KPIs
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    with c1: render_kpi("Rows (filtered)", fmt_num(len(df)), f"raw {len(df_raw):,}")
    with c2: render_kpi("Columns", str(df_raw.shape[1]))
    with c3: render_kpi("Missing Values", fmt_num(df_raw.isna().sum().sum()), "in raw dataset")
    with c4: render_kpi("Duplicate Patient IDs", fmt_num(df_raw.duplicated(subset=["patient_id"]).sum()))
    with c5: render_kpi("Duplicate Rows", fmt_num(df_raw.duplicated().sum()))
    with c6: render_kpi("Services", str(df_raw["service"].nunique()), ", ".join(sorted(df_raw["service"].unique())))
    c7,c8 = st.columns(2)
    with c7: render_kpi("Date Range (raw)", f"{df_raw['arrival_date'].min().date()} → {df_raw['arrival_date'].max().date()}")
    with c8: render_kpi("Churn Distribution (raw)", f"{(df_raw['patient_churn']==1).sum():,} churned / {(df_raw['patient_churn']==0).sum():,} retained", f"{df_raw['patient_churn'].mean():.1%} churn rate")

    # Validation warnings
    render_section_header("Data Validation")
    issues=[]
    if df_raw.duplicated(subset=["patient_id"]).sum() > 0:
        issues.append(f"Duplicate patient_id: {df_raw.duplicated(subset=['patient_id']).sum()} records.")
    if (df_raw["patients_refused"] > df_raw["patients_request"]).sum() > 0:
        issues.append(f"Operational inconsistency: patients_refused > patients_request in {(df_raw['patients_refused'] > df_raw['patients_request']).sum():,} records (raw).")
    
    # staff_present + staff_absent vs total
    mismatch = ((df_raw["staff_present"]+df_raw["staff_absent"]).round() != df_raw["total_staff"]).sum()
    if mismatch>0:
        issues.append(f"Staffing inconsistency: staff_present + staff_absent ≠ total_staff in {mismatch:,} records (raw). Also staff_present exceeds total_staff in {(df_raw['staff_present'] > df_raw['total_staff']).sum():,} records — likely synthetic/aggregated fields, not a literal roster.")
    if (df_raw["departure_date"] < df_raw["arrival_date"]).sum() >0:
        issues.append(f"Invalid dates: departure_date < arrival_date in {(df_raw['departure_date'] < df_raw['arrival_date']).sum()} records.")
    if (df_raw["length_of_stay"]<0).sum()>0:
        issues.append("Negative length_of_stay detected.")
    if not set(df_raw["patient_churn"].unique()).issubset({0,1}):
        issues.append(f"Invalid churn values: {sorted(df_raw['patient_churn'].unique().tolist())}")
    if df_raw.isna().sum().sum()>0:
        issues.append("Missing values detected — see table below.")
    else:
        issues.append("No missing values — dataset is complete.")
    
    # LOS vs date diff already validated as consistent
    for iss in issues:
        if "inconsistency" in iss or "exceeds" in iss or "Invalid" in iss or "Duplicate" in iss:
            render_insight(iss, kind="warn")
        else:
            render_insight(iss)
    st.caption("Records are not modified. Warnings are surfaced for analytical awareness.")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Dataset Preview","Data Types","Missing Values","Descriptive Statistics", "Model Columns"])
    with tab1:
        st.dataframe(df.head(200), use_container_width=True, hide_index=True, height=400)
        # filtered download
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download filtered dataset (CSV)", csv, file_name="hospital_filtered.csv", mime="text/csv")
        # also filtered parquet? offer CSV only for portability
        st.caption(f"Showing first 200 of {len(df):,} filtered rows. Download contains all filtered rows.")
    with tab2:
        dtypes = pd.DataFrame({"column": df_raw.columns, "dtype": df_raw.dtypes.astype(str).values})
        st.dataframe(dtypes, use_container_width=True, hide_index=True)
    with tab3:
        miss = df_raw.isna().sum().reset_index()
        miss.columns=["column","missing"]
        miss["missing_pct"] = miss["missing"]/len(df_raw)*100
        st.dataframe(miss.style.format({"missing_pct":"{:.2f}%"}), use_container_width=True, hide_index=True)
        fig = px.bar(miss, x="column", y="missing", color_discrete_sequence=["#0F2A44"])
        fig.update_layout(height=300, xaxis_tickangle=-30, xaxis_title="", yaxis_title="Missing count")
        st.plotly_chart(fig, use_container_width=True)
    with tab4:
        st.markdown("**Numeric summary (raw dataset)**")
        st.dataframe(df_raw.describe().T, use_container_width=True)
        st.markdown("**Categorical — service**")
        svc_counts = df_raw["service"].value_counts().reset_index()
        svc_counts.columns = ["service", "count"]
        st.dataframe(svc_counts, use_container_width=True, hide_index=True)
        st.markdown("**Correlation (numeric columns)**")
        num_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
        # drop constant-ish columns if needed
        corr = df_raw[num_cols].corr(numeric_only=True)
        fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale="RdBu", zmid=0, zmin=-1, zmax=1, text=np.round(corr.values,2), texttemplate="%{text}"))
        fig.update_layout(height=500, xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Correlation indicates association and does not establish causation.")
    with tab5:
        st.markdown("**Model output columns present on `df`**")
        model_cols = [c for c in df.columns
                    if c.endswith("_pred") or c.endswith("_prob")
                    or "_prob_class_" in c or c.endswith("_submodel")]

        if model_cols:
            st.dataframe(df[model_cols].head(50), use_container_width=True, hide_index=True)
        else:
            st.info("No model columns — check the Models panel in the sidebar for errors.")

        st.markdown("**Submodel per concept**")
        rows = []
        for c in MODEL_CONCEPTS:
            col = f"{c}_submodel"
            if col in df.columns and df[col].notna().any():
                rows.append({"Concept": c, "Submodel": df[col].dropna().iloc[0]})
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ──────────────────────────────────────────────
# Page: Models performance
# ──────────────────────────────────────────────
elif nav == "🧠 Model Performance":
    st.markdown("# Model Performance")
    st.markdown('<div class="section-desc">Trained submodels per concept with their evaluation metrics — sourced from manifests stored in MinIO.</div>', unsafe_allow_html=True)

    manifests = {}
    for concept in MODEL_CONCEPTS:
        try:
            manifests[concept] = load_manifest_cached(concept)
        except Exception as e:
            st.warning(f"Failed to load manifest for concept '{concept}': {e}")


    if not manifests:
        st.info("No manifests found. Train the model first")
        st.stop()

    # --- Summary table ---
    render_section_header("Summary", "Best submodel per concept by primary metric")
    rows = []
    for concept, m in manifests.items():
        task = m["task"]
        metric_key = "roc_auc" if task=="classification" else "r2"
        metrics = m["metrics"]
        best = max(metrics.items(), key=lambda kv: kv[1].get(metric_key, -1) or -1)
        best_name, best_metrics = best

        # Append row
        rows.append({
            "Concept": concept,
            "Task": task,
            "Submodels": len(metrics),
            "Best Submodel": best_name,
            "Primary Metric": f"{metric_key} = {best_metrics.get(metric_key):.4f}",
            "Feature Count": len(m["features"])
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # --- Per-concept detail ---
    render_section_header("Per-Concept Detail", "All submodels, all metrics")
    for concept, m in manifests.items():
        with st.expander(f"{concept} - {len(m['submodels'])} submodels", expanded=False):
            df_metrics = pd.DataFrame(m["metrics"]).T.reset_index()
            df_metrics.rename(columns={"index": "submodel"}, inplace=True)

            # Numeric formatting
            fmt = {}
            for c in df_metrics.columns:
                if c != "submodel":
                    fmt[c] = "{:.4f}"
            st.dataframe(df_metrics.style.format(fmt), use_container_width=True, hide_index=True)

            c1, c2 = st.columns([3, 2])
            with c1:
                fig = px.bar(
                    df_metrics.melt(id_vars="submodel", var_name="metric", value_name="score"),
                    x="submodel", y="score", color="metric", barmode="group", title=f"{concept} - metrics by submodel")
                fig.update_layout(height=320, xaxis_title="", yaxis_title="Score")
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                st.markdown("**Features used**")
                st.code("\n".join(m["features"]), language="text")

    # --- Prediction coverage ---
    render_section_header("Prediction Coverage", "How many filtered rows got a model prediction")
    cov_rows = []
    for concept in MODEL_CONCEPTS:
        col = f"{concept}_pred"
        if col in df.columns:
            cov = df[col].notna().mean()
            cov_rows.append({"Concept": concept, "Coverage": f"{cov:.1%}",
                             "Non-null rows": int(df[col].notna().sum())})

        else:
            cov_rows.append({"Concept": concept, "Coverage": "—", "Non-null rows": 0})
    st.dataframe(pd.DataFrame(cov_rows), use_container_width=True, hide_index=True)


# Footer
st.markdown("---")
st.caption("Hospital Care Analytics — Operational dashboard. Data source: fix_final_hospital_database.parquet. All metrics are computed live from the filtered dataset.")
