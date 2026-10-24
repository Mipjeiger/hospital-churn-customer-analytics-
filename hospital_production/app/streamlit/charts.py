import pandas as pd
import plotly.express as px
import streamlit as st

"""Pre-built visualizations that consume predictions on df."""

def churn_distribution(df: pd.DataFrame, color_map: dict):
    """Bar chart: predicted churn probability histogram."""
    if "patient_churn_prob" not in df.columns:
        st.info("No churn predictions available.")
        return
    fig = px.histogram(df, x="patient_churn_prob", nbins=40,
                       color_discrete_sequence=["#0E7C7B"])
    fig.update_layout(height=300, xaxis_title="Predicted churn probability",
                      yaxis_title="Patients")
    st.plotly_chart(fig, use_container_width=True)

def churn_confusion(df: pd.DataFrame):
    """Confusion matrix: observed vs predicted (threshold=0.5)."""
    if "patient_churn_prob" not in df.columns:
        return
    cm = pd.crosstab(
        df["patient_churn"].map({0: "Actual Retained", 1: "Actual Churned"}),
        (df["patient_churn_prob"] > 0.5).map({False: "Pred Retained", True: "Pred Churned"}),
    )
    fig = px.imshow(cm.values, text_auto=True, aspect="auto",
                    x=cm.columns.tolist(), y=cm.index.tolist(),
                    color_continuous_scale="Blues")
    fig.update_layout(height=300, xaxis_title="", yaxis_title="",
                      title=dict(text="Confusion matrix (t=0.5)", font=dict(size=12)))
    st.plotly_chart(fig, use_container_width=True)

def model_prediction_vs_observed(df: pd.DataFrame, concept: str, time_col: str = "arrival_date"):
    """Line chart: observed vs predicted mean over time."""
    obs_col = concept.split("_")[-1] if concept != "risk_score" else "risk_score"
    pred_col = f"{concept}_pred"
    if pred_col not in df.columns or obs_col not in df.columns:
        return
    monthly = df.groupby(df[time_col].dt.to_period("M")).agg(
        observed=(obs_col, "mean"),
        predicted=(pred_col, "mean"),
    ).reset_index()
    monthly[time_col] = monthly[time_col].astype(str)
    fig = px.line(monthly, x=time_col, y=["observed", "predicted"], markers=True)
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)


def service_prediction_bar(df: pd.DataFrame, concept: str, color_map: dict):
    """Bar chart: mean predicted value per service."""
    col = f"{concept}_pred"
    if col not in df.columns:
        return
    svc = df.groupby("service")[col].mean().reset_index().sort_values(col, ascending=False)
    fig = px.bar(svc, x="service", y=col, color="service",
                 color_discrete_map=color_map, text_auto=".2f")
    fig.update_traces(textposition="outside")
    fig.update_layout(height=300, showlegend=False, xaxis_title="", yaxis_title="Predicted")
    st.plotly_chart(fig, use_container_width=True)