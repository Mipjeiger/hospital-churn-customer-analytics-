import pandas as pd
import plotly.express as px
import streamlit as st

"""Pre-built visualizations that consume predictions on df."""

def churn_distribution(df: pd.DataFrame, color_map: dict) -> None:
    """Display the distribution of API-predicted churn probabilities."""
    if "patient_churn_prob" not in df.columns:
        st.info("Run predict to retrieve churn predictions from FastAPI.")
        return

    probabilities = df["patient_churn_prob"].dropna()

    if probabilities.empty:
        st.info("Run predict to retrieve churn predictions from FastAPI.")
        return

    # Prediction labels
    predicted_labels = (
        probabilities.ge(0.5).map({
            False: "Retained",
            True: "Churned"
        }).value_counts().reindex(["Retained", "Churned"], fill_value=0)
        .rename_axis("status").reset_index(name="patients")
    )

    # Pie chart
    fig = px.pie(
        predicted_labels,
        names="status",
        values="patients",
        hole=0.62,
        color="status",
        color_discrete_map=color_map,
    )

    fig.update_traces(
        textinfo="percent+value",
        textposition="inside",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )    

def churn_confusion(df: pd.DataFrame) -> None:
    """Display observed churn vs the FastAPI predictions."""
    required_columns = {
        "patient_churn",
        "patient_churn_prob"
    }

    if not required_columns.issubset(df.columns):
        st.info("Observed churn and FastAPI predictions are required to display the confusion matrix.")
        return

    work = df[list(required_columns)].dropna()

    if work.empty:
        st.info("No observed churn and FastAPI predictions available for the filtered rows.")
        return

    # Get actual and predicted values
    actual = work["patient_churn"].astype(int)
    predicted = work["patient_churn_prob"].ge(0.5)

    # Create confusion matrix
    matrix = pd.crosstab(
        actual.map({
            0: "Actual Retained",
            1: "Actual Churned"
        }),
        predicted.map({
            False: "Pred Retained",
            True: "Pred Churned"
        }),
    )

    matrix = matrix.reindex(
        index=["Actual Retained", "Actual Churned"],
        columns=["Pred Retained", "Pred Churned"],
        fill_value=0,
    )

    fig = px.imshow(
        matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
    )

    fig.update_layout(
        height=300,
        xaxis_title="",
        yaxis_title="",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )    

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

def prediction_coverage_table(df: pd.DataFrame, concepts: list[str]):
    """Table: how many filtered rows got predictions per concept."""
    rows = []
    for concept in concepts:
        col = f"{concept}_pred"
        if col in df.columns:
            rows.append({
                "Concept": concept,
                "Coverage": f"{df[col].notna().mean():.1%}",
                "Non-null rows": int(df[col].notna().sum()),
            })
        else:
            rows.append({"Concept": concept, "Coverage": "—", "Non-null rows": 0})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

def service_prediction_bar(
    df: pd.DataFrame,
    concept: str,
    color_map: dict,
) -> None:
    prediction_column = f"{concept}_pred"

    if prediction_column not in df.columns:
        st.info(
            f"Run Predict to retrieve {concept} results from FastAPI."
        )
        return

    if "service" not in df.columns:
        st.error("The 'service' column is required to display the service prediction bar chart.")
        return

    work = df[["service", prediction_column]].dropna()

    if work.empty:
        st.info("FastAPI returned no prediction rows.")
        return

    service_predictions = (
        work
        .groupby("service", as_index=False)[prediction_column]
        .mean()
        .sort_values(prediction_column, ascending=False)
    )

    fig = px.bar(
        service_predictions,
        x="service",
        y=prediction_column,
        color="service",
        color_discrete_map=color_map,
        text_auto=".2f",
    )

    fig.update_traces(textposition="outside")

    fig.update_layout(
        height=300,
        showlegend=False,
        xaxis_title="",
        yaxis_title="Predicted",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )