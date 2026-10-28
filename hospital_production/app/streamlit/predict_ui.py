import streamlit as st
import pandas as pd

"""Reusable Streamlit form for manual single-row prediction."""

def _render_result(result: dict):
    concept = result["concept"]
    submodel = result["submodel"]
    task = result["task"]
    st.markdown(f"**Submodel**: `{submodel}` ({task})")
    st.markdown(f"**Prediction**: `{result['prediction']}`")

    if result["probability"] is not None:
        p = result["probability"]
        st.markdown(f"**Positive-class probability**: {p:.1%}")
        st.progress(min(max(p, 0.0), 1.0))
    elif result["probabilities"] is not None:
        labels = result["labels"] or [f"class_{i}" for i in range(len(result["probabilities"]))]
        df = pd.DataFrame({
            "class": labels,
            "probability": [f"{v:.1%}" for v in result["probabilities"]],
        }).sort_values("probability", ascending=False)
        st.dataframe(df, use_container_width=True, hide_index=True)

def prediction_form(
    concept: str,
    submodel: str,
    defaults: dict,
    form_key: str,
    services: list[str] | None = None,
) -> dict | None:
    """Render a form the concept's features and show the prediction result."""
    unique_form_key = f"{form_key}_{concept}"

    with st.form(key=unique_form_key):
        cols = st.columns(3)
        values = {}

        for index, feature in enumerate(defaults):
            with cols[index % 3]:
                default_value = defaults[feature]

                if feature == "service" and services:
                    service_index = services.index(default_value) if default_value in services else 0
                    values[feature] = st.selectbox(
                        feature,
                        services,
                        index=service_index,
                        key=f"{unique_form_key}_{feature}"
                    )

                elif isinstance(default_value, bool):
                    values[feature] = st.checkbox(
                        feature,
                        value=default_value,
                        key=f"{unique_form_key}_{feature}"
                    )

                elif isinstance(default_value, int):
                    values[feature] = st.number_input(
                        feature,
                        value=default_value,
                        step=1,
                        key=f"{unique_form_key}_{feature}"
                    )

                else:
                    values[feature] = st.number_input(
                        feature,
                        value=float(default_value),
                        step=0.1,
                        key=f"{unique_form_key}_{feature}"
                    )

        submitted = st.form_submit_button("Predict", key=f"{unique_form_key}_submit")

    if not submitted:
        return None

    # Call the prediction API
    return {
        "concept": concept,
        "submodel": submodel,
        "submitted": True,
        "input": values
    }