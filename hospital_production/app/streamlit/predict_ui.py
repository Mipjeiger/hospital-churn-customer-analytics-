import streamlit as st
import pandas as pd
from app.streamlit.predict import predict_one

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
) -> None:
    """Render a form the concept's features and show the prediction result."""
    from app.ml.registry import load_concept
    bundle = load_concept(concept)
    features = bundle["features"]

    # Make the form key globally unique
    unique_form_key = f"{form_key}__{concept}"

    with st.form(key=unique_form_key):
        cols = st.columns(3)
        values = {}
        for i, f in enumerate(features):
            with cols[i % 3]:
                d = defaults.get(f, 0)

                if f == "service" and services:
                    idx = services.index(d) if d in services else 0
                    values[f] = st.selectbox(f, services, index=idx, key=f"{unique_form_key}_{f}")
                elif isinstance(d, bool):
                    values[f] = st.checkbox(f, value=d, key=f"{unique_form_key}_{f}")
                elif isinstance(d, int):
                    values[f] = st.number_input(f, value=int(d), step=1, key=f"{unique_form_key}_{f}")
                else:
                    values[f] = st.number_input(f, value=float(d), step=0.1, key=f"{unique_form_key}_{f}")

        # ✅ Pass a unique key to the submit button
        submitted = st.form_submit_button("🔮 Predict", key=f"{unique_form_key}_submit")

    if submitted:
        try:
            result = predict_one(concept, submodel, values)
            _render_result(result)
        except Exception as e:
            st.error(f"Prediction failed: {e}")