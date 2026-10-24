import pandas as pd
import streamlit as st
from app.ml.registry import load_concept

@st.cache_resource
def _bundle(concept: str):
    return load_concept(concept)

def predict_one(concept: str, submodel: str, row: dict) -> dict:
    """
    Predict a single row with the chosen submodel of a concept.
    
    Returns:
        {
          "concept": str,
          "submodel": str,
          "task": "classification" | "regression",
          "prediction": value,           # class label or numeric
          "probability": float | None,   # binary prob of positive class
          "probabilities": list[float] | None,  # multi-class probs
          "labels": list[str] | None,    # class labels for multi-class
        }
    """
    bundle = _bundle(concept)
    entry = bundle["models"][submodel]
    features = bundle["features"]

    missing = [f for f in features if f not in row]
    if missing:
        raise ValueError(f"Missing features for {concept}/{submodel}: {missing}")

    # Prepare input
    X = pd.DataFrame([{f: row[f] for f in features}])
    pipe = entry["pipeline"]
    task = bundle["task"]

    pred = pipe.predict(X)[0]

    # decode via stored Label encoder
    le = entry.get("label_encoder")
    if le is not None:
        try:
            pred = le.inverse_transform([pred])[0]
        except Exception:
            pass

    result = {
        "concept": concept,
        "submodel": submodel,
        "task": task,
        "prediction": pred,
        "probability": None,
        "probabilities": None,
        "labels": None,
    }

    # Get attr the prediction probabilities if available
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X)[0]

        if len(proba) == 2:
            result["probability"] = float(proba[1])
        else:
            result["probabilities"] = [float(v) for v in proba]
            if le is not None:
                try:
                    result["labels"] = [str(c) for c in le.classes_]
                except Exception:
                    result["labels"] = None

    return result