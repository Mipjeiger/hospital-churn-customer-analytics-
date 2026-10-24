import numbers
import pandas as pd
import mlflow
from mlflow.metrics.genai import answer_correctness, relevance, faithfulness
from app.api.config import settings
from app.llm.mlflow_llm import traced_llm
from app.ml.pg_store import load_data

"""LLM Evaluation Suite"""

def add_integrity_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Add new features which are define the frauds dependencies and integrity of the data
    df["request_mismatch"] = (df["patients_admitted"] + df["patients_refused"] != df["patients_request"])
    df["admission_over_request"] = (df["patients_admitted"] > df["patients_request"])
    df["staff_over_capacity"] = (df["staff_present"] > df["total_staff"])
    df["invalid_length_of_stay"] = (df["length_of_stay"] < 0)
    df["integrity_flagged"] = (
        df["request_mismatch"]
        | df["admission_over_request"]
        | df["staff_over_capacity"]
        | df["invalid_length_of_stay"]
    ).astype(int)

    return df

def build_eval_dataset() -> pd.DataFrame:
    """Build a Dataframe for LLM evaluation with questions, expected answers, and context."""
    df = add_integrity_features(load_data())

    service_churn = (
        df.groupby("service")["patient_churn"]
        .mean()
        .sort_values(ascending=False)
    )

    surgery_churn = float(
        df.loc[
            df["service"].eq("surgery"),
            "patient_churn",
        ].mean()
    )

    integrity_by_service = (
        df.groupby("service")["integrity_flagged"]
        .mean()
        .sort_values(ascending=False)
    )

    highest_integrity_service = str(integrity_by_service.index[0])

    return pd.DataFrame([
        {
            "question": "What is the churn rate for surgery?",
            "expected": (
                f"The churn rate for surgery is "
                f"{surgery_churn:.2%}."
            ),
            "context": (
                f"The calculated surgery churn rate is "
                f"{surgery_churn:.2%}."
            ),
        },
        {
            "question": "Which service has the highest churn?",
            "expected": (
                f"{service_churn.index[0]} has the highest churn rate."
            ),
            "context": (
                f"Service churn rates are "
                f"{service_churn.round(4).to_dict()}."
            ),
        },
        {
            "question": "How many patients are in the database?",
            "expected": (
                f"There are {len(df)} patients in the database."
            ),
            "context": (
                f"The database contains {len(df)} patient rows."
            ),
        },
        {
            "question": "How many operational integrity concerns were detected?",
            "expected": (
                f"{int(df['integrity_flagged'].sum())} "
                "operational integrity concerns were detected."
            ),
            "context": (
                f"The rule-based integrity check detected "
                f"{int(df['integrity_flagged'].sum())} flagged rows."
            ),
        },
        {
            "question": (
                "Which service has the highest operational "
                "integrity concern?"
            ),
            "expected": (
                f"{highest_integrity_service} has the highest "
                "operational integrity concern rate."
            ),
            "context": (
                "An integrity concern is flagged when admissions plus "
                "refusals do not equal requests, admissions exceed "
                "requests, staff present exceeds total staff, or length "
                f"of stay is negative. Rates by service: "
                f"{integrity_by_service.round(4).to_dict()}."
            ),
        },
    ])

# ------- LLM Evaluation Datasets -------
EVAL_DATASET = build_eval_dataset()

def predict_fn(inputs, model_name="router" ,**kwargs):
    """MLflow passes the row (dict or Series). Extract the question string."""
    if isinstance(inputs, pd.DataFrame):
        df = inputs
    elif isinstance(inputs, dict):
        df = pd.DataFrame([inputs])
    else:
        df = pd.DataFrame([{"question": str(inputs)}])

    # Build list and store into predictions storage
    predictions = []

    for _, row in df.iterrows():
        question = str(row["question"])
        context = str(row.get("context", ""))

        prompt = f"""
        Answer the question using only the provided context.
        Do not invent facts or numbers.

        Context:
        {context}

        Question:
        {question}
        """

        response = traced_llm(
            [
                {"role": "system", "content": prompt,},
                {"role": "user","content": question,}
            ],
            temperature=0.0,
            model_name=model_name,
        )
        predictions.append(response["text"])

    return predictions

def predict_fn_for_eval(inputs, **kwargs):
    return predict_fn(
        inputs,
        model_name=kwargs.pop("model_name", "router"),
        **kwargs
    )

def run_eval(model_name: str = "router"):
    """Run evaluation with optional provider checks and disabled autologging."""
    mlflow.autolog(disable=True) # Disable autologging to avoid logging the evaluation as a model run

    if model_name == "huggingface" and not settings.HUGGINGFACE_API_KEY:
        raise RuntimeError("HUGGINGFACE_API_KEY not set – cannot run HuggingFace evaluation.")

    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(settings.MLFLOW_EXPERIMENT)

    def model_fn(inputs, **kwargs):
        return predict_fn(
            inputs,
            model_name=model_name,
            **kwargs
        )

    with mlflow.start_run(run_name=f"llm_eval_{model_name}"):
        results = mlflow.evaluate(
            model=model_fn,
            data=EVAL_DATASET,
            targets="expected",
            model_type="question-answering",
            extra_metrics=[
            answer_correctness(),
            relevance(),
            faithfulness(),
        ],
            evaluator_config={
                "col_mapping": {
                    "inputs": "question",
                    "context": "context",
                    "targets": "expected",
                }
            }
        )

        # Mlflow log traces
        mlflow.log_metrics({
            key: float(value)
            for key, value in results.metrics.items() if isinstance(value, numbers.Real) and pd.notna(value)
        })
        mlflow.log_param("model", model_name)
        mlflow.log_param("dataset_size", len(EVAL_DATASET))
        print(results.metrics)
    
    return results