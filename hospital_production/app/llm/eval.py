import mlflow
from mlflow.metrics.genai import answer_correctness, relevance, faithfulness
from app.api.config import settings
from app.llm.prompts import render_prompt
from app.llm.mlflow_llm import traced_llm

"""LLM Evaluation Suite"""

# ------- LLM Evaluation Datasets -------
EVAL_DATASET = [
    {
        "inputs":  {"question": "What is the churn rate for surgery?"},
        "outputs": {"expected": "Churn rate for surgery is approximately X%"},
        "context": "surgery churn is computed as mean of patient_churn in service='surgery'",
    },
    {
        "inputs":  {"question": "Which service has the highest churn?"},
        "outputs": {"expected": "The service with max churn rate"},
        "context": "compare groupby('service').patient_churn.mean()",
    },
    {
        "inputs":  {"question": "How many patients are in the database?"},
        "outputs": {"expected": "Total row count"},
        "context": "len(df)",
    },
    {
        "inputs":  {"question": "Which the mangement has fraud indications?"},
        "outputs": {"expected": "The management with fraud indications is X"},
        "context": "compare df.groupby('service')['iso_flag'].mean().sort_vales(ascending=False)",
    },
]

def predict_fn(question: str, **kwargs) -> str:
    """LLM call for evaluation"""
    prompt = render_prompt("hospital_agent_system", role="analyst")
    r = traced_llm([
        {"role": "system", "content": prompt},
        {"role": "user", "content": question},
    ], 
    temperature=0.3)
    return r["text"]

def run_eval(model_name: str = None):
    """Run LLM evaluation on the EVAL DATASET and log metrics to MLflow"""
    mlflow.set_experiment(settings.MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=f"llm_eval_{model_name}"):
        results = mlflow.evaluate(
            model=predict_fn,
            data=EVAL_DATASET,
            predictions="outputs",
            extra_metrics=[
                answer_correctness(),
                relevance(),
                faithfulness()
            ]
        )

        # Mlflow log traces
        mlflow.log_metrics({
            k: v for k, v in results.metrics.items() if isinstance(v, (int, float))
        })
        mlflow.log_param("model", model_name)
        mlflow.log_param("dataset_size", len(EVAL_DATASET))
        print(results.metrics)
    
    return results