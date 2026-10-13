import mlflow
import mlflow.sklearn
from app.ml.minio_store import save_artifact, save_json, load_artifact
from app.api.config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT

"""MinIO + MLFlow model registry"""

def log_and_register(concept: str, results: dict, task: str, features: list):
    """
    concept: 'integrity_audit' | 'service_performance' | 'staff' | 'risk' | 'churn'
    Logs every submodel to MLflow, saves full multimodel bundle to MinIO
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=f"{concept}_multimodel"):
        for name, r in results.items():
            for k, v in r["metrics"].items():
                if v is not None:
                    mlflow.log_metric(f"{name}_{k}", v)

        # Save bundle to MinIO
        minio_key = f"models/{concept}_bundle"
        save_artifact(minio_key, {"models": results, "features": features, "task": task})

        # Manifest for MLflow
        manifest = {
            "concept": concept,
            "task": task,
            "features": features,
            "minio_key": minio_key,
            "submodels": list(results.keys()),
            "metrics": {n: r["metrics"] for n, r in results.items()}
        }
        save_json(f"manifest/{concept}", manifest)

        # Log MLFlow parameters
        mlflow.log_param("minio_key", minio_key)
        mlflow.log_param("n_submodels", len(results))
        mlflow.log_param("submodels", ",".join(results.keys()))

    return manifest

def load_concept(concept: str):
    return load_artifact(f"models/{concept}_bundle")