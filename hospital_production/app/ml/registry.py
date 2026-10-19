import os
import mlflow
import mlflow.sklearn
import logging
from app.ml.minio_store import save_artifact, save_json, load_artifact
from app.api.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define trusted types for unpickling to avoid security issues with joblib
TRUSTED_SKOPS_TYPES = [
    "numpy.dtype",
    "sklearn.compose._column_transformer._RemainderColsList",
    "sklearn.tree._tree.Tree",
    "xgboost.core.Booster",
    "xgboost.sklearn.XGBClassifier",
    "xgboost.sklearn.XGBRegressor",
]

"""MinIO + MLFlow model registry"""

def log_and_register(concept: str, results: dict, task: str, features: list):
    """
    concept: 'integrity_audit' | 'service_performance' | 'staff' | 'risk' | 'churn'
    - Prefect trains the pipelines (pipe.fit already happened).
    - This function ONLY records: metrics, params, and the model itself.

    Layout in MLflow:
        Parent run: "<concept>_multimodel"
            ├─ params: minio_key, n_submodels, submodels
            ├─ metrics: <submodel>_<metric> for every submodel
            └─ artifacts: models/<submodel>/  (each submodel logged via mlflow.sklearn)
    """
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio:9000"
    os.environ["AWS_ACCESS_KEY_ID"] = settings.MINIO_ROOT_USER
    os.environ["AWS_SECRET_ACCESS_KEY"] = settings.MINIO_ROOT_PASSWORD

    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(settings.MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=f"{concept}_multimodel"):
        # 1. Log aggregate metrics from every submodel
        for name, r in results.items():
            for k, v in r["metrics"].items():
                if v is not None:
                    mlflow.log_metric(f"{name}_{k}", v)

        # 2. Log each fitted pipeline as an MLflow model
        for name, r in results.items():
            pipe = r["pipeline"]
            try:
                mlflow.sklearn.log_model(
                    sk_model=pipe,
                    name=name,
                    registered_model_name=None,
                    input_example=None,
                    metadata={"concept": concept, "algorithm": name, "task": task},
                    skops_trusted_types=TRUSTED_SKOPS_TYPES
                )
            except Exception as e:
                logger.error(f"⚠️ Could not log model '{name}' to MLFlow: {e}")
                
        # 3. Save bundle to MinIO
        minio_key = f"models/{concept}_bundle"
        save_artifact(minio_key, {"models": results, "features": features, "task": task})

        # 4. Manifest JSON → both MinIO and MLflow 
        manifest = {
            "concept": concept,
            "task": task,
            "features": features,
            "minio_key": minio_key,
            "submodels": list(results.keys()),
            "metrics": {n: r["metrics"] for n, r in results.items()}
        }
        save_json(f"manifest/{concept}", manifest)

        # 5. Log params + manifest artifact to MLflow ──
        mlflow.log_param("minio_key", minio_key)
        mlflow.log_param("n_submodels", len(results))
        mlflow.log_param("submodels", ",".join(results.keys()))
        mlflow.log_dict(manifest, f"manifest_{concept}.json")

    return manifest

def load_concept(concept: str):
    """FastAPI loads the bundle from MinIO (fast, single pickle)."""
    try:
        return load_artifact(f"{concept}_bundle")
    except Exception:

        # Return the dummy if the real bundle is missing
        from app.ml.minio_store import _client
        from app.api.config import settings
        dummy_path = "dummy_bundle.json"

        # Upload on first-use
        if not _client().stat_object(settings.MINIO_BUCKET, dummy_path):
            # Create a tiny JSON object locally and put it properly
            _client.put_object(
                settings.MINIO_BUCKET,
                dummy_path,
                data=b'{"pipeline": null, "best_submodel":"placeholder", "features":[], "task":"classification"}',
                length=102
            )

        return load_artifact(dummy_path)