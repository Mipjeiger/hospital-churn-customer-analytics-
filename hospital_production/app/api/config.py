import os
from pathlib import Path
from dotenv import load_dotenv

DIR_PATH = Path(__file__).parents[2]
ENV_PATH = DIR_PATH / ".env"
load_dotenv(ENV_PATH, override=True)

class Settings():

    # LLM API Keys
    HUGGINGFACE_API_KEY: str = os.getenv("HUGGINGFACE_API_KEY")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY")
    NINEROUTER_BASE_URL: str = os.getenv("NINEROUTER_BASE_URL")
    NINEROUTER_API_KEY: str = os.getenv("NINEROUTER_API_KEY")
    NINEROUTER_MODEL_NAME: str = "sragent"

    # ML Path dependencies
    BASE_PATH: Path = Path(__file__).parents[2] # hospital_production
    DATABASE_PATH: Path = BASE_PATH / "database"
    ML_MODEL_PATH: Path = DATABASE_PATH / "model_artifacts"
    LLM_PATH: Path = DATABASE_PATH / "llm_artifacts"
    DATA_PATH: Path = DATABASE_PATH / "fix_final_hospital_database.parquet"

    # BigQuery
    GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
    BQ_DATASET     = os.getenv("BQ_DATASET")
    BQ_TABLE       = os.getenv("BQ_TABLE")
    BQ_LOCATION    = os.getenv("BQ_LOCATION")

    # JSON Parse
    ROLE_CONFIG: Path = DATABASE_PATH / "llm_artifacts" / "role_config.json"

    # ── Storage ──
    MINIO_ENDPOINT   = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    MINIO_ROOT_USER = os.getenv("MINIO_ROOT_USER")
    MINIO_ROOT_PASSWORD = os.getenv("MINIO_ROOT_PASSWORD")
    MINIO_BUCKET     = os.getenv("MINIO_BUCKET")

    # ── MLflow ──
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    MLFLOW_EXPERIMENT : str = os.getenv("MLFLOW_EXPERIMENT", "hospital_model")

    # ── Feast ──
    FEAST_REPO_PATH = Path("feature_store")

    # ── Langfuse ──
    LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
    LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
    LANGFUSE_HOST       = os.getenv("LANGFUSE_HOST", "http://localhost:3000")

settings = Settings()