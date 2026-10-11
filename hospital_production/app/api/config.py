import os
from pathlib import Path

class Settings():
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    MLFLOW_MODEL_NAME: str = os.getenv("MLFLOW_MODEL_NAME", "hospital_model")

    # LLM API Keys
    HUGGINGFACE_API_KEY: str = os.getenv("HUGGINGFACE_API_KEY")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY")
    NINEROUTER_BASE_URL: str = os.getenv("NINEROUTER_BASE_URL")
    NINEROUTER_API_KEY: str = os.getenv("NINEROUTER_API_KEY")
    NINEROUTER_MODEL_NAME: str = "sragent"

    # ML Path dependencies
    BASE_PATH: Path = Path(__file__).parents[3] # Hospital_healthcare
    DATABASE_PATH: Path = BASE_PATH / "database"
    ML_MODEL_PATH: Path = DATABASE_PATH / "model_artifacts"
    LLM_PATH: Path = DATABASE_PATH / "llm_artifacts"
    DATA_PATH: Path = DATABASE_PATH / "final_hospital_database.parquet"

    # JSON Parse
    ROLE_CONFIG: Path = DATABASE_PATH / "llm_artifacts" / "role_config.json"

settings = Settings()