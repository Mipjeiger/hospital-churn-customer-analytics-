import logging
from functools import lru_cache
from sqlalchemy import create_engine, text
import pandas as pd
from typing import Any
from app.api.config import settings

logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def _engine():
    """Create a SQLAlchemy engine for PostgreSQL connection"""
    url = f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
    return create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=10)

def load_data() -> pd.DataFrame:
    """Load data from PostgreSQL into a Pandas DataFrame."""
    return pd.read_sql(f"SELECT * FROM {settings.POSTGRES_TABLE}", _engine())

def query(sql: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
    """Execute a SQL query and return the result as a Pandas DataFrame."""
    with _engine().connect() as conn:
        return pd.read_sql_query(
            text(sql),
            conn,
            params=params
        )

def fetch_patient(patient_id: str) -> pd.Series | None:
    """Fetch a single patient row by ID"""
    df = pd.read_sql(
        text(f"SELECT * FROM {settings.POSTGRES_TABLE} WHERE patient_id = :patient_id"),
        _engine(),
        params={"patient_id": patient_id}
    )
    return df.iloc[0] if not df.empty else None

def fetch_features_for_patients(patient_ids: list[str]) -> pd.DataFrame:
    """Bulk fetch for batched inference"""
    return pd.read_sql(
        text(f"SELECT * FROM {settings.POSTGRES_TABLE} WHERE patient_id = ANY(:patient_ids)"),
        _engine(),
        params={"patient_ids": patient_ids}
    )