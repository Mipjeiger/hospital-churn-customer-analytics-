import os
import pandas as pd
from sqlalchemy import create_engine, text
from app.api.config import settings
from sqlalchemy.engine import Engine

"""ETL script to load data from Parquet file into PostgreSQL database"""

# Define the PostgreSQL connection URL
SCHEMA_NAME = settings.POSTGRES_SCHEMA
TABLE_NAME = settings.POSTGRES_TABLE

# Chunk size for batch inserts
CHUNK_SIZE = 1000

POSTGRES_HOST = "localhost"

# Build postgresql connection string URL
def engine() -> Engine:
    url = f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}@{POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
    return create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=10)

def load_parquet(path: str | None = None) -> pd.DataFrame:
    """Load data from settings.DATA_PATH or provided path into a Pandas DataFrame."""
    if path is None:
        path = settings.DATA_PATH
    df = pd.read_parquet(path)
    return df

def load_to_postgres(engine: Engine, df: pd.DataFrame, schema_name: str, table_name: str) -> None:
    """Load DataFrame into PostgreSQL table in chunks."""
    with engine.begin() as conn:
        conn.execute(
            text(f"CREATE SCHEMA IF NOT EXISTS {schema_name}")
        )

        for start in range(0, len(df), CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, len(df))
            chunk = df.iloc[start:end]

            chunk.to_sql(
                name=table_name,
                con=conn,
                schema=schema_name,
                if_exists="append",
                index=False,
                method="multi",
                chunksize=CHUNK_SIZE
            )

            print(f"Inserted rows {start} to {end} into {schema_name}.{table_name}")

def ensure_database() -> None:
    """Ensure the database is created and already inserted with data from the Parquet file."""
    engine_ = engine()
    df = load_parquet()
    load_to_postgres(engine_, df, SCHEMA_NAME, TABLE_NAME)
    print(f"Data loaded into {SCHEMA_NAME}.{TABLE_NAME} successfully.")

# Usage
if __name__ == "__main__":
    ensure_database()