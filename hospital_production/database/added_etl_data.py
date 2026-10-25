from pathlib import Path
import re

import pandas as pd
from sqlalchemy import create_engine, Text, Float, text
from sqlalchemy.engine import Engine

from app.api.config import settings


SOURCE_FILE = (
    Path(__file__).resolve().parent / "Fix_final_2_database.parquet"
)

SCHEMA_NAME = settings.POSTGRES_SCHEMA
TABLE_NAME = settings.POSTGRES_TABLE
POSTGRES_HOST = "localhost"

REQUIRED_SOURCE_COLUMNS = {
    "patient_id",
    "capacity_pressure",
    "admission_rate_row",
    "refusal_rate_row",
    "staff_attendance_rate",
}

def quote_identifier(identifier: str) -> str:
    """Safely quote a PostgreSQL identifier."""
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", identifier):
        raise ValueError(f"Invalid PostgreSQL identifier: {identifier}")

    return f'"{identifier}"'

def get_engine() -> Engine:
    database_url = (
        f"postgresql+psycopg2://"
        f"{settings.POSTGRES_USER}:"
        f"{settings.POSTGRES_PASSWORD}@"
        f"{POSTGRES_HOST}:"
        f"{settings.POSTGRES_PORT}/"
        f"{settings.POSTGRES_DB}"
    )

    return create_engine(
        database_url,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
    )

def load_source_data() -> pd.DataFrame:
    if not SOURCE_FILE.is_file():
        raise FileNotFoundError(
            f"Parquet file was not found: {SOURCE_FILE}"
        )

    source_df = pd.read_parquet(SOURCE_FILE)

    missing_columns = REQUIRED_SOURCE_COLUMNS - set(source_df.columns)

    if missing_columns:
        raise ValueError(
            "The Parquet file is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    source_df = source_df[
        [
            "patient_id",
            "capacity_pressure",
            "admission_rate_row",
            "refusal_rate_row",
            "staff_attendance_rate",
        ]
    ].copy()

    source_df["patient_id"] = (
        source_df["patient_id"]
        .astype("string")
        .str.strip()
    )

    source_df = source_df.dropna(subset=["patient_id"])
    source_df = source_df.drop_duplicates(
        subset=["patient_id"],
        keep="last",
    )

    numeric_columns = [
        "capacity_pressure",
        "admission_rate_row",
        "refusal_rate_row",
        "staff_attendance_rate",
    ]

    for column in numeric_columns:
        source_df[column] = pd.to_numeric(
            source_df[column],
            errors="coerce",
        )

    return source_df

def add_columns_if_missing(connection) -> None:
    table_name = (
        f"{quote_identifier(SCHEMA_NAME)}."
        f"{quote_identifier(TABLE_NAME)}"
    )

    connection.execute(
        text(
            f"""
            ALTER TABLE {table_name}
                ADD COLUMN IF NOT EXISTS capacity_pressure DOUBLE PRECISION,
                ADD COLUMN IF NOT EXISTS admission_rate_row DOUBLE PRECISION,
                ADD COLUMN IF NOT EXISTS refusal_rate_row DOUBLE PRECISION,
                ADD COLUMN IF NOT EXISTS staff_attendance_rate DOUBLE PRECISION
            """
        )
    )

def update_existing_rows(
    connection,
    source_df: pd.DataFrame,
) -> int:
    table_name = (
        f"{quote_identifier(SCHEMA_NAME)}."
        f"{quote_identifier(TABLE_NAME)}"
    )

    staging_table = "tmp_hospital_derived_features"

    source_df = source_df.copy()

    source_df["patient_id"] = (
        source_df["patient_id"]
        .astype("string")
        .str.strip()
        .astype(object)
    )

    source_df.to_sql(
        name=staging_table,
        con=connection,
        if_exists="replace",
        index=False,
        dtype={
            "patient_id": Text(),
            "capacity_pressure": Float(),
            "admission_rate_row": Float(),
            "refusal_rate_row": Float(),
            "staff_attendance_rate": Float(),
        },
    )

    result = connection.execute(
        text(
            f"""
            UPDATE {table_name} AS target
            SET
                capacity_pressure = source.capacity_pressure,
                admission_rate_row = source.admission_rate_row,
                refusal_rate_row = source.refusal_rate_row,
                staff_attendance_rate = source.staff_attendance_rate
            FROM {quote_identifier(staging_table)} AS source
            WHERE target.patient_id = source.patient_id
            """
        )
    )

    connection.execute(
        text(
            f"""
            DROP TABLE IF EXISTS {quote_identifier(staging_table)}
            """
        )
    )

    return result.rowcount

def main() -> None:
    source_df = load_source_data()
    engine = get_engine()

    with engine.begin() as connection:
        add_columns_if_missing(connection)

        updated_rows = update_existing_rows(
            connection,
            source_df,
        )

    print(
        f"Processed {len(source_df):,} source rows."
    )
    print(
        f"Updated {updated_rows:,} existing PostgreSQL rows."
    )


if __name__ == "__main__":
    main()