from typing import Any
from fastapi import APIRouter, HTTPException, Query
from app.api.config import settings
from app.ml.pg_store import query

router = APIRouter(prefix="/database", tags=["database"])


@router.get("")
def get_database_rows(
    limit: int = Query(default=100, ge=1, le=10000),
    offset: int = Query(default=0, ge=0)
) -> dict[str, Any]:
    """
    Get rows from the database with pagination.

    Args:
        limit (int): The maximum number of rows to return (default: 100).
        offset (int): The number of rows to skip before starting to return rows (default: 0).

    Returns:
        dict: A dictionary containing the retrieved rows.
    """
    table = settings.POSTGRES_TABLE
    schema = settings.POSTGRES_SCHEMA
    try:
        rows = query(
            f"""
            SELECT *
            FROM {schema}.{table}
            ORDER BY patient_id
            LIMIT :limit OFFSET :offset
            """,
            params={
                "limit": limit,
                "offset": offset
            }
        )

        total_result = query(
            f"""
            SELECT COUNT(*) AS total
            FROM {schema}.{table}
            """
        )