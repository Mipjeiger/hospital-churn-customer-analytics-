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
        # Read data query
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

        # Get total count of rows in the table
        total_result = query(
            f"""
            SELECT COUNT(*) AS total
            FROM {schema}.{table}
            """
        )

        columns = list(rows.columns)
        records = rows.astype(object).to_dict(orient="records")

        return {
            "columns": columns,
            "rows": records,
            "limit": limit, 
            "offset": offset,
            "total": int(total_result.iloc[0]["total"])
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving rows from the database: {e}")