import os
from typing import Any
import requests

API_BASE_URL = os.getenv("API_BASE_URL", "http://fastapi:8000")

class PredictionAPIError(RuntimeError):
    """Raised when the FastAPI prediction service fails."""

def predict_from_api(
    concept: str,
    patient_ids: list[str],
    submodel: str | None = None
) -> dict[str, Any]:
    """Call the FastAPI prediction service for a given concept and patient IDs."""
    if not patient_ids:
        raise PredictionAPIError("No patient IDs provided for prediction.")

    payload: dict[str, Any] = {
        "patient_ids": patient_ids
    }

    if submodel:
        payload["submodel"] = submodel

    try:
        response = requests.post(
            f"{API_BASE_URL}/predict/{concept}/batch",
            json=payload,
            timeout=120
        )
    except requests.RequestException as e:
        raise PredictionAPIError(
            f"Unable to connect to FastAPI at {API_BASE_URL}."
        ) from e

    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text

        raise PredictionAPIError(f"FastAPI returned HTTP {response.status_code}: {detail}")

    # Response JSON
    try:
        result = response.json()
    except ValueError as e:
        raise PredictionAPIError("FastAPI returned invalid JSON.") from e

    if not isinstance(result, dict):
        raise PredictionAPIError("FastAPI returned unexpected response format.")

    if "predictions" not in result:
        raise PredictionAPIError("FastAPI response missing 'predictions' key.")

    return result