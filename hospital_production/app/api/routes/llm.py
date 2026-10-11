import logging
from fastapi import APIRouter, HTTPException
from api.schemas import LLMRequest, LLMResponse
from api.llm_providers import LLMClient

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Router for LLM endpoints
router = APIRouter(tags=["llm"])

# Sytem prompt for Agentic AI LLM
from api.llm_providers import SYSTEM_PROMPT

# -------------------------------------------------
# LLM Engine deciding explanations
# -------------------------------------------------
@router.post("/llm", response_model=LLMResponse)
async def llm_inference(request: LLMRequest):
    """LLM Inference endpoint for hospital operations analysis."""
    try:
        # Initialize LLM Client
        llm_client = LLMClient()

        messages = [message.model_dump() for message in request.messages]
        response = llm_client.llm(messages=messages)

        return LLMResponse(
            text=response["text"],
            role=response["role"],
            provider=response["provider"]
        )

    except Exception as e:
        logger.exception("Error during LLM inference")
        raise HTTPException(status_code=500, detail=str(e)) from e