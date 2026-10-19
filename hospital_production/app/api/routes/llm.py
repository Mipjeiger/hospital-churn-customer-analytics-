import logging
from fastapi import APIRouter, HTTPException
from api.schemas import LLMRequest, LLMResponse, LLMAgent, LLMAgentResponse
from api.llm_providers import LLMClient
from app.ml.tools import TOOLS, TOOL_MAP
from app.llm.mlflow_llm import traced_agent
from app.llm.prompts import load_prompt

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Router for LLM endpoints
router = APIRouter(tags=["llm"])

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

# -------------------------------------------------
# LLM Ask endpoint for role-based questions fetching with agent tools
# -------------------------------------------------
@router.post("/llm/agent")
async def agent(req: LLMAgent) -> LLMAgentResponse: 
    """LLM Ask endpoint for role-based questions with agent tools."""
    try:
        answer = traced_agent(
            user_message=req.question,
            tools_schema=TOOLS,
            tool_map=TOOL_MAP,
            history=req.history
        )
        return LLMAgentResponse(answer=answer, role=req.role)

    except Exception as e:
        logger.exception("Error during LLM agent processing")
        raise HTTPException(status_code=500, detail=str(e)) from e

@router.get("/prompt/{name}")
def get_prompt(name: str, version: int | None = None):
    return {"name": name, "template": load_prompt(name, version)}