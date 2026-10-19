from typing import List, Dict, Literal, Optional
from pydantic import BaseModel, Field

class LLMMessages(BaseModel):
    role: Literal["system", "user", "assistant"] = Field(..., description="Role of the message sender.")
    content: str
    #role: Literal["executive", "clinical", "ops", "risk"] = Field(..., description="Role context for the message.")

class LLMRequest(BaseModel):
    messages: List[LLMMessages] = Field(..., description="List of messages for LLM inference.")

class HospitalDataRequest(BaseModel):
    patient_id: str = Field(...,  pattern=r"^PAT-\d{8}$")
    service: Literal["emergency", "surgery", "general_medicine", "ICU"]
    name: Optional[str] = Field(None, description="Patient's name.")

class PredictionResponse(BaseModel):
    model_user: str
    prediction: int
    probability: List[float]

class LLMResponse(BaseModel):
    text: str
    role: str
    provider: str

class LLMAgent(BaseModel):
    question: str
    role: Literal["executive", "clinical", "ops", "risk"] = Field(..., description="Role context for the message.")
    history: List[dict] | None = None

class LLMAgentResponse(BaseModel):
    answer: str
    role: str

class ChurnRequest(BaseModel):
    patient_id: str = Field(..., pattern=r"^PAT-\w{8}$")

class ChurnResponse(BaseModel):
    patient_id: str
    churn_probability: float
    churn_prediction: int
    best_submodel: str
    cached: bool