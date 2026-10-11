from typing import List, Dict, Literal, Optional
from pydantic import BaseModel, Field

class LLMMessages(BaseModel):
    role: Literal["system", "user", "assistant"] = Field(..., description="Role of the message sender.")
    content: str

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