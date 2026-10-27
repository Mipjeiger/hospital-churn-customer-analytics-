from typing import List, Dict, Literal, Optional, Any
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

class PredictionRequest(BaseModel):
    patient_ids: List[str] = Field(
        ...,
        min_length=1,
        max_length=10_000,
        description="Patient IDs used for database-backed prediction.",
    )
    submodel: Optional[str] = Field(
        default=None,
        description="Optional model submodel. The API selects the best model when omitted.",
    )

class PredictionItem(BaseModel):
    patient_id: str
    prediction: Any
    probability: Optional[float] = None
    probabilities: Optional[List[float]] = None
    model_user: str
    submodel: str

class PredictionResponse(BaseModel):
    concept: str
    task: str
    count: int
    predictions: List[PredictionItem]

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