from typing import Dict, Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)


class PredictResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    model_version: str
    model_stage: str
    container_id: str
    drift_score: float


class FeedbackRequest(BaseModel):
    text: str
    predicted_label: str
    correct_label: Optional[str] = None
    was_correct: bool


class FeedbackResponse(BaseModel):
    status: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    container_id: str
