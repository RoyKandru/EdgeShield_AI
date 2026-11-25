from pydantic import BaseModel, Field
from typing import Literal

class FeedbackRequest(BaseModel):
    event_id: str
    verdict: Literal["correct", "false_positive", "false_negative"]
    verified: bool = False
    reviewer: str = Field(default="user", min_length=1, max_length=80)
    notes: str = Field(default="", max_length=2000)
    label: Literal["benign", "malicious"]

class NetworkObservation(BaseModel):
    metrics: dict[str, float] = Field(min_length=1, max_length=30)

class CandidatePredictionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10000)
