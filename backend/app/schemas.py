from typing import Literal

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    order_id: str = Field(min_length=1, max_length=100)
    stage: Literal["A", "B", "C"]


class PredictionResponse(BaseModel):
    order_id: str
    stage: Literal["A", "B", "C"]
    delay_probability: float = Field(ge=0, le=1)
    risk_category: Literal["low", "medium", "high"]
    threshold: float = Field(ge=0, le=1)
    prediction_type: str = "retrospective_stage_simulation"
