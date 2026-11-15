from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field

class VerificationSubmit(BaseModel):
    observation_id: str
    vote: str = Field(..., description="'confirm', 'dispute', or 'reject'")
    suggested_barrier_type: Optional[str] = Field(None, description="Optional override if model misclassified")
    notes: Optional[str] = Field(None, max_length=500, description="Observer field notes")

class VerificationResponse(BaseModel):
    review_id: str
    observation_id: str
    new_status: str # tentative, verified, disputed, rejected
    verification_score: int
    consensus_reached: bool
    status_changed: bool
    created_at: datetime

    model_config = {"from_attributes": True}
