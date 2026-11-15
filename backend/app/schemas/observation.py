from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field

class BoundingBox(BaseModel):
    ymin: float = Field(..., ge=0.0, le=1.0)
    xmin: float = Field(..., ge=0.0, le=1.0)
    ymax: float = Field(..., ge=0.0, le=1.0)
    xmax: float = Field(..., ge=0.0, le=1.0)

class BarrierEvidence(BaseModel):
    feature_type: str = Field(..., description="e.g. edge_rungs, surface_roughness, clearance_width")
    measured_metric: float = Field(..., description="Numerical feature metric extracted by model")
    threshold_applied: float = Field(..., description="Baseline threshold used for classification")
    explanation: str = Field(..., description="Human-understandable justification of detection")
    metadata: Dict[str, Any] = Field(default_factory=dict)

class BarrierDetectionSchema(BaseModel):
    id: str
    barrier_type: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    uncertainty_score: float = Field(..., ge=0.0, le=1.0)
    is_confirmed: bool = False
    is_uncertain: bool = False
    bbox: BoundingBox
    polygon: Optional[List[List[float]]] = None # [[x1, y1], [x2, y2], ...]
    evidence: Optional[BarrierEvidence] = None
    created_at: datetime

    model_config = {"from_attributes": True}

class ObservationResponse(BaseModel):
    id: str
    contributor_id: Optional[str] = None
    segment_id: Optional[str] = None
    media_url: str
    media_type: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    status: str # tentative, verified, disputed, rejected
    verification_score: int
    captured_at: Optional[datetime] = None
    created_at: datetime
    detections: List[BarrierDetectionSchema] = []

    model_config = {"from_attributes": True}

class ObservationFilterParams(BaseModel):
    barrier_type: Optional[str] = None
    status: Optional[str] = None
    min_confidence: Optional[float] = None
    segment_id: Optional[str] = None
    limit: int = 100
    offset: int = 0
