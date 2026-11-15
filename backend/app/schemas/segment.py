from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field

class ScoringCriteriaBreakdown(BaseModel):
    base_score: float = 100.0
    stairs_penalty: float = 0.0
    curb_ramp_missing_penalty: float = 0.0
    surface_damage_penalty: float = 0.0
    obstruction_penalty: float = 0.0
    curb_ramp_bonus: float = 0.0
    verified_weight_applied: float = 1.0
    net_score: float = 100.0
    penalties_detail: List[str] = []
    bonuses_detail: List[str] = []

class PedestrianSegmentSchema(BaseModel):
    id: str
    osm_way_id: Optional[str] = None
    name: str
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float
    geometry_geojson: str
    length_meters: float
    accessibility_score: float = Field(..., ge=0.0, le=100.0)
    score_grade: str # A, B, C, D, F, U
    score_breakdown: ScoringCriteriaBreakdown
    barrier_count: int
    last_evaluated_at: datetime

    model_config = {"from_attributes": True}

class SegmentGeoJSONFeature(BaseModel):
    type: str = "Feature"
    geometry: Dict[str, Any]
    properties: Dict[str, Any]

class SegmentGeoJSONCollection(BaseModel):
    type: str = "FeatureCollection"
    features: List[SegmentGeoJSONFeature]
