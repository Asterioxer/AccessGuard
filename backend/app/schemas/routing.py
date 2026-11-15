from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class RouteCoordinates(BaseModel):
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)

class RouteRequest(BaseModel):
    start: RouteCoordinates
    destination: RouteCoordinates
    wheelchair_profile: str = Field(default="manual_wheelchair", description="manual_wheelchair, powered_wheelchair, stroller")
    avoid_unverified_barriers: bool = True
    max_acceptable_grade: str = "C" # Don't route through D or F segments unless no other path exists

class BarrierEncountered(BaseModel):
    segment_id: str
    barrier_type: str
    confidence: float
    status: str
    severity: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None

class RouteResponse(BaseModel):
    success: bool
    total_distance_meters: float
    estimated_duration_minutes: float
    overall_accessibility_score: float
    route_grade: str
    avoids_stairs: bool
    barriers_encountered: List[BarrierEncountered] = []
    path_coordinates: List[List[float]] = [] # [[lat, lon], ...]
    segments_traversed: List[str] = []
    routing_notes: List[str] = []
    geojson: Dict[str, Any]
