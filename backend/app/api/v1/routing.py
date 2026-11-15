from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.db.models import PedestrianSegment
from app.schemas.routing import RouteRequest, RouteResponse
from app.gis.routing_engine import AccessibilityRoutingEngine
from app.gis.osm_importer import create_pedestrian_segment_models

router = APIRouter(prefix="/routing", tags=["Routing"])

@router.post("/accessible-route", response_model=RouteResponse)
def compute_accessible_route(
    request: RouteRequest,
    db: Session = Depends(get_db)
):
    """
    Computes an optimal wheelchair-accessible pedestrian route:
    - Avoids stairs and impassable vertical obstacles.
    - Heavily penalizes damaged surfaces, missing curb ramps, and steep slopes.
    - Recommends verified safe routes with detailed turn-by-turn criteria.
    """
    segments = db.query(PedestrianSegment).all()
    if not segments:
        # Seed demo network if empty
        demo_segs = create_pedestrian_segment_models()
        for s in demo_segs:
            db.add(s)
        db.commit()
        segments = db.query(PedestrianSegment).all()

    engine = AccessibilityRoutingEngine(segments)
    route_result = engine.find_accessible_route(
        start_lat=request.start.latitude,
        start_lon=request.start.longitude,
        end_lat=request.destination.latitude,
        end_lon=request.destination.longitude,
        wheelchair_profile=request.wheelchair_profile,
        avoid_unverified=request.avoid_unverified_barriers
    )

    if not route_result.success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=route_result.routing_notes[0] if route_result.routing_notes else "Could not calculate accessible route."
        )

    return route_result
