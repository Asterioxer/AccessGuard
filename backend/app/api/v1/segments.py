import json
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.db.models import PedestrianSegment, Observation
from app.schemas.segment import (
    PedestrianSegmentSchema, ScoringCriteriaBreakdown,
    SegmentGeoJSONCollection, SegmentGeoJSONFeature
)
from app.gis.scoring import calculate_segment_accessibility_score
from app.core.security import require_role, Role
from app.gis.osm_importer import fetch_osm_pedestrian_segment_models

router = APIRouter(prefix="/segments", tags=["Pedestrian Segments"])

def format_segment_schema(seg: PedestrianSegment) -> PedestrianSegmentSchema:
    breakdown_data = {}
    if seg.score_criteria_breakdown:
        try:
            breakdown_data = json.loads(seg.score_criteria_breakdown)
        except Exception:
            pass

    breakdown = ScoringCriteriaBreakdown(
        base_score=breakdown_data.get("base_score", 100.0),
        stairs_penalty=breakdown_data.get("stairs_penalty", 0.0),
        curb_ramp_missing_penalty=breakdown_data.get("curb_ramp_missing_penalty", 0.0),
        surface_damage_penalty=breakdown_data.get("surface_damage_penalty", 0.0),
        obstruction_penalty=breakdown_data.get("obstruction_penalty", 0.0),
        curb_ramp_bonus=breakdown_data.get("curb_ramp_bonus", 0.0),
        verified_weight_applied=breakdown_data.get("verified_weight_applied", 1.0),
        net_score=breakdown_data.get("net_score", seg.accessibility_score),
        penalties_detail=breakdown_data.get("penalties_detail", []),
        bonuses_detail=breakdown_data.get("bonuses_detail", [])
    )

    return PedestrianSegmentSchema(
        id=seg.id,
        osm_way_id=seg.osm_way_id,
        name=seg.name,
        start_lat=seg.start_lat,
        start_lon=seg.start_lon,
        end_lat=seg.end_lat,
        end_lon=seg.end_lon,
        geometry_geojson=seg.geometry_geojson,
        length_meters=seg.length_meters,
        accessibility_score=seg.accessibility_score,
        score_grade=seg.score_grade,
        score_breakdown=breakdown,
        barrier_count=seg.barrier_count,
        last_evaluated_at=seg.last_evaluated_at
    )

@router.get("", response_model=List[PedestrianSegmentSchema])
def list_segments(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    min_score: Optional[float] = Query(None, ge=0.0, le=100.0),
    grade: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """Retrieves pedestrian segments with accessibility scoring and pagination."""
    query = db.query(PedestrianSegment)
    
    # An empty network is a valid state. Populate this table only from an explicit,
    # provenance-bearing OSM/network import; never invent demo streets at read time.

    if grade:
        query = query.filter(PedestrianSegment.score_grade == grade.upper())
    if min_score is not None:
        query = query.filter(PedestrianSegment.accessibility_score >= min_score)

    segments = query.order_by(PedestrianSegment.name.asc()).offset(offset).limit(limit).all()
    return [format_segment_schema(s) for s in segments]

@router.get("/geojson", response_model=SegmentGeoJSONCollection)
def get_segments_geojson(db: Session = Depends(get_db)):
    """Returns pedestrian network in standard GeoJSON format for interactive map layers."""
    segments = db.query(PedestrianSegment).all()
    # Return an empty FeatureCollection until real pedestrian-network data is imported.

    features = []
    for s in segments:
        try:
            coords = json.loads(s.geometry_geojson)
        except Exception:
            coords = [[s.start_lon, s.start_lat], [s.end_lon, s.end_lat]]

        feature = SegmentGeoJSONFeature(
            type="Feature",
            geometry={
                "type": "LineString",
                "coordinates": coords
            },
            properties={
                "id": s.id,
                "name": s.name,
                "accessibility_score": s.accessibility_score,
                "score_grade": s.score_grade,
                "barrier_count": s.barrier_count,
                "length_meters": s.length_meters
            }
        )
        features.append(feature)

    return SegmentGeoJSONCollection(type="FeatureCollection", features=features)

@router.post("/import-osm")
def import_osm_pedestrian_network(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    payload: dict = Depends(require_role(Role.ADMIN)),
    db: Session = Depends(get_db)
):
    """Explicitly import real OSM pedestrian ways for a small bounding box."""
    try:
        imported = fetch_osm_pedestrian_segment_models(min_lon, min_lat, max_lon, max_lat)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"OpenStreetMap Overpass import failed: {exc}")

    inserted = 0
    for segment in imported:
        existing = db.query(PedestrianSegment).filter(PedestrianSegment.id == segment.id).first()
        if existing:
            # Refresh geometry and source tags, but preserve scores and linked observations.
            existing.name = segment.name
            existing.start_lat = segment.start_lat
            existing.start_lon = segment.start_lon
            existing.end_lat = segment.end_lat
            existing.end_lon = segment.end_lon
            existing.geometry_geojson = segment.geometry_geojson
            existing.length_meters = segment.length_meters
            existing.osm_way_id = segment.osm_way_id
        else:
            db.add(segment)
            inserted += 1

    db.commit()
    return {
        "source": "OpenStreetMap",
        "license": "ODbL-1.0",
        "bounding_box": [min_lon, min_lat, max_lon, max_lat],
        "ways_returned": len(imported),
        "new_segments_inserted": inserted,
        "message": "Imported pedestrian ways are unassessed until real observations and verification are available."
    }

@router.get("/{segment_id}", response_model=PedestrianSegmentSchema)
def get_segment(segment_id: str, db: Session = Depends(get_db)):
    """Retrieves single segment with full transparent criteria breakdown."""
    seg = db.query(PedestrianSegment).filter(PedestrianSegment.id == segment_id).first()
    if not seg:
        raise HTTPException(status_code=404, detail="Segment not found")
    return format_segment_schema(seg)

@router.post("/{segment_id}/recalculate", response_model=PedestrianSegmentSchema)
def recalculate_segment_score(segment_id: str, db: Session = Depends(get_db)):
    """Forces recalculation of accessibility score based on linked observations."""
    seg = db.query(PedestrianSegment).filter(PedestrianSegment.id == segment_id).first()
    if not seg:
        raise HTTPException(status_code=404, detail="Segment not found")

    observations = db.query(Observation).filter(Observation.segment_id == segment_id).all()
    score, grade, breakdown = calculate_segment_accessibility_score(observations)

    seg.accessibility_score = score
    seg.score_grade = grade
    seg.score_criteria_breakdown = json.dumps(breakdown.model_dump())
    seg.barrier_count = sum(len(o.detections) for o in observations)
    seg.last_evaluated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(seg)

    return format_segment_schema(seg)
