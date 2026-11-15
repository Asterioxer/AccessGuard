from typing import Dict, Any
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.db.session import get_db
from app.db.models import Observation, BarrierDetection, PedestrianSegment, User

router = APIRouter(prefix="/analytics", tags=["Analytics"])

@router.get("/summary", response_model=Dict[str, Any])
def get_platform_analytics(db: Session = Depends(get_db)):
    """
    Returns platform-wide metrics and transparency stats:
    - Total observations, verified ratio
    - Barrier distribution by class
    - City-wide pedestrian accessibility index
    """
    total_obs = db.query(Observation).count()
    verified_obs = db.query(Observation).filter(Observation.status == "verified").count()
    tentative_obs = db.query(Observation).filter(Observation.status == "tentative").count()
    disputed_obs = db.query(Observation).filter(Observation.status == "disputed").count()

    # Barrier type distribution
    barrier_counts = (
        db.query(BarrierDetection.barrier_type, func.count(BarrierDetection.id))
        .group_by(BarrierDetection.barrier_type)
        .all()
    )
    barrier_dist = {b[0]: b[1] for b in barrier_counts}

    # Segment grades
    grade_counts = (
        db.query(PedestrianSegment.score_grade, func.count(PedestrianSegment.id))
        .group_by(PedestrianSegment.score_grade)
        .all()
    )
    grades_dist = {g[0]: g[1] for g in grade_counts}

    # Mean accessibility score
    avg_score = db.query(func.avg(PedestrianSegment.accessibility_score)).scalar() or 100.0

    return {
        "total_observations": total_obs,
        "verified_observations": verified_obs,
        "tentative_observations": tentative_obs,
        "disputed_observations": disputed_obs,
        "verification_rate": round(verified_obs / total_obs, 2) if total_obs > 0 else 0.0,
        "average_city_accessibility_score": round(float(avg_score), 1),
        "barrier_distribution": barrier_dist,
        "segment_grade_distribution": grades_dist
    }
