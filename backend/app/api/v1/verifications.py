import json
from datetime import datetime, timezone
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.db.models import Observation, VerificationReview, User, PedestrianSegment
from app.schemas.verification import VerificationSubmit, VerificationResponse
from app.schemas.observation import ObservationResponse
from app.api.v1.observations import format_observation_response
from app.core.security import get_current_user_payload
from app.core.config_wrapper import get_settings
from app.gis.scoring import calculate_segment_accessibility_score

router = APIRouter(prefix="/verifications", tags=["Human Verification"])

@router.post("/submit", response_model=VerificationResponse)
def submit_verification(
    review_in: VerificationSubmit,
    user_payload: dict = Depends(get_current_user_payload),
    db: Session = Depends(get_db)
):
    """
    Submits human verification / community review for an observation.
    Calculates consensus thresholds to upgrade tentative detections to verified.
    """
    settings = get_settings()
    user_id = user_payload.get("sub")
    
    obs = db.query(Observation).filter(Observation.id == review_in.observation_id).first()
    if not obs:
        raise HTTPException(status_code=404, detail="Observation not found")

    # Prevent duplicate reviews by the same user
    existing_review = db.query(VerificationReview).filter(
        VerificationReview.observation_id == obs.id,
        VerificationReview.reviewer_id == user_id
    ).first()
    if existing_review:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You have already submitted a verification review for this observation."
        )

    # Create review
    review = VerificationReview(
        observation_id=obs.id,
        reviewer_id=user_id,
        vote=review_in.vote.lower(),
        suggested_barrier_type=review_in.suggested_barrier_type,
        notes=review_in.notes
    )
    db.add(review)
    db.commit()

    # Re-evaluate consensus on this observation
    reviews = db.query(VerificationReview).filter(VerificationReview.observation_id == obs.id).all()
    confirms = sum(1 for r in reviews if r.vote == "confirm")
    disputes = sum(1 for r in reviews if r.vote == "dispute")
    rejects = sum(1 for r in reviews if r.vote == "reject")
    total_votes = len(reviews)
    
    old_status = obs.status
    new_status = old_status
    consensus_reached = False
    
    # Net verification score
    obs.verification_score = confirms - (disputes + rejects)

    if total_votes >= settings.MIN_VERIFICATIONS_TO_CONFIRM:
        agreement_ratio = confirms / total_votes
        if agreement_ratio >= settings.CONSENSUS_AGREEMENT_RATIO:
            new_status = "verified"
            consensus_reached = True
            for det in obs.detections:
                det.is_confirmed = True
        elif (disputes + rejects) / total_votes >= 0.5:
            new_status = "disputed"
            consensus_reached = True
            for det in obs.detections:
                det.is_confirmed = False

    obs.status = new_status
    db.commit()

    # If status changed and linked to a segment, update segment score
    if obs.segment_id:
        seg = db.query(PedestrianSegment).filter(PedestrianSegment.id == obs.segment_id).first()
        if seg:
            seg_obs = db.query(Observation).filter(Observation.segment_id == seg.id).all()
            net_score, grade, breakdown = calculate_segment_accessibility_score(seg_obs)
            seg.accessibility_score = net_score
            seg.score_grade = grade
            seg.score_criteria_breakdown = json.dumps(breakdown.model_dump())
            seg.last_evaluated_at = datetime.now(timezone.utc)
            db.commit()

    return VerificationResponse(
        review_id=review.id,
        observation_id=obs.id,
        new_status=new_status,
        verification_score=obs.verification_score,
        consensus_reached=consensus_reached,
        status_changed=(old_status != new_status),
        created_at=review.created_at
    )

@router.get("/queue", response_model=List[ObservationResponse])
def get_verification_queue(
    limit: int = 25,
    db: Session = Depends(get_db)
):
    """
    Returns unverified / tentative observations needing human verification reviews.
    """
    tentative_obs = db.query(Observation).filter(
        Observation.status == "tentative"
    ).order_by(Observation.created_at.asc()).limit(limit).all()

    return [format_observation_response(o) for o in tentative_obs]
