from typing import List, Dict, Any, Tuple
import json
from app.schemas.segment import ScoringCriteriaBreakdown

# Transparent scoring weights (penalties are negative, bonuses are positive)
BARRIER_WEIGHTS = {
    "stairs": -45.0,
    "curb_ramp_missing": -35.0,
    "steep_incline": -30.0,
    "narrow_sidewalk": -25.0,
    "construction_obstacle": -25.0,
    "surface_damage": -20.0,
    "pole_or_bollard": -20.0,
    "curb_ramp_present": 15.0,
}

STATUS_WEIGHTS = {
    "verified": 1.0,
    "tentative": 0.5,
    "disputed": 0.0,
    "rejected": 0.0
}

def calculate_segment_accessibility_score(
    observations: List[Any]
) -> Tuple[float, str, ScoringCriteriaBreakdown]:
    """
    Calculates transparent wheelchair accessibility score (0.0 to 100.0)
    for a pedestrian segment based on its associated observations and detections.
    
    Returns:
        (net_score, score_grade, breakdown_details)
    """
    base_score = 100.0
    total_penalty = 0.0
    total_bonus = 0.0
    
    penalties_detail: List[str] = []
    bonuses_detail: List[str] = []
    
    stairs_penalty = 0.0
    curb_ramp_missing_penalty = 0.0
    surface_damage_penalty = 0.0
    obstruction_penalty = 0.0
    curb_ramp_bonus = 0.0
    
    for obs in observations:
        status = getattr(obs, "status", "tentative")
        status_weight = STATUS_WEIGHTS.get(status, 0.5)
        
        # Disputed or rejected observations have 0 weight
        if status_weight <= 0.0:
            continue
            
        detections = getattr(obs, "detections", [])
        for det in detections:
            barrier_type = det.barrier_type
            raw_weight = BARRIER_WEIGHTS.get(barrier_type, 0.0)
            effective_impact = raw_weight * status_weight * det.confidence
            
            if raw_weight < 0: # Penalty
                penalty_mag = abs(effective_impact)
                total_penalty += penalty_mag
                
                if barrier_type == "stairs":
                    stairs_penalty += penalty_mag
                    penalties_detail.append(
                        f"Stairs detected (-{penalty_mag:.1f} pts, status: {status}, confidence: {det.confidence:.0%})"
                    )
                elif barrier_type == "curb_ramp_missing":
                    curb_ramp_missing_penalty += penalty_mag
                    penalties_detail.append(
                        f"Missing curb ramp at crossing (-{penalty_mag:.1f} pts, status: {status})"
                    )
                elif barrier_type == "surface_damage":
                    surface_damage_penalty += penalty_mag
                    penalties_detail.append(
                        f"Cracked/damaged surface (-{penalty_mag:.1f} pts, status: {status})"
                    )
                else:
                    obstruction_penalty += penalty_mag
                    penalties_detail.append(
                        f"Obstruction: {barrier_type.replace('_', ' ')} (-{penalty_mag:.1f} pts, status: {status})"
                    )
            elif raw_weight > 0: # Bonus
                total_bonus += effective_impact
                curb_ramp_bonus += effective_impact
                bonuses_detail.append(
                    f"Accessible curb ramp present (+{effective_impact:.1f} pts, status: {status})"
                )

    # Net score capped between 0 and 100
    net_score = max(0.0, min(100.0, base_score - total_penalty + total_bonus))
    net_score = round(net_score, 1)

    # Assign accessibility letter grade
    if net_score >= 90.0:
        grade = "A"
    elif net_score >= 75.0:
        grade = "B"
    elif net_score >= 55.0:
        grade = "C"
    elif net_score >= 35.0:
        grade = "D"
    else:
        grade = "F"

    breakdown = ScoringCriteriaBreakdown(
        base_score=base_score,
        stairs_penalty=round(stairs_penalty, 1),
        curb_ramp_missing_penalty=round(curb_ramp_missing_penalty, 1),
        surface_damage_penalty=round(surface_damage_penalty, 1),
        obstruction_penalty=round(obstruction_penalty, 1),
        curb_ramp_bonus=round(curb_ramp_bonus, 1),
        verified_weight_applied=1.0,
        net_score=net_score,
        penalties_detail=penalties_detail,
        bonuses_detail=bonuses_detail
    )

    return net_score, grade, breakdown
