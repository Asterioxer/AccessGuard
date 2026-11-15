import pytest
from app.gis.scoring import calculate_segment_accessibility_score
from app.db.models import Observation, BarrierDetection

class MockDet:
    def __init__(self, barrier_type, confidence):
        self.barrier_type = barrier_type
        self.confidence = confidence

class MockObs:
    def __init__(self, status, detections):
        self.status = status
        self.detections = detections

def test_empty_observations_gives_perfect_score():
    score, grade, breakdown = calculate_segment_accessibility_score([])
    assert score == 100.0
    assert grade == "A"
    assert breakdown.base_score == 100.0
    assert breakdown.stairs_penalty == 0.0

def test_stairs_impassable_penalty():
    # Observation with stairs
    obs = MockObs(status="verified", detections=[MockDet("stairs", 0.95)])
    score, grade, breakdown = calculate_segment_accessibility_score([obs])
    # Base 100 - (45 * 1.0 * 0.95) = 100 - 42.75 = 57.25 -> ~57.2 (Grade C or lower)
    assert score < 60.0
    assert breakdown.stairs_penalty > 40.0
    assert any("Stairs" in p for p in breakdown.penalties_detail)

def test_unverified_tentative_discount_factor():
    # Tentative has 50% discount weight
    obs_tentative = MockObs(status="tentative", detections=[MockDet("stairs", 0.90)])
    obs_verified = MockObs(status="verified", detections=[MockDet("stairs", 0.90)])
    
    score_tentative, _, _ = calculate_segment_accessibility_score([obs_tentative])
    score_verified, _, _ = calculate_segment_accessibility_score([obs_verified])
    
    # Verified penalty should be heavier than tentative
    assert score_verified < score_tentative

def test_curb_ramp_bonus():
    obs_damage = MockObs(status="verified", detections=[MockDet("surface_damage", 0.80)])
    obs_ramp = MockObs(status="verified", detections=[MockDet("curb_ramp_present", 0.90)])
    
    score_damage_only, _, _ = calculate_segment_accessibility_score([obs_damage])
    score_with_ramp, _, breakdown = calculate_segment_accessibility_score([obs_damage, obs_ramp])
    
    assert score_with_ramp > score_damage_only
    assert breakdown.curb_ramp_bonus > 0.0
