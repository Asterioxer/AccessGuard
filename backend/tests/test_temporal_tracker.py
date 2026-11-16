"""
Tests for TemporalObstacleTracker, track persistence, and duplicate suppression.
"""

import pytest
from app.cv.temporal_tracker import TemporalObstacleTracker
from app.cv.base import DetectedBarrier

def test_temporal_tracking_and_deduplication():
    tracker = TemporalObstacleTracker(iou_match_threshold=0.35, min_frames_for_confirmation=3)

    # Frame 1: Detection of stairs at (0.4, 0.4, 0.7, 0.7)
    det1 = DetectedBarrier(
        barrier_type="stairs",
        confidence=0.75,
        uncertainty_score=0.25,
        bbox=(0.40, 0.40, 0.70, 0.70)
    )
    active, confirmed = tracker.update([det1], current_timestamp=100.0)
    assert len(active) == 1
    assert len(confirmed) == 0
    track_id = active[0].track_id
    assert active[0].frame_count == 1
    assert active[0].is_confirmed is False

    # Frame 2: Same obstacle slightly jittered
    det2 = DetectedBarrier(
        barrier_type="stairs",
        confidence=0.78,
        uncertainty_score=0.22,
        bbox=(0.41, 0.39, 0.71, 0.69)
    )
    active2, confirmed2 = tracker.update([det2], current_timestamp=100.2)
    assert len(active2) == 1
    # MUST MATCH SAME TRACK (duplicate suppressed)
    assert active2[0].track_id == track_id
    assert active2[0].frame_count == 2
    assert len(confirmed2) == 0

    # Frame 3: Same obstacle seen 3rd time -> Promotion to confirmed!
    det3 = DetectedBarrier(
        barrier_type="stairs",
        confidence=0.82,
        uncertainty_score=0.18,
        bbox=(0.40, 0.40, 0.70, 0.70)
    )
    active3, confirmed3 = tracker.update([det3], current_timestamp=100.4)
    assert len(active3) == 1
    assert active3[0].track_id == track_id
    assert active3[0].frame_count == 3
    assert active3[0].is_confirmed is True
    assert len(confirmed3) == 1
    assert confirmed3[0].track_id == track_id

def test_different_obstacle_types_not_merged():
    tracker = TemporalObstacleTracker()

    det_stairs = DetectedBarrier(
        barrier_type="stairs",
        confidence=0.80,
        uncertainty_score=0.20,
        bbox=(0.40, 0.40, 0.70, 0.70)
    )
    tracker.update([det_stairs], current_timestamp=10.0)

    # Coincident bounding box but DIFFERENT category (curb_ramp)
    det_ramp = DetectedBarrier(
        barrier_type="curb_ramp_present",
        confidence=0.85,
        uncertainty_score=0.15,
        bbox=(0.40, 0.40, 0.70, 0.70)
    )
    active, _ = tracker.update([det_ramp], current_timestamp=10.2)
    assert len(active) == 2 # Must be two distinct tracks

def test_stale_track_pruning():
    tracker = TemporalObstacleTracker(max_track_staleness_seconds=2.0)
    det = DetectedBarrier(
        barrier_type="stairs",
        confidence=0.70,
        uncertainty_score=0.30,
        bbox=(0.40, 0.40, 0.70, 0.70)
    )
    tracker.update([det], current_timestamp=1.0)
    assert len(tracker.active_tracks) == 1

    # Advance time beyond 2.0s staleness window with empty frame
    active, _ = tracker.update([], current_timestamp=5.0)
    assert len(active) == 0
    assert len(tracker.active_tracks) == 0
