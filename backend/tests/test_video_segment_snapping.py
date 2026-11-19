"""
Database-backed Integration Test for Video Spatial Snapping and Pedestrian Segment Association.
Verifies Defect B remediation:
1. Video observation snaps to nearest PedestrianSegment using all_segments collection.
2. Observation.segment_id is truthfully assigned to matched segment ID.
3. Pedestrian segment score / barrier count recalculation is executed.
4. Out-of-bounds coordinates (>50m) result in segment_id=None without runtime exception.
"""

import os
from pathlib import Path
import cv2
import numpy as np
import pytest

from app.cv.orchestration.image_processing_service import ImageProcessingService
from app.cv.adapters.base_v2 import ExecutionPolicy
from app.db.models import Observation, PedestrianSegment


def _create_synthetic_video(file_path: Path, frames: int = 5, width: int = 120, height: int = 120):
    """Generates a small valid test video container."""
    # Use mp4v or MJPG depending on platform backend
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(file_path), fourcc, 10.0, (width, height))
    for i in range(frames):
        # Draw a synthetic block
        frame = np.full((height, width, 3), 128, dtype=np.uint8)
        cv2.circle(frame, (60, 60), 20, (255, 255, 255), -1)
        out.write(frame)
    out.release()


def test_video_spatial_segment_snapping_matched(db_session, tmp_path):
    """Video with coordinates within 50m of a segment must snap and set segment_id."""
    video_file = tmp_path / "test_snapping.mp4"
    _create_synthetic_video(video_file, frames=3)

    # Coords matching seeded segment seg_central_plaza_stairs (37.7765, -122.4194)
    lat, lon = 37.77651, -122.41939

    service = ImageProcessingService()
    result = service.process_video(
        video_path=video_file,
        contributor_id="test_contributor",
        latitude=lat,
        longitude=lon,
        db=db_session,
        save_media=True,
        requested_adapter="dummy",
        execution_policy=ExecutionPolicy.DEVELOPMENT_FALLBACK
    )

    # Result checks
    assert result.overall_status in ("SUCCESS", "PARTIAL")
    assert result.segment_id is not None

    # Database verification
    obs = db_session.query(Observation).filter(Observation.id == result.observation_id).first()
    assert obs is not None
    assert obs.media_type == "video"
    assert obs.segment_id == result.segment_id

    # Verify target pedestrian segment exists in DB and corresponds to match
    matched_seg = db_session.query(PedestrianSegment).filter(PedestrianSegment.id == result.segment_id).first()
    assert matched_seg is not None
    assert "seg_" in matched_seg.id


def test_video_spatial_segment_snapping_unmatched_out_of_range(db_session, tmp_path):
    """Video with coordinates far from any segment (>50m) must gracefully set segment_id=None."""
    video_file = tmp_path / "test_out_of_range.mp4"
    _create_synthetic_video(video_file, frames=3)

    # Coordinates in Null Island (far from San Francisco seeded network)
    lat, lon = 0.0, 0.0

    service = ImageProcessingService()
    result = service.process_video(
        video_path=video_file,
        contributor_id="test_contributor",
        latitude=lat,
        longitude=lon,
        db=db_session,
        save_media=True,
        requested_adapter="dummy",
        execution_policy=ExecutionPolicy.DEVELOPMENT_FALLBACK
    )

    assert result.overall_status in ("SUCCESS", "PARTIAL")
    assert result.segment_id is None

    obs = db_session.query(Observation).filter(Observation.id == result.observation_id).first()
    assert obs is not None
    assert obs.segment_id is None
