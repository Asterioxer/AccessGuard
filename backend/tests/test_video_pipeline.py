"""
Unit & Integration Tests for Video Intelligence and Temporal Deduplication.
"""

import os
import cv2
import numpy as np
from app.cv.video import VideoProcessor


def test_video_processor_temporal_deduplication(tmp_path):
    """
    Verify video processor suppresses multi-frame duplicates:
    An obstacle appearing continuously across 30 frames must not generate 30 reports.
    """
    # Create temporary synthetic video file (30 frames @ 30 FPS = 1.0 second)
    video_file = tmp_path / "test_walkway_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_file), fourcc, 30.0, (640, 480))

    for frame_i in range(30):
        frame = np.full((480, 640, 3), 160, dtype=np.uint8)
        # Persistent staircase across all 30 frames
        for step_y in range(250, 400, 20):
            cv2.line(frame, (100, step_y), (540, step_y), (30, 30, 30), 3)
        out.write(frame)
    out.release()

    processor = VideoProcessor(sample_rate_fps=10.0)
    result = processor.process_video_file(str(video_file))

    assert result.status == "COMPLETED"
    assert result.total_frames == 30
    assert result.sampled_frames >= 5
    # The persistent stair obstacle must be deduplicated into 1 physical track
    assert result.unique_obstacles_count <= 2
    for trk in result.confirmed_tracks:
        assert trk["is_physical_obstacle"] is True
        assert trk["frame_count"] >= 3
