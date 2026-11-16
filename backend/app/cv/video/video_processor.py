"""
Asynchronous Video Intelligence Pipeline and Temporal Deduplication.

Extracts frames at a configurable sampling rate, feeds them through detection
and segmentation pipelines, aggregates continuous detections into object tracks,
and distils unique physical obstacles from streaming video files.
"""

import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.cv.base import DetectedBarrier
from app.cv.pipeline import CVPipeline
from app.cv.temporal_tracker import TemporalObstacleTracker, TrackedBarrier
from app.core.logging import logger

MAX_VIDEO_FRAMES_DEFAULT = 2000
DEFAULT_SAMPLE_RATE_FPS = 2.0


@dataclass
class VideoProcessingProgress:
    """Real-time progress telemetry for video processing jobs."""
    job_id: str
    status: str                            # PENDING, PROCESSING, COMPLETED, FAILED
    progress_percent: float = 0.0          # 0.0 to 100.0
    total_frames: int = 0
    sampled_frames: int = 0
    processed_frames: int = 0
    duration_seconds: float = 0.0
    fps: float = 0.0
    unique_obstacles_count: int = 0
    confirmed_tracks: List[Dict[str, Any]] = field(default_factory=list)
    error_message: Optional[str] = None


class VideoProcessor:
    """
    Asynchronous video intelligence engine.
    Separates:
      1. Detection (per-frame bounding box)
      2. Object Track (temporal sequence of sightings across frames)
      3. Unique Physical Obstacle (deduplicated persistent hazard)
      4. Verified Infrastructure Observation (human/peer approved)
    """

    def __init__(
        self,
        cv_pipeline: Optional[CVPipeline] = None,
        tracker: Optional[TemporalObstacleTracker] = None,
        sample_rate_fps: float = DEFAULT_SAMPLE_RATE_FPS
    ):
        self.cv_pipeline = cv_pipeline or CVPipeline()
        self.tracker = tracker or TemporalObstacleTracker(
            iou_match_threshold=0.35,
            max_track_staleness_seconds=2.5,
            min_frames_for_confirmation=3
        )
        self.sample_rate_fps = sample_rate_fps
        self._jobs: Dict[str, VideoProcessingProgress] = {}

    def get_job_progress(self, job_id: str) -> Optional[VideoProcessingProgress]:
        return self._jobs.get(job_id)

    def process_video_file(
        self,
        video_path: str,
        job_id: Optional[str] = None,
        max_frames: int = MAX_VIDEO_FRAMES_DEFAULT,
        progress_callback: Optional[Callable[[VideoProcessingProgress], None]] = None
    ) -> VideoProcessingProgress:
        """
        Executes video processing with temporal deduplication.
        One obstacle in 100 frames becomes exactly 1 physical obstacle record.
        """
        jid = job_id or f"vid_{uuid.uuid4().hex[:8]}"
        progress = VideoProcessingProgress(
            job_id=jid,
            status="PROCESSING",
            progress_percent=0.0
        )
        self._jobs[jid] = progress

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            progress.status = "FAILED"
            progress.error_message = f"Unable to open video file at {video_path}"
            return progress

        try:
            total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            native_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
            duration_sec = total_video_frames / native_fps if native_fps > 0 else 0.0

            progress.total_frames = total_video_frames
            progress.fps = native_fps
            progress.duration_seconds = round(duration_sec, 2)

            # Frame skip interval to achieve sample_rate_fps
            step = max(1, int(round(native_fps / self.sample_rate_fps)))

            frame_idx = 0
            sampled_idx = 0
            unique_tracks_map: Dict[str, TrackedBarrier] = {}

            logger.info(f"Beginning video analysis for {video_path}: {total_video_frames} frames @ {native_fps:.1f} fps (step: {step})")

            while cap.isOpened() and frame_idx < total_video_frames and sampled_idx < max_frames:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                if frame_idx % step == 0:
                    sampled_idx += 1
                    current_video_time = frame_idx / native_fps

                    # Run model inference on sampled frame
                    detections: List[DetectedBarrier] = self.cv_pipeline.adapter.predict(frame)

                    # Update temporal tracker with current video timestamp
                    active_tracks, newly_confirmed = self.tracker.update(detections, current_timestamp=current_video_time)

                    for trk in active_tracks:
                        unique_tracks_map[trk.track_id] = trk

                    progress.sampled_frames = sampled_idx
                    progress.processed_frames = frame_idx
                    progress.progress_percent = round((frame_idx / max(1, total_video_frames)) * 100.0, 1)
                    progress.unique_obstacles_count = len(unique_tracks_map)

                    if progress_callback and sampled_idx % 5 == 0:
                        progress_callback(progress)

                frame_idx += 1

            # Finalize confirmed unique physical obstacles
            confirmed_list = []
            for tid, trk in unique_tracks_map.items():
                if trk.frame_count >= self.tracker.min_frames_for_confirmation or trk.is_confirmed:
                    confirmed_list.append({
                        "track_id": trk.track_id,
                        "barrier_type": trk.barrier_type,
                        "first_seen_sec": round(trk.first_seen_timestamp, 2),
                        "last_seen_sec": round(trk.last_seen_timestamp, 2),
                        "duration_sec": round(trk.duration_seconds, 2),
                        "frame_count": trk.frame_count,
                        "max_confidence": round(trk.max_confidence, 4),
                        "bbox": trk.bbox,
                        "is_physical_obstacle": True
                    })

            progress.status = "COMPLETED"
            progress.progress_percent = 100.0
            progress.unique_obstacles_count = len(confirmed_list)
            progress.confirmed_tracks = confirmed_list

            logger.info(f"Video analysis completed for {jid}: {len(confirmed_list)} unique physical obstacles identified from {sampled_idx} sampled frames.")
            return progress

        except Exception as e:
            logger.error(f"Error during video processing: {e}")
            progress.status = "FAILED"
            progress.error_message = str(e)
            return progress
        finally:
            cap.release()
