"""
Temporal Obstacle Tracker for Real-Time Video and Camera Streams.

Prevents duplicate barrier observations across consecutive video/webcam frames.
Groups detections into continuous temporal tracks using spatial IoU and centroid proximity.
Ensures repeated frames of the same physical obstacle do not spam the database.
"""

import time
import uuid
from typing import List, Dict, Optional, Tuple
from pydantic import BaseModel, Field

from app.cv.base import DetectedBarrier
from app.cv.evaluation.metrics import compute_box_iou
from app.core.logging import logger

class TrackedBarrier(BaseModel):
    """A continuous obstacle track spanning multiple temporal frames."""
    track_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    barrier_type: str
    first_seen_timestamp: float
    last_seen_timestamp: float
    frame_count: int = 1
    max_confidence: float
    min_uncertainty: float
    bbox: Tuple[float, float, float, float] # (ymin, xmin, ymax, xmax)
    is_confirmed: bool = False
    is_persisted_to_db: bool = False

    @property
    def duration_seconds(self) -> float:
        return max(0.0, self.last_seen_timestamp - self.first_seen_timestamp)

class TemporalObstacleTracker:
    """
    Manages active obstacle tracks across streaming camera or video frames.
    Suppresses duplicate reporting of the same physical barrier.
    """

    def __init__(
        self,
        iou_match_threshold: float = 0.35,
        max_track_staleness_seconds: float = 3.0,
        min_frames_for_confirmation: int = 3
    ):
        self.iou_match_threshold = iou_match_threshold
        self.max_track_staleness_seconds = max_track_staleness_seconds
        self.min_frames_for_confirmation = min_frames_for_confirmation
        self.active_tracks: Dict[str, TrackedBarrier] = {}

    def update(
        self,
        detections: List[DetectedBarrier],
        current_timestamp: Optional[float] = None
    ) -> Tuple[List[TrackedBarrier], List[TrackedBarrier]]:
        """
        Updates tracks with new detections from the current frame.
        
        Returns:
            (all_current_active_tracks, newly_confirmed_tracks)
        """
        now = current_timestamp if current_timestamp is not None else time.time()
        self._prune_stale_tracks(now)

        matched_track_ids = set()
        newly_confirmed: List[TrackedBarrier] = []

        for det in detections:
            best_iou = 0.0
            best_track_id = None

            for tid, track in self.active_tracks.items():
                if tid in matched_track_ids:
                    continue
                # Require matching barrier type
                if track.barrier_type != det.barrier_type:
                    continue

                iou = compute_box_iou(det.bbox, track.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = tid

            if best_iou >= self.iou_match_threshold and best_track_id is not None:
                # Update existing track
                matched_track_ids.add(best_track_id)
                track = self.active_tracks[best_track_id]
                track.last_seen_timestamp = now
                track.frame_count += 1
                track.max_confidence = max(track.max_confidence, det.confidence)
                track.min_uncertainty = min(track.min_uncertainty, det.uncertainty_score)
                # Exponential moving average smoothing for bounding box
                y1, x1, y2, x2 = track.bbox
                ny1, nx1, ny2, nx2 = det.bbox
                track.bbox = (
                    round(0.7 * y1 + 0.3 * ny1, 4),
                    round(0.7 * x1 + 0.3 * nx1, 4),
                    round(0.7 * y2 + 0.3 * ny2, 4),
                    round(0.7 * x2 + 0.3 * nx2, 4)
                )

                # Check if this frame confirms track
                if not track.is_confirmed and track.frame_count >= self.min_frames_for_confirmation:
                    if track.max_confidence >= 0.70:
                        track.is_confirmed = True
                        newly_confirmed.append(track)
                        logger.info(f"Track {track.track_id} ({track.barrier_type}) promoted to CONFIRMED after {track.frame_count} frames.")
            else:
                # Start new track
                new_track = TrackedBarrier(
                    barrier_type=det.barrier_type,
                    first_seen_timestamp=now,
                    last_seen_timestamp=now,
                    frame_count=1,
                    max_confidence=det.confidence,
                    min_uncertainty=det.uncertainty_score,
                    bbox=det.bbox,
                    is_confirmed=(det.confidence >= 0.88 and self.min_frames_for_confirmation <= 1)
                )
                self.active_tracks[new_track.track_id] = new_track
                matched_track_ids.add(new_track.track_id)
                if new_track.is_confirmed:
                    newly_confirmed.append(new_track)

        return list(self.active_tracks.values()), newly_confirmed

    def _prune_stale_tracks(self, current_time: float) -> None:
        """Removes tracks that have not been observed within the staleness window."""
        stale_ids = [
            tid for tid, track in self.active_tracks.items()
            if (current_time - track.last_seen_timestamp) > self.max_track_staleness_seconds
        ]
        for tid in stale_ids:
            del self.active_tracks[tid]

    def reset(self) -> None:
        """Clears all active tracks."""
        self.active_tracks.clear()
