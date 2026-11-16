"""
Live Streaming & Camera Stream Connector.

Supports authorized RTSP streams, municipal CCTV feeds, and live browser camera feeds.
Enforces strict SSRF protections against internal network probing and cloud metadata abuse.
"""

import time
from typing import Any, Dict, Generator, List, Optional, Tuple
import cv2
import numpy as np

from app.ingestion.connectors.base import (
    BaseImageryConnector,
    ImageryItem,
    ImagerySearchResult,
    validate_remote_url,
    SSRFValidationError,
)
from app.core.logging import logger

LIVE_STREAM_ATTRIBUTION = "Authorized Live Source / User Stream"
LIVE_STREAM_LICENSE = "Proprietary/Authorized-Live-Feed"


class LiveStreamConnector(BaseImageryConnector):
    """
    Connects to authorized RTSP or HTTP/HTTPS live camera feeds.
    Safely samples frames at configurable intervals.
    """

    def __init__(
        self,
        stream_url: Optional[str] = None,
        sample_fps: float = 1.0,
        stream_timeout_sec: float = 10.0
    ):
        super().__init__(
            provider_name="live_stream",
            rate_limit_per_min=120,
            max_retries=2,
            circuit_trip_threshold=3,
            circuit_recovery_time_sec=30.0
        )
        self.stream_url = stream_url
        self.sample_fps = sample_fps
        self.stream_timeout_sec = stream_timeout_sec

    def validate_stream_url(self, url: str) -> str:
        """Validates RTSP or HTTP streaming URL against SSRF rules."""
        return validate_remote_url(
            url,
            allowed_schemes=("rtsp", "rtsps", "http", "https")
        )

    def search_imagery(
        self,
        bbox: Tuple[float, float, float, float],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 10,
        page_token: Optional[str] = None
    ) -> ImagerySearchResult:
        """Live stream connector returns active live feeds within geographic scope."""
        min_lon, min_lat, max_lon, max_lat = bbox
        center_lon = (min_lon + max_lon) / 2.0
        center_lat = (min_lat + max_lat) / 2.0

        item = ImageryItem(
            item_id="live_camera_feed",
            provider="live_stream",
            coordinates=(round(center_lon, 6), round(center_lat, 6)),
            capture_timestamp=str(int(time.time())),
            ingestion_timestamp=str(int(time.time())),
            url=self.stream_url,
            license=LIVE_STREAM_LICENSE,
            attribution=LIVE_STREAM_ATTRIBUTION,
            is_live=True,
            raw_metadata={"type": "realtime_stream", "sample_fps": self.sample_fps}
        )
        return ImagerySearchResult(items=[item], total_count=1, provider="live_stream", query_bbox=bbox)

    def get_image_metadata(self, image_id: str) -> Optional[ImageryItem]:
        return ImageryItem(
            item_id=image_id,
            provider="live_stream",
            coordinates=(77.2090, 28.6139),
            capture_timestamp=str(int(time.time())),
            ingestion_timestamp=str(int(time.time())),
            url=self.stream_url,
            license=LIVE_STREAM_LICENSE,
            attribution=LIVE_STREAM_ATTRIBUTION,
            is_live=True,
            raw_metadata={"stream_id": image_id}
        )

    def fetch_image_bytes(self, image_id_or_url: str) -> bytes:
        """Captures a single frame from the live stream and encodes it as JPEG."""
        url = image_id_or_url or self.stream_url
        if not url:
            raise ValueError("No live stream URL specified.")

        validated_url = self.validate_stream_url(url)
        cap = cv2.VideoCapture(validated_url)
        try:
            if not cap.isOpened():
                raise ConnectionError(f"Unable to open live video stream at {url}")
            ret, frame = cap.read()
            if not ret or frame is None:
                raise ConnectionError("Failed to read video frame from live stream.")

            success, buffer = cv2.imencode(".jpg", frame)
            if not success:
                raise ValueError("Failed to encode frame to JPEG format.")
            return buffer.tobytes()
        finally:
            cap.release()

    def stream_frames(
        self,
        url: str,
        max_frames: int = 100
    ) -> Generator[Tuple[int, np.ndarray, float], None, None]:
        """
        Yields (frame_index, frame_bgr, timestamp) from validated RTSP stream.
        """
        validated_url = self.validate_stream_url(url)
        cap = cv2.VideoCapture(validated_url)
        frame_idx = 0
        try:
            while cap.isOpened() and frame_idx < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                yield (frame_idx, frame, time.time())
                time.sleep(1.0 / max(0.1, self.sample_fps))
        finally:
            cap.release()
