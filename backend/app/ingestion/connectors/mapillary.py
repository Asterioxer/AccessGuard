"""
Mapillary API v4 Imagery Connector.

Integrates the official Mapillary v4 Graph API for retrieving street-level
imagery within geographic bounding boxes, extracting capture timestamps, sequence IDs,
camera compass angles, and high-resolution thumbnail resources.
"""

import os
import time
from typing import Any, Dict, List, Optional, Tuple
import requests

from app.ingestion.connectors.base import (
    BaseImageryConnector,
    ImageryItem,
    ImagerySearchResult,
    validate_remote_url,
    MAX_IMAGE_BYTES,
    DEFAULT_TIMEOUT_SEC,
)
from app.core.logging import logger

MAPILLARY_GRAPH_URL = "https://graph.mapillary.com"
MAPILLARY_ALLOWED_DOMAINS = [
    "graph.mapillary.com",
    "images.mapillary.com",
    "mapillary.com",
    "fbcdn.net",
    "fbsbx.com",
]
MAPILLARY_ATTRIBUTION = "Imagery © Mapillary, Open Database License / CC BY-SA 4.0"
MAPILLARY_LICENSE = "CC-BY-SA-4.0"


class MapillaryConnector(BaseImageryConnector):
    """
    Connects to Mapillary API v4.
    If MAPILLARY_CLIENT_TOKEN is not configured in the environment,
    it provides explicit fixture responses for automated offline testing
    while marking status as unauthenticated/fixture.
    """

    def __init__(
        self,
        client_token: Optional[str] = None,
        rate_limit_per_min: int = 60,
        min_quality_score: float = 0.4
    ):
        super().__init__(
            provider_name="mapillary",
            rate_limit_per_min=rate_limit_per_min,
            max_retries=3,
            circuit_trip_threshold=5,
            circuit_recovery_time_sec=60.0
        )
        self.client_token = client_token or os.environ.get("MAPILLARY_CLIENT_TOKEN", "").strip()
        self.min_quality_score = min_quality_score

    @property
    def is_authenticated(self) -> bool:
        return bool(self.client_token)

    def search_imagery(
        self,
        bbox: Tuple[float, float, float, float], # (min_lon, min_lat, max_lon, max_lat)
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 50,
        page_token: Optional[str] = None,
        allow_fixtures: bool = False
    ) -> ImagerySearchResult:
        """
        Queries Mapillary images within a bounding box.
        Format: bbox = min_lon,min_lat,max_lon,max_lat
        """
        self._check_circuit()
        self._throttle()

        min_lon, min_lat, max_lon, max_lat = bbox

        if not self.is_authenticated:
            if allow_fixtures:
                logger.info("Mapillary token unset: generating explicit fixture results for bounding box.")
                return self._generate_fixture_results(bbox, limit)
            logger.warning("Mapillary token unset: returning unauthenticated result.")
            return ImagerySearchResult(
                items=[],
                total_count=0,
                provider="mapillary",
                query_bbox=bbox,
                status="unauthenticated",
                error_message="CREDENTIALS_MISSING: MAPILLARY_CLIENT_TOKEN environment variable not set. Real imagery acquisition requires an authorized token."
            )

        # Authenticated live query
        fields = "id,captured_at,geometry,thumb_1024_url,sequence,quality_score,compass_angle"
        bbox_str = f"{min_lon},{min_lat},{max_lon},{max_lat}"
        url = f"{MAPILLARY_GRAPH_URL}/images"
        params = {
            "bbox": bbox_str,
            "fields": fields,
            "limit": min(limit, 100),
            "access_token": self.client_token
        }
        if start_date:
            params["start_time"] = start_date
        if end_date:
            params["end_time"] = end_date
        if page_token:
            params["after"] = page_token

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT_SEC)
                if resp.status_code == 200:
                    data = resp.json()
                    self._record_success()
                    return self._parse_api_response(data, bbox)
                elif resp.status_code == 429:
                    logger.warning(f"Mapillary rate limit hit (attempt {attempt}). Backing off.")
                    time.sleep(2.0 ** attempt)
                elif resp.status_code in (401, 403):
                    logger.error(f"Mapillary authentication error {resp.status_code}: {resp.text}")
                    self._record_failure()
                    raise PermissionError(f"Mapillary API authentication failed ({resp.status_code}).")
                else:
                    logger.warning(f"Mapillary HTTP error {resp.status_code}: {resp.text}")
                    self._record_failure()
            except requests.RequestException as e:
                logger.warning(f"Mapillary connection error (attempt {attempt}): {e}")
                if attempt == self.max_retries:
                    self._record_failure()
                    raise ConnectionError(f"Mapillary API request failed after {self.max_retries} attempts: {e}")
                time.sleep(1.0 * attempt)

        return ImagerySearchResult(items=[], total_count=0, provider="mapillary", query_bbox=bbox)

    def _parse_api_response(self, data: Dict[str, Any], bbox: Tuple[float, float, float, float]) -> ImagerySearchResult:
        raw_items = data.get("data", [])
        items: List[ImageryItem] = []

        for entry in raw_items:
            q_score = entry.get("quality_score", 1.0)
            if q_score is not None and q_score < self.min_quality_score:
                continue

            geom = entry.get("geometry", {})
            coords = geom.get("coordinates", [0.0, 0.0])
            lon, lat = coords[0], coords[1]

            item = ImageryItem(
                item_id=str(entry.get("id")),
                provider="mapillary",
                coordinates=(lon, lat),
                capture_timestamp=str(entry.get("captured_at")),
                ingestion_timestamp=str(int(time.time())),
                url=entry.get("thumb_1024_url"),
                heading=entry.get("compass_angle"),
                sequence_id=entry.get("sequence"),
                quality_score=q_score,
                license=MAPILLARY_LICENSE,
                attribution=MAPILLARY_ATTRIBUTION,
                is_live=False,
                raw_metadata=entry
            )
            items.append(item)

        paging = data.get("paging", {})
        next_token = paging.get("cursors", {}).get("after")

        return ImagerySearchResult(
            items=items,
            total_count=len(items),
            next_page_token=next_token,
            provider="mapillary",
            query_bbox=bbox
        )

    def _generate_fixture_results(self, bbox: Tuple[float, float, float, float], limit: int) -> ImagerySearchResult:
        """Returns deterministic fixture items for unit testing without network dependencies."""
        min_lon, min_lat, max_lon, max_lat = bbox
        items: List[ImageryItem] = []
        for i in range(min(limit, 3)):
            lon = min_lon + (max_lon - min_lon) * ((i + 1) / 4.0)
            lat = min_lat + (max_lat - min_lat) * ((i + 1) / 4.0)
            item = ImageryItem(
                item_id=f"mapillary_fixture_{i+1}",
                provider="mapillary",
                coordinates=(round(lon, 6), round(lat, 6)),
                capture_timestamp="2026-05-10T14:30:00Z",
                ingestion_timestamp=str(int(time.time())),
                url=f"https://images.mapillary.com/fixture_{i+1}.jpg",
                heading=45.0 * i,
                sequence_id=f"seq_{int(min_lon * 1000)}",
                quality_score=0.88,
                license=MAPILLARY_LICENSE,
                attribution=MAPILLARY_ATTRIBUTION,
                is_live=False,
                raw_metadata={"mode": "fixture_unauthenticated", "test_index": i}
            )
            items.append(item)

        return ImagerySearchResult(
            items=items,
            total_count=len(items),
            next_page_token=None,
            provider="mapillary",
            query_bbox=bbox
        )

    def get_image_metadata(self, image_id: str) -> Optional[ImageryItem]:
        self._check_circuit()
        if not self.is_authenticated:
            return ImageryItem(
                item_id=image_id,
                provider="mapillary",
                coordinates=(77.2090, 28.6139),
                capture_timestamp="2026-05-10T14:30:00Z",
                ingestion_timestamp=str(int(time.time())),
                url=f"https://images.mapillary.com/{image_id}.jpg",
                license=MAPILLARY_LICENSE,
                attribution=MAPILLARY_ATTRIBUTION,
                is_live=False,
                raw_metadata={"mode": "fixture_unauthenticated"}
            )

        fields = "id,captured_at,geometry,thumb_1024_url,sequence,quality_score,compass_angle"
        url = f"{MAPILLARY_GRAPH_URL}/{image_id}?fields={fields}&access_token={self.client_token}"
        resp = requests.get(url, timeout=DEFAULT_TIMEOUT_SEC)
        if resp.status_code == 200:
            entry = resp.json()
            geom = entry.get("geometry", {})
            coords = geom.get("coordinates", [0.0, 0.0])
            return ImageryItem(
                item_id=str(entry.get("id")),
                provider="mapillary",
                coordinates=(coords[0], coords[1]),
                capture_timestamp=str(entry.get("captured_at")),
                ingestion_timestamp=str(int(time.time())),
                url=entry.get("thumb_1024_url"),
                heading=entry.get("compass_angle"),
                sequence_id=entry.get("sequence"),
                quality_score=entry.get("quality_score"),
                license=MAPILLARY_LICENSE,
                attribution=MAPILLARY_ATTRIBUTION,
                is_live=False,
                raw_metadata=entry
            )
        return None

    def fetch_image_bytes(self, image_id_or_url: str) -> bytes:
        """Downloads image bytes with strict SSRF and size checks."""
        url = image_id_or_url
        if not url.startswith("http"):
            meta = self.get_image_metadata(image_id_or_url)
            if not meta or not meta.url:
                raise ValueError(f"Could not resolve image URL for ID {image_id_or_url}")
            url = meta.url

        # SSRF validation against allowed domains
        validated_url = validate_remote_url(
            url,
            allowed_schemes=("http", "https"),
            allowed_domains=MAPILLARY_ALLOWED_DOMAINS
        )

        resp = requests.get(validated_url, stream=True, timeout=DEFAULT_TIMEOUT_SEC)
        resp.raise_for_status()

        content = bytearray()
        for chunk in resp.iter_content(chunk_size=65536):
            content.extend(chunk)
            if len(content) > MAX_IMAGE_BYTES:
                raise ValueError(f"Image download exceeded maximum size limit of {MAX_IMAGE_BYTES} bytes.")

        return bytes(content)
