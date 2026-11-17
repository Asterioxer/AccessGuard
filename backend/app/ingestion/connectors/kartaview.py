"""
KartaView (OpenStreetCam) Public & Authenticated Imagery Connector.

Integrates KartaView's public photo and sequence endpoints for retrieving
geocoded street-level imagery, tracks, heading angles, and capture timestamps.
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

KARTAVIEW_API_URL = "https://api.openstreetcam.org"
KARTAVIEW_ALLOWED_DOMAINS = ["api.openstreetcam.org", "kartaview.org", "openstreetcam.org", "images.openstreetcam.org"]
KARTAVIEW_ATTRIBUTION = "Imagery © KartaView contributors, CC BY-SA 4.0"
KARTAVIEW_LICENSE = "CC-BY-SA-4.0"


class KartaViewConnector(BaseImageryConnector):
    """
    Connects to KartaView public imagery endpoints.
    Supports both public and authenticated queries.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        rate_limit_per_min: int = 40
    ):
        super().__init__(
            provider_name="kartaview",
            rate_limit_per_min=rate_limit_per_min,
            max_retries=3,
            circuit_trip_threshold=5,
            circuit_recovery_time_sec=60.0
        )
        self.api_key = api_key or os.environ.get("KARTAVIEW_API_KEY", "").strip()

    @property
    def is_authenticated(self) -> bool:
        return bool(self.api_key)

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
        Searches KartaView photos in bbox.
        Format: bbox = min_lon, min_lat, max_lon, max_lat
        KartaView expects bbTopLeft (lat, lon) and bbBottomRight (lat, lon).
        """
        self._check_circuit()
        self._throttle()

        min_lon, min_lat, max_lon, max_lat = bbox

        # Check offline / test environment override
        if not self.is_authenticated:
            if allow_fixtures:
                logger.info("KartaView unauthenticated or offline: generating explicit fixture items.")
                return self._generate_fixture_results(bbox, limit)
            logger.warning("KartaView API key not configured; returning unauthenticated search result.")
            return ImagerySearchResult(
                items=[],
                total_count=0,
                provider="kartaview",
                query_bbox=bbox,
                status="unauthenticated",
                error_message="CREDENTIALS_MISSING: KARTAVIEW_API_KEY environment variable not set. Real imagery acquisition requires an authorized key."
            )

        page = int(page_token) if page_token and page_token.isdigit() else 1
        url = f"{KARTAVIEW_API_URL}/1.0/list/photos/"
        payload = {
            "bbTopLeft": f"{max_lat},{min_lon}",
            "bbBottomRight": f"{min_lat},{max_lon}",
            "page": page,
            "ipp": min(limit, 100)
        }
        if start_date:
            payload["startDate"] = start_date
        if end_date:
            payload["endDate"] = end_date

        headers = {}
        if self.is_authenticated:
            headers["Authorization"] = f"Bearer {self.api_key}"

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.post(url, data=payload, headers=headers, timeout=DEFAULT_TIMEOUT_SEC)
                if resp.status_code == 200:
                    data = resp.json()
                    self._record_success()
                    return self._parse_api_response(data, bbox, page)
                elif resp.status_code == 429:
                    logger.warning(f"KartaView rate limit hit (attempt {attempt}). Backing off.")
                    time.sleep(2.0 ** attempt)
                else:
                    logger.warning(f"KartaView HTTP {resp.status_code}: {resp.text}")
                    self._record_failure()
            except requests.RequestException as e:
                logger.warning(f"KartaView connection error (attempt {attempt}): {e}")
                if attempt == self.max_retries:
                    self._record_failure()
                    raise ConnectionError(f"KartaView request failed after {self.max_retries} attempts: {e}")
                time.sleep(1.0 * attempt)

        return ImagerySearchResult(items=[], total_count=0, provider="kartaview", query_bbox=bbox)

    def _parse_api_response(
        self,
        data: Dict[str, Any],
        bbox: Tuple[float, float, float, float],
        current_page: int
    ) -> ImagerySearchResult:
        osv_data = data.get("osv", {})
        raw_photos = osv_data.get("photos", [])
        total_items = int(osv_data.get("totalFilteredItems", len(raw_photos)))

        items: List[ImageryItem] = []
        for p in raw_photos:
            lat = float(p.get("lat", 0.0))
            lon = float(p.get("lng", 0.0))
            img_url = p.get("photo_url") or p.get("fileurl") or p.get("thumb_photo_url")

            item = ImageryItem(
                item_id=str(p.get("id")),
                provider="kartaview",
                coordinates=(lon, lat),
                capture_timestamp=p.get("shot_date"),
                ingestion_timestamp=str(int(time.time())),
                url=img_url,
                heading=float(p.get("heading", 0.0)) if p.get("heading") else None,
                sequence_id=str(p.get("sequence_id", "")),
                quality_score=0.85,
                license=KARTAVIEW_LICENSE,
                attribution=KARTAVIEW_ATTRIBUTION,
                is_live=False,
                raw_metadata=p
            )
            items.append(item)

        next_page = str(current_page + 1) if (current_page * len(items)) < total_items else None

        return ImagerySearchResult(
            items=items,
            total_count=total_items,
            next_page_token=next_page,
            provider="kartaview",
            query_bbox=bbox
        )

    def _generate_fixture_results(self, bbox: Tuple[float, float, float, float], limit: int) -> ImagerySearchResult:
        min_lon, min_lat, max_lon, max_lat = bbox
        items: List[ImageryItem] = []
        for i in range(min(limit, 3)):
            lon = min_lon + (max_lon - min_lon) * ((i + 1) / 5.0)
            lat = min_lat + (max_lat - min_lat) * ((i + 1) / 5.0)
            item = ImageryItem(
                item_id=f"kartaview_fixture_{i+101}",
                provider="kartaview",
                coordinates=(round(lon, 6), round(lat, 6)),
                capture_timestamp="2026-06-15T09:15:00Z",
                ingestion_timestamp=str(int(time.time())),
                url=f"https://images.openstreetcam.org/fixture_{i+101}.jpg",
                heading=90.0 * i,
                sequence_id=f"seq_kv_{i}",
                quality_score=0.90,
                license=KARTAVIEW_LICENSE,
                attribution=KARTAVIEW_ATTRIBUTION,
                is_live=False,
                raw_metadata={"mode": "fixture_mode", "index": i}
            )
            items.append(item)

        return ImagerySearchResult(
            items=items,
            total_count=len(items),
            next_page_token=None,
            provider="kartaview",
            query_bbox=bbox
        )

    def get_image_metadata(self, image_id: str) -> Optional[ImageryItem]:
        self._check_circuit()
        if not self.is_authenticated or os.environ.get("ACCESSGUARD_OFFLINE_DATASETS") == "1":
            return ImageryItem(
                item_id=image_id,
                provider="kartaview",
                coordinates=(77.2100, 28.6140),
                capture_timestamp="2026-06-15T09:15:00Z",
                ingestion_timestamp=str(int(time.time())),
                url=f"https://images.openstreetcam.org/{image_id}.jpg",
                license=KARTAVIEW_LICENSE,
                attribution=KARTAVIEW_ATTRIBUTION,
                is_live=False,
                raw_metadata={"mode": "fixture_mode"}
            )

        url = f"{KARTAVIEW_API_URL}/1.0/photo/{image_id}"
        resp = requests.get(url, timeout=DEFAULT_TIMEOUT_SEC)
        if resp.status_code == 200:
            p = resp.json().get("osv", {}).get("photo", {})
            return ImageryItem(
                item_id=str(p.get("id")),
                provider="kartaview",
                coordinates=(float(p.get("lng", 0.0)), float(p.get("lat", 0.0))),
                capture_timestamp=p.get("shot_date"),
                ingestion_timestamp=str(int(time.time())),
                url=p.get("photo_url"),
                heading=float(p.get("heading", 0.0)) if p.get("heading") else None,
                sequence_id=str(p.get("sequence_id", "")),
                license=KARTAVIEW_LICENSE,
                attribution=KARTAVIEW_ATTRIBUTION,
                is_live=False,
                raw_metadata=p
            )
        return None

    def fetch_image_bytes(self, image_id_or_url: str) -> bytes:
        url = image_id_or_url
        if not url.startswith("http"):
            meta = self.get_image_metadata(image_id_or_url)
            if not meta or not meta.url:
                raise ValueError(f"Could not resolve image URL for ID {image_id_or_url}")
            url = meta.url

        validated_url = validate_remote_url(
            url,
            allowed_schemes=("http", "https"),
            allowed_domains=KARTAVIEW_ALLOWED_DOMAINS
        )

        resp = requests.get(validated_url, stream=True, timeout=DEFAULT_TIMEOUT_SEC)
        resp.raise_for_status()

        content = bytearray()
        for chunk in resp.iter_content(chunk_size=65536):
            content.extend(chunk)
            if len(content) > MAX_IMAGE_BYTES:
                raise ValueError(f"Image download exceeded maximum size limit of {MAX_IMAGE_BYTES} bytes.")

        return bytes(content)
