"""
Pexels Supplementary Imagery Connector.

Provides optional supplemental stock imagery for baseline visual training
when legitimate API credentials are configured.
AccessGuard has ZERO hard dependency on Pexels.
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

PEXELS_API_URL = "https://api.pexels.com/v1"
PEXELS_ALLOWED_DOMAINS = ["api.pexels.com", "images.pexels.com", "pexels.com"]
PEXELS_ATTRIBUTION = "Photos provided by Pexels contributors"
PEXELS_LICENSE = "Pexels License (Free to use)"


class PexelsConnector(BaseImageryConnector):
    """
    Supplementary dataset connector for Pexels.
    Completely optional; gracefully degrades when no API key is provided.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        rate_limit_per_min: int = 30
    ):
        super().__init__(
            provider_name="pexels",
            rate_limit_per_min=rate_limit_per_min,
            max_retries=2,
            circuit_trip_threshold=3,
            circuit_recovery_time_sec=60.0
        )
        self.api_key = api_key or os.environ.get("PEXELS_API_KEY", "").strip()

    @property
    def is_available(self) -> bool:
        return bool(self.api_key)

    def search_imagery(
        self,
        bbox: Tuple[float, float, float, float],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 10,
        page_token: Optional[str] = None
    ) -> ImagerySearchResult:
        """Searches pedestrian or street scenes on Pexels."""
        self._check_circuit()
        self._throttle()

        if not self.is_available or os.environ.get("ACCESSGUARD_OFFLINE_DATASETS") == "1":
            return ImagerySearchResult(items=[], total_count=0, provider="pexels", query_bbox=bbox)

        headers = {"Authorization": self.api_key}
        params = {
            "query": "sidewalk pedestrian street wheelchair",
            "per_page": min(limit, 20),
            "page": int(page_token) if page_token and page_token.isdigit() else 1
        }
        try:
            resp = requests.get(f"{PEXELS_API_URL}/search", headers=headers, params=params, timeout=DEFAULT_TIMEOUT_SEC)
            if resp.status_code == 200:
                data = resp.json()
                self._record_success()
                return self._parse_pexels_response(data, bbox)
            else:
                logger.warning(f"Pexels API error {resp.status_code}: {resp.text}")
                self._record_failure()
                return ImagerySearchResult(items=[], total_count=0, provider="pexels", query_bbox=bbox)
        except Exception as e:
            logger.warning(f"Pexels request failed: {e}")
            self._record_failure()
            return ImagerySearchResult(items=[], total_count=0, provider="pexels", query_bbox=bbox)

    def _parse_pexels_response(self, data: Dict[str, Any], bbox: Tuple[float, float, float, float]) -> ImagerySearchResult:
        min_lon, min_lat, max_lon, max_lat = bbox
        center_lon = (min_lon + max_lon) / 2.0
        center_lat = (min_lat + max_lat) / 2.0

        photos = data.get("photos", [])
        items: List[ImageryItem] = []
        for p in photos:
            src = p.get("src", {})
            img_url = src.get("large") or src.get("medium") or src.get("original")
            items.append(
                ImageryItem(
                    item_id=str(p.get("id")),
                    provider="pexels",
                    coordinates=(round(center_lon, 6), round(center_lat, 6)),
                    capture_timestamp=str(int(time.time())),
                    ingestion_timestamp=str(int(time.time())),
                    url=img_url,
                    license=PEXELS_LICENSE,
                    attribution=f"{PEXELS_ATTRIBUTION} (Photographer: {p.get('photographer')})",
                    is_live=False,
                    raw_metadata=p
                )
            )

        next_page = str(data.get("page", 1) + 1) if data.get("next_page") else None
        return ImagerySearchResult(
            items=items,
            total_count=int(data.get("total_results", len(items))),
            next_page_token=next_page,
            provider="pexels",
            query_bbox=bbox
        )

    def get_image_metadata(self, image_id: str) -> Optional[ImageryItem]:
        if not self.is_available:
            return None
        headers = {"Authorization": self.api_key}
        try:
            resp = requests.get(f"{PEXELS_API_URL}/photos/{image_id}", headers=headers, timeout=DEFAULT_TIMEOUT_SEC)
            if resp.status_code == 200:
                p = resp.json()
                src = p.get("src", {})
                return ImageryItem(
                    item_id=str(p.get("id")),
                    provider="pexels",
                    coordinates=(77.2090, 28.6139),
                    capture_timestamp=str(int(time.time())),
                    ingestion_timestamp=str(int(time.time())),
                    url=src.get("large"),
                    license=PEXELS_LICENSE,
                    attribution=f"{PEXELS_ATTRIBUTION} (Photographer: {p.get('photographer')})",
                    is_live=False,
                    raw_metadata=p
                )
        except Exception as e:
            logger.warning(f"Failed to fetch Pexels metadata: {e}")
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
            allowed_domains=PEXELS_ALLOWED_DOMAINS
        )

        resp = requests.get(validated_url, stream=True, timeout=DEFAULT_TIMEOUT_SEC)
        resp.raise_for_status()

        content = bytearray()
        for chunk in resp.iter_content(chunk_size=65536):
            content.extend(chunk)
            if len(content) > MAX_IMAGE_BYTES:
                raise ValueError(f"Image download exceeded maximum size limit of {MAX_IMAGE_BYTES} bytes.")

        return bytes(content)
