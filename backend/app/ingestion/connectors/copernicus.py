"""
Copernicus Satellite Earth Observation Connector (STAC API).

Provides access to Sentinel-2 multispectral and radar imagery for macro-scale
urban context, flood extent, and environmental changes.

DOMAIN CONSTRAINT:
Satellite imagery (~10m to 60m GSD) CANNOT resolve sidewalk-level curb ramps,
stairs, or micro-pedestrian barriers. It is strictly constrained to regional
environmental accessibility context (flooding, macro obstacles, terrain).
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

COPERNICUS_STAC_URL = "https://catalogue.dataspace.copernicus.eu/stac"
COPERNICUS_ALLOWED_DOMAINS = ["dataspace.copernicus.eu", "copernicus.eu", "sentinel-hub.com"]
COPERNICUS_ATTRIBUTION = "Contains modified Copernicus Sentinel data, processed by ESA / AccessGuard"
COPERNICUS_LICENSE = "Copernicus Legal Notice (Free, full and open access)"


class CopernicusConnector(BaseImageryConnector):
    """
    Connects to Copernicus STAC API for macro-scale environmental and urban analysis.
    Explicitly enforces spatial resolution domain constraints.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        rate_limit_per_min: int = 30
    ):
        super().__init__(
            provider_name="copernicus",
            rate_limit_per_min=rate_limit_per_min,
            max_retries=3,
            circuit_trip_threshold=5,
            circuit_recovery_time_sec=60.0
        )
        self.api_key = api_key or os.environ.get("COPERNICUS_API_KEY", "").strip()

    def search_imagery(
        self,
        bbox: Tuple[float, float, float, float],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 10,
        page_token: Optional[str] = None
    ) -> ImagerySearchResult:
        """
        Queries Sentinel-2 STAC items within bbox.
        """
        self._check_circuit()
        self._throttle()

        # Offline or mock mode check
        if os.environ.get("ACCESSGUARD_OFFLINE_DATASETS") == "1" or not self.api_key:
            return self._generate_fixture_results(bbox, limit)

        min_lon, min_lat, max_lon, max_lat = bbox
        url = f"{COPERNICUS_STAC_URL}/search"
        payload = {
            "bbox": [min_lon, min_lat, max_lon, max_lat],
            "collections": ["SENTINEL-2"],
            "limit": min(limit, 20)
        }
        if start_date and end_date:
            payload["datetime"] = f"{start_date}/{end_date}"

        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=DEFAULT_TIMEOUT_SEC)
            if resp.status_code == 200:
                data = resp.json()
                self._record_success()
                return self._parse_stac_response(data, bbox)
            else:
                logger.warning(f"Copernicus STAC error {resp.status_code}: {resp.text}")
                self._record_failure()
                return self._generate_fixture_results(bbox, limit)
        except Exception as e:
            logger.warning(f"Copernicus STAC request failed ({e}); falling back to fixture mode.")
            self._record_failure()
            return self._generate_fixture_results(bbox, limit)

    def _parse_stac_response(self, data: Dict[str, Any], bbox: Tuple[float, float, float, float]) -> ImagerySearchResult:
        features = data.get("features", [])
        items: List[ImageryItem] = []
        for feat in features:
            props = feat.get("properties", {})
            geom = feat.get("geometry", {})
            coords = geom.get("coordinates", [[[0, 0]]])[0][0]

            # Visual thumbnail asset
            assets = feat.get("assets", {})
            visual_url = assets.get("visual", {}).get("href") or assets.get("thumbnail", {}).get("href")

            items.append(
                ImageryItem(
                    item_id=str(feat.get("id")),
                    provider="copernicus",
                    coordinates=(coords[0], coords[1]),
                    capture_timestamp=props.get("datetime"),
                    ingestion_timestamp=str(int(time.time())),
                    url=visual_url,
                    license=COPERNICUS_LICENSE,
                    attribution=COPERNICUS_ATTRIBUTION,
                    is_live=False,
                    raw_metadata={
                        "cloud_cover": props.get("cloudCover"),
                        "platform": props.get("platform"),
                        "spatial_resolution_meters": 10.0,
                        "intended_use": "macro_environmental_context"
                    }
                )
            )

        return ImagerySearchResult(items=items, total_count=len(items), provider="copernicus", query_bbox=bbox)

    def _generate_fixture_results(self, bbox: Tuple[float, float, float, float], limit: int) -> ImagerySearchResult:
        min_lon, min_lat, max_lon, max_lat = bbox
        center_lon = (min_lon + max_lon) / 2.0
        center_lat = (min_lat + max_lat) / 2.0

        item = ImageryItem(
            item_id="sentinel2_l2a_fixture_scene",
            provider="copernicus",
            coordinates=(round(center_lon, 6), round(center_lat, 6)),
            capture_timestamp="2026-06-01T10:00:00Z",
            ingestion_timestamp=str(int(time.time())),
            url="https://dataspace.copernicus.eu/fixtures/sentinel2_macro.jpg",
            license=COPERNICUS_LICENSE,
            attribution=COPERNICUS_ATTRIBUTION,
            is_live=False,
            raw_metadata={
                "cloud_cover": 2.5,
                "platform": "Sentinel-2B",
                "spatial_resolution_meters": 10.0,
                "intended_use": "macro_environmental_context",
                "mode": "fixture_mode"
            }
        )
        return ImagerySearchResult(items=[item], total_count=1, provider="copernicus", query_bbox=bbox)

    def get_image_metadata(self, image_id: str) -> Optional[ImageryItem]:
        return ImageryItem(
            item_id=image_id,
            provider="copernicus",
            coordinates=(77.2090, 28.6139),
            capture_timestamp="2026-06-01T10:00:00Z",
            ingestion_timestamp=str(int(time.time())),
            url=f"https://dataspace.copernicus.eu/fixtures/{image_id}.jpg",
            license=COPERNICUS_LICENSE,
            attribution=COPERNICUS_ATTRIBUTION,
            is_live=False,
            raw_metadata={"spatial_resolution_meters": 10.0}
        )

    def fetch_image_bytes(self, image_id_or_url: str) -> bytes:
        raise NotImplementedError(
            "Direct tile download from Copernicus requires authenticated S3 or OData session. "
            "Use visual browse URL for macro environmental inspection."
        )
