"""
AccessGuard External Imagery Connectors.

Modular, SSRF-safe connectors for acquiring street-level and remote sensing imagery
from Mapillary, KartaView, Authorized Live Streams, Copernicus, and Pexels.
"""

from app.ingestion.connectors.base import (
    BaseImageryConnector,
    ImageryItem,
    ImagerySearchResult,
    validate_remote_url,
    SSRFValidationError,
    CircuitState,
)
from app.ingestion.connectors.mapillary import MapillaryConnector
from app.ingestion.connectors.kartaview import KartaViewConnector
from app.ingestion.connectors.live_stream import LiveStreamConnector
from app.ingestion.connectors.copernicus import CopernicusConnector
from app.ingestion.connectors.pexels import PexelsConnector

__all__ = [
    "BaseImageryConnector",
    "ImageryItem",
    "ImagerySearchResult",
    "validate_remote_url",
    "SSRFValidationError",
    "CircuitState",
    "MapillaryConnector",
    "KartaViewConnector",
    "LiveStreamConnector",
    "CopernicusConnector",
    "PexelsConnector",
]
