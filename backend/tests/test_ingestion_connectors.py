"""
Unit Tests for External Imagery Connectors and SSRF Protections.
"""

import pytest
from app.ingestion.connectors.base import (
    validate_remote_url,
    is_private_ip,
    SSRFValidationError,
    CircuitState,
)
from app.ingestion.connectors.mapillary import MapillaryConnector
from app.ingestion.connectors.kartaview import KartaViewConnector
from app.ingestion.connectors.live_stream import LiveStreamConnector
from app.ingestion.connectors.copernicus import CopernicusConnector
from app.ingestion.connectors.pexels import PexelsConnector


def test_ssrf_validation_blocks_private_ips():
    """Verify SSRF validator blocks loopback, private IPs, and cloud metadata hosts."""
    blocked_urls = [
        "http://127.0.0.1:8000/internal",
        "http://localhost:3000/admin",
        "http://10.0.0.1/secrets",
        "http://192.168.1.1/router",
        "http://169.254.169.254/latest/meta-data/",
        "http://metadata.google.internal/computeMetadata/v1/",
        "ftp://example.com/file.txt",
    ]
    for url in blocked_urls:
        with pytest.raises(SSRFValidationError):
            validate_remote_url(url)


def test_ssrf_validation_allows_valid_public_urls():
    """Verify valid public URLs pass validation."""
    valid_urls = [
        "https://images.mapillary.com/test.jpg",
        "https://api.openstreetcam.org/photo/123",
        "https://dataspace.copernicus.eu/browse.png"
    ]
    for url in valid_urls:
        validated = validate_remote_url(url)
        assert validated == url


def test_mapillary_connector_fixture_mode():
    """Verify Mapillary connector returns valid geocoded items with attribution in fixture mode."""
    conn = MapillaryConnector(client_token="")
    bbox = (77.20, 28.60, 77.25, 28.65)

    # By default without credentials, returns explicit unauthenticated status with 0 items
    res_unauth = conn.search_imagery(bbox=bbox, limit=3)
    assert res_unauth.provider == "mapillary"
    assert len(res_unauth.items) == 0
    assert res_unauth.status == "unauthenticated"

    # When explicit allow_fixtures=True is passed for test harnesses, returns fixtures
    res = conn.search_imagery(bbox=bbox, limit=3, allow_fixtures=True)
    assert res.provider == "mapillary"
    assert len(res.items) == 3
    assert res.items[0].license == "CC-BY-SA-4.0"
    assert "Mapillary" in res.items[0].attribution
    assert 77.20 <= res.items[0].coordinates[0] <= 77.25
    assert 28.60 <= res.items[0].coordinates[1] <= 28.65
    assert res.items[0].is_live is False


def test_kartaview_connector_fixture_mode():
    """Verify KartaView connector returns geocoded photos and track metadata."""
    conn = KartaViewConnector(api_key="")
    bbox = (12.48, 41.89, 12.50, 41.91)

    # By default without credentials, returns explicit unauthenticated status with 0 items
    res_unauth = conn.search_imagery(bbox=bbox, limit=2)
    assert res_unauth.provider == "kartaview"
    assert len(res_unauth.items) == 0
    assert res_unauth.status == "unauthenticated"

    # When explicit allow_fixtures=True is passed for test harnesses, returns fixtures
    res = conn.search_imagery(bbox=bbox, limit=2, allow_fixtures=True)
    assert res.provider == "kartaview"
    assert len(res.items) == 2
    assert "KartaView" in res.items[0].attribution
    assert res.items[0].sequence_id.startswith("seq_kv_")


def test_live_stream_connector_metadata():
    """Verify live stream connector distinguishes live streams from historical imagery."""
    conn = LiveStreamConnector(stream_url="rtsp://public-cam.org/live", sample_fps=2.0)
    bbox = (77.20, 28.60, 77.25, 28.65)
    res = conn.search_imagery(bbox=bbox)

    assert res.provider == "live_stream"
    assert len(res.items) == 1
    assert res.items[0].is_live is True
    assert res.items[0].attribution.startswith("Authorized Live Source")


def test_copernicus_connector_stac_resolution_metadata():
    """Verify Copernicus connector documents macro-scale spatial resolution constraint."""
    conn = CopernicusConnector(api_key="")
    bbox = (77.0, 28.0, 77.5, 28.5)
    res = conn.search_imagery(bbox=bbox, limit=1)

    assert res.provider == "copernicus"
    assert len(res.items) == 1
    assert res.items[0].raw_metadata.get("spatial_resolution_meters") == 10.0
    assert res.items[0].raw_metadata.get("intended_use") == "macro_environmental_context"


def test_pexels_connector_graceful_degradation():
    """Verify Pexels connector gracefully returns empty results when unauthenticated."""
    conn = PexelsConnector(api_key="")
    res = conn.search_imagery(bbox=(0.0, 0.0, 1.0, 1.0))
    assert res.total_count == 0
    assert len(res.items) == 0
