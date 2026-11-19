import pytest
from unittest.mock import Mock, patch

from app.gis.osm_importer import fetch_osm_pedestrian_segment_models


def test_osm_import_parses_real_way_geometry_and_preserves_source_identity():
    payload = {
        "elements": [
            {
                "type": "way",
                "id": 123456,
                "tags": {"highway": "footway", "name": "Market Walk"},
                "geometry": [
                    {"lat": 20.2961, "lon": 85.8245},
                    {"lat": 20.2965, "lon": 85.8250},
                    {"lat": 20.2970, "lon": 85.8257},
                ],
            }
        ]
    }
    response = Mock()
    response.json.return_value = payload
    response.raise_for_status.return_value = None

    with patch("requests.post", return_value=response) as post:
        segments = fetch_osm_pedestrian_segment_models(
            min_lon=85.82,
            min_lat=20.29,
            max_lon=85.83,
            max_lat=20.30,
        )

    post.assert_called_once()
    assert len(segments) == 1
    segment = segments[0]
    assert segment.id == "osm-way-123456"
    assert segment.osm_way_id == "123456"
    assert segment.name == "Market Walk"
    assert segment.score_grade == "U"
    assert '"assessment_status": "UNASSESSED"' in segment.score_criteria_breakdown
    assert '"source": "OpenStreetMap"' in segment.score_criteria_breakdown
    assert segment.length_meters > 0


@pytest.mark.parametrize(
    "bbox",
    [
        (85.83, 20.29, 85.82, 20.30),
        (85.82, 20.30, 85.83, 20.29),
        (-181, 20.29, 85.83, 20.30),
        (85.82, -91, 85.83, 20.30),
        (85.0, 20.0, 86.0, 21.0),
    ],
)
def test_osm_import_rejects_invalid_or_oversized_bbox_without_network_call(bbox):
    with patch("requests.post") as post:
        with pytest.raises(ValueError):
            fetch_osm_pedestrian_segment_models(*bbox)
    post.assert_not_called()
