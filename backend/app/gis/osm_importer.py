import json
from typing import List, Dict, Any
from app.db.models import PedestrianSegment
from app.gis.spatial_index import haversine_distance_meters

# Demo urban corridor: Downtown Innovation District
# Coords around (37.7749, -122.4194)
DEMO_PEDESTRIAN_NETWORK = [
    {
        "id": "seg_main_st_north",
        "osm_way_id": "way_101",
        "name": "Main Street Sidewalk (Northbound)",
        "start_lat": 37.7749,
        "start_lon": -122.4194,
        "end_lat": 37.7765,
        "end_lon": -122.4194,
        "coordinates": [
            [-122.4194, 37.7749],
            [-122.4194, 37.7757],
            [-122.4194, 37.7765]
        ]
    },
    {
        "id": "seg_central_plaza_stairs",
        "osm_way_id": "way_102",
        "name": "Central Plaza Steps (Direct Path - Contains Stairs)",
        "start_lat": 37.7765,
        "start_lon": -122.4194,
        "end_lat": 37.7778,
        "end_lon": -122.4194,
        "coordinates": [
            [-122.4194, 37.7765],
            [-122.4194, 37.7771],
            [-122.4194, 37.7778]
        ]
    },
    {
        "id": "seg_plaza_bypass_ramp_east",
        "osm_way_id": "way_103",
        "name": "Plaza East Accessible Ramp Bypass (Part 1)",
        "start_lat": 37.7765,
        "start_lon": -122.4194,
        "end_lat": 37.7771,
        "end_lon": -122.4178,
        "coordinates": [
            [-122.4194, 37.7765],
            [-122.4186, 37.7768],
            [-122.4178, 37.7771]
        ]
    },
    {
        "id": "seg_plaza_bypass_ramp_north",
        "osm_way_id": "way_104",
        "name": "Plaza East Accessible Ramp Bypass (Part 2 - Smooth Paved)",
        "start_lat": 37.7771,
        "start_lon": -122.4178,
        "end_lat": 37.7778,
        "end_lon": -122.4194,
        "coordinates": [
            [-122.4178, 37.7771],
            [-122.4186, 37.7775],
            [-122.4194, 37.7778]
        ]
    },
    {
        "id": "seg_market_crosswalk",
        "osm_way_id": "way_105",
        "name": "Market Street Crosswalk with Tactile Paving",
        "start_lat": 37.7778,
        "start_lon": -122.4194,
        "end_lat": 37.7788,
        "end_lon": -122.4194,
        "coordinates": [
            [-122.4194, 37.7778],
            [-122.4194, 37.7788]
        ]
    },
    {
        "id": "seg_civic_walkway_west",
        "osm_way_id": "way_106",
        "name": "Civic Center West Promenade",
        "start_lat": 37.7749,
        "start_lon": -122.4194,
        "end_lat": 37.7749,
        "end_lon": -122.4215,
        "coordinates": [
            [-122.4194, 37.7749],
            [-122.4205, 37.7749],
            [-122.4215, 37.7749]
        ]
    },
    {
        "id": "seg_transit_access_north",
        "osm_way_id": "way_107",
        "name": "Transit Station Northbound Corridor",
        "start_lat": 37.7749,
        "start_lon": -122.4215,
        "end_lat": 37.7778,
        "end_lon": -122.4215,
        "coordinates": [
            [-122.4215, 37.7749],
            [-122.4215, 37.7763],
            [-122.4215, 37.7778]
        ]
    },
    {
        "id": "seg_transit_to_market",
        "osm_way_id": "way_108",
        "name": "Transit to Market St Connector",
        "start_lat": 37.7778,
        "start_lon": -122.4215,
        "end_lat": 37.7778,
        "end_lon": -122.4194,
        "coordinates": [
            [-122.4215, 37.7778],
            [-122.4204, 37.7778],
            [-122.4194, 37.7778]
        ]
    }
]

def create_pedestrian_segment_models() -> List[PedestrianSegment]:
    """Builds database models from demo network with calculated lengths."""
    segments = []
    for data in DEMO_PEDESTRIAN_NETWORK:
        coords = data["coordinates"]
        
        # Calculate metric length along polyline
        length = 0.0
        for i in range(len(coords) - 1):
            p1 = coords[i]
            p2 = coords[i+1]
            length += haversine_distance_meters(p1[1], p1[0], p2[1], p2[0])
            
        seg = PedestrianSegment(
            id=data["id"],
            osm_way_id=data["osm_way_id"],
            name=data["name"],
            start_lat=data["start_lat"],
            start_lon=data["start_lon"],
            end_lat=data["end_lat"],
            end_lon=data["end_lon"],
            geometry_geojson=json.dumps(coords),
            length_meters=round(length, 1),
            accessibility_score=50.0,
            score_grade="U",
            score_criteria_breakdown="{}",
            barrier_count=0
        )
        segments.append(seg)
    return segments


def fetch_osm_pedestrian_segment_models(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    timeout_seconds: int = 45
) -> List[PedestrianSegment]:
    """
    Fetch real pedestrian ways from OpenStreetMap Overpass API.

    This function performs an explicit network import. It is never called during
    application startup or ordinary map reads. Returned geometries retain their
    OSM way IDs and names so imported map features can be traced to OSM.
    """
    import requests

    if not (-180 <= min_lon < max_lon <= 180):
        raise ValueError("Longitude bounds must satisfy -180 <= min_lon < max_lon <= 180.")
    if not (-90 <= min_lat < max_lat <= 90):
        raise ValueError("Latitude bounds must satisfy -90 <= min_lat < max_lat <= 90.")
    if (max_lon - min_lon) * (max_lat - min_lat) > 0.02:
        raise ValueError("Requested bounding box is too large. Import a smaller area to respect Overpass limits.")

    # Overpass bbox order is south, west, north, east.
    query = (
        '[out:json][timeout:25];'
        '('
        f'way["highway"~"^(footway|path|pedestrian|steps)$"]({min_lat},{min_lon},{max_lat},{max_lon});'
        ');out tags geom;'
    )
    response = requests.post(
        "https://overpass-api.de/api/interpreter",
        data={"data": query},
        headers={"User-Agent": "AccessGuard/1.0 (open-source accessibility mapping)"},
        timeout=timeout_seconds
    )
    response.raise_for_status()
    payload = response.json()
    elements = payload.get("elements", [])
    segments: List[PedestrianSegment] = []

    for way in elements:
        geometry = way.get("geometry") or []
        if len(geometry) < 2:
            continue
        coords = [[float(point["lon"]), float(point["lat"])] for point in geometry if "lon" in point and "lat" in point]
        if len(coords) < 2:
            continue

        length = sum(
            haversine_distance_meters(coords[i][1], coords[i][0], coords[i + 1][1], coords[i + 1][0])
            for i in range(len(coords) - 1)
        )
        tags = way.get("tags", {})
        highway = tags.get("highway", "footway")
        way_id = str(way["id"])
        name = tags.get("name") or tags.get("ref") or f"OSM {highway} way {way_id}"

        segments.append(PedestrianSegment(
            id=f"osm-way-{way_id}",
            osm_way_id=way_id,
            name=name,
            start_lat=coords[0][1],
            start_lon=coords[0][0],
            end_lat=coords[-1][1],
            end_lon=coords[-1][0],
            geometry_geojson=json.dumps(coords),
            length_meters=round(length, 1),
            accessibility_score=50.0,
            score_grade="U",
            score_criteria_breakdown=json.dumps({
                "assessment_status": "UNASSESSED",
                "source": "OpenStreetMap",
                "osm_way_id": way_id,
                "highway": highway
            }),
            barrier_count=0
        ))

    return segments
