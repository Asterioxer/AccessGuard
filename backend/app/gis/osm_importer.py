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
