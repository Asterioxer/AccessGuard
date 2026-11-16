import json
import math
from typing import List, Dict, Tuple, Optional, Any
import networkx as nx

from app.gis.spatial_index import haversine_distance_meters
from app.schemas.routing import RouteResponse, BarrierEncountered

class AccessibilityRoutingEngine:
    """
    Wheelchair accessibility route planning engine.
    Constructs a graph from pedestrian segments, applies barrier-aware edge weighting,
    and computes optimal wheelchair routes prioritizing safety and accessibility.
    """
    def __init__(self, segments: List[Any]):
        self.segments = segments
        self.graph = nx.Graph()
        self._build_graph()

    def _round_coord(self, lat: float, lon: float, precision: int = 5) -> Tuple[float, float]:
        return (round(lat, precision), round(lon, precision))

    def _build_graph(self) -> None:
        """Builds NetworkX graph where edges are pedestrian segments."""
        for seg in self.segments:
            u = self._round_coord(seg.start_lat, seg.start_lon)
            v = self._round_coord(seg.end_lat, seg.end_lon)
            
            # Base length
            dist = seg.length_meters if seg.length_meters > 0 else haversine_distance_meters(
                seg.start_lat, seg.start_lon, seg.end_lat, seg.end_lon
            )
            
            # Weight multiplier based on accessibility score and barriers
            score = getattr(seg, "accessibility_score", 50.0)
            grade = getattr(seg, "score_grade", "U")
            
            # Check if segment has severe impassable obstacles (stairs, missing curb ramps)
            has_stairs = False
            for obs in getattr(seg, "observations", []):
                for det in getattr(obs, "detections", []):
                    if det.barrier_type == "stairs":
                        has_stairs = True
                        break

            # Freshness / staleness check (> 180 days since evaluation)
            is_stale = False
            last_eval = getattr(seg, "last_evaluated_at", None)
            if last_eval:
                try:
                    from datetime import datetime, timezone
                    now_utc = datetime.now(timezone.utc)
                    if hasattr(last_eval, "tzinfo") and last_eval.tzinfo is None:
                        last_eval = last_eval.replace(tzinfo=timezone.utc)
                    age_days = (now_utc - last_eval).days
                    if age_days > 180:
                        is_stale = True
                except Exception:
                    pass

            if has_stairs:
                cost_multiplier = 100.0 # Extreme penalty to force path around stairs
            elif grade == "A" or score >= 90.0:
                cost_multiplier = 1.0 if not is_stale else 1.3
            elif grade == "B" or score >= 75.0:
                cost_multiplier = 1.25 if not is_stale else 1.6
            elif grade == "C" or score >= 55.0:
                cost_multiplier = 2.0 if not is_stale else 2.5
            elif grade == "U":
                cost_multiplier = 3.5 # Caution penalty for unassessed routes to prioritize verified accessible paths
            elif grade == "D" or score >= 35.0:
                cost_multiplier = 4.5
            else:
                cost_multiplier = 30.0 # Grade F
                
            cost = dist * cost_multiplier
            
            # Extract line coordinates
            coords = []
            if hasattr(seg, 'geometry_geojson') and seg.geometry_geojson:
                try:
                    # GeoJSON is [[lon, lat], ...]
                    geojson_coords = json.loads(seg.geometry_geojson)
                    coords = [[c[1], c[0]] for c in geojson_coords] # convert to [[lat, lon], ...]
                except Exception:
                    coords = [[seg.start_lat, seg.start_lon], [seg.end_lat, seg.end_lon]]
            else:
                coords = [[seg.start_lat, seg.start_lon], [seg.end_lat, seg.end_lon]]
                
            self.graph.add_edge(
                u, v,
                segment_id=seg.id,
                name=seg.name,
                distance=dist,
                cost=cost,
                score=score,
                coords=coords,
                has_stairs=has_stairs,
                segment_obj=seg
            )

    def _find_nearest_node(self, lat: float, lon: float) -> Tuple[Tuple[float, float], float]:
        """Finds closest node in graph to given coordinate."""
        best_node = None
        min_dist = float('inf')
        for node in self.graph.nodes():
            dist = haversine_distance_meters(lat, lon, node[0], node[1])
            if dist < min_dist:
                min_dist = dist
                best_node = node
        return best_node, min_dist

    def find_accessible_route(
        self,
        start_lat: float,
        start_lon: float,
        end_lat: float,
        end_lon: float,
        wheelchair_profile: str = "manual_wheelchair",
        avoid_unverified: bool = True,
        algorithm: str = "dijkstra",
        include_alternatives: bool = False
    ) -> RouteResponse:
        """
        Finds the most accessible path between start and end coordinates.
        Uses barrier-aware edge costs to navigate around stairs, missing curb ramps, and severe cracks.
        """
        if len(self.graph.nodes()) < 2:
            return RouteResponse(
                success=False,
                total_distance_meters=0.0,
                estimated_duration_minutes=0.0,
                overall_accessibility_score=0.0,
                route_grade="F",
                avoids_stairs=True,
                barriers_encountered=[],
                path_coordinates=[],
                segments_traversed=[],
                routing_notes=["Network graph is empty or has insufficient segments."],
                geojson={"type": "FeatureCollection", "features": []}
            )

        start_node, d_start = self._find_nearest_node(start_lat, start_lon)
        end_node, d_end = self._find_nearest_node(end_lat, end_lon)

        MAX_SNAP_METERS = 500.0
        if not start_node or not end_node:
            return RouteResponse(
                success=False,
                total_distance_meters=0.0,
                estimated_duration_minutes=0.0,
                overall_accessibility_score=0.0,
                route_grade="F",
                avoids_stairs=True,
                barriers_encountered=[],
                path_coordinates=[],
                segments_traversed=[],
                routing_notes=["Could not locate accessible pedestrian network near endpoints."],
                geojson={"type": "FeatureCollection", "features": []}
            )

        if d_start > MAX_SNAP_METERS or d_end > MAX_SNAP_METERS:
            return RouteResponse(
                success=False,
                total_distance_meters=0.0,
                estimated_duration_minutes=0.0,
                overall_accessibility_score=0.0,
                route_grade="F",
                avoids_stairs=True,
                barriers_encountered=[],
                path_coordinates=[],
                segments_traversed=[],
                routing_notes=[
                    f"Selected location is too far from mapped pedestrian network (Start: {d_start:.0f}m, End: {d_end:.0f}m. Max allowed: {MAX_SNAP_METERS:.0f}m)."
                ],
                geojson={"type": "FeatureCollection", "features": []}
            )

        try:
            # Path search using barrier-aware accessibility cost weight
            algo_choice = algorithm.lower() if algorithm else "dijkstra"
            if algo_choice == "astar":
                def _haversine_h(u_node, v_node):
                    return haversine_distance_meters(u_node[0], u_node[1], v_node[0], v_node[1])
                node_path = nx.astar_path(self.graph, start_node, end_node, heuristic=_haversine_h, weight="cost")
            else:
                algo_choice = "dijkstra"
                node_path = nx.dijkstra_path(self.graph, start_node, end_node, weight="cost")
        except nx.NetworkXNoPath:
            return RouteResponse(
                success=False,
                total_distance_meters=0.0,
                estimated_duration_minutes=0.0,
                overall_accessibility_score=0.0,
                route_grade="F",
                avoids_stairs=True,
                barriers_encountered=[],
                path_coordinates=[],
                segments_traversed=[],
                routing_notes=["No accessible continuous pedestrian route connects these locations."],
                geojson={"type": "FeatureCollection", "features": []},
                algorithm_used=algo_choice
            )

        # Reconstruct path details
        self._temp_unassessed_count = 0
        self._temp_stale_count = 0
        total_distance = 0.0
        path_coords: List[List[float]] = []
        segments_traversed: List[str] = []
        score_weighted_sum = 0.0
        barriers_encountered: List[BarrierEncountered] = []
        avoids_stairs = True
        has_unassessed = False
        assessed_count = 0
        
        # Add initial connector if start node is slightly offset
        path_coords.append([start_lat, start_lon])
        total_distance += d_start

        for i in range(len(node_path) - 1):
            u = node_path[i]
            v = node_path[i+1]
            edge_data = self.graph.get_edge_data(u, v)
            
            dist = edge_data["distance"]
            seg_score = edge_data["score"]
            total_distance += dist
            score_weighted_sum += (seg_score * dist)
            segments_traversed.append(edge_data["segment_id"])
            
            # Append polyline points
            edge_coords = edge_data["coords"]
            # Ensure coordinates order matches direction from u to v
            if edge_coords:
                first_pt = edge_coords[0]
                dist_u_first = haversine_distance_meters(u[0], u[1], first_pt[0], first_pt[1])
                dist_v_first = haversine_distance_meters(v[0], v[1], first_pt[0], first_pt[1])
                if dist_v_first < dist_u_first:
                    edge_coords = list(reversed(edge_coords))
                for pt in edge_coords:
                    if not path_coords or path_coords[-1] != pt:
                        path_coords.append(pt)

            if edge_data["has_stairs"]:
                avoids_stairs = False

            # Check barriers on this segment
            seg_obj = edge_data.get("segment_obj")
            seg_grade = getattr(seg_obj, "score_grade", "U") if seg_obj else "U"
            if seg_grade == "U":
                has_unassessed = True
                unassessed_count = getattr(self, "_temp_unassessed_count", 0) + 1
                self._temp_unassessed_count = unassessed_count
            else:
                assessed_count += 1

            # Check if segment data is stale (> 180 days)
            last_eval = getattr(seg_obj, "last_evaluated_at", None) if seg_obj else None
            if last_eval:
                try:
                    from datetime import datetime, timezone
                    now_utc = datetime.now(timezone.utc)
                    if hasattr(last_eval, "tzinfo") and last_eval.tzinfo is None:
                        last_eval = last_eval.replace(tzinfo=timezone.utc)
                    if (now_utc - last_eval).days > 180:
                        stale_count = getattr(self, "_temp_stale_count", 0) + 1
                        self._temp_stale_count = stale_count
                except Exception:
                    pass

            if seg_obj:
                for obs in getattr(seg_obj, "observations", []):
                    for det in getattr(obs, "detections", []):
                        if det.barrier_type in ["stairs", "surface_damage", "pole_or_bollard", "curb_ramp_missing"]:
                            barriers_encountered.append(
                                BarrierEncountered(
                                    segment_id=edge_data["segment_id"],
                                    barrier_type=det.barrier_type,
                                    confidence=det.confidence,
                                    status=getattr(obs, "status", "tentative"),
                                    severity="high" if det.barrier_type in ["stairs", "curb_ramp_missing"] else "medium",
                                    latitude=obs.latitude,
                                    longitude=obs.longitude
                                )
                            )

        # Add destination connector
        if path_coords[-1] != [end_lat, end_lon]:
            path_coords.append([end_lat, end_lon])
            total_distance += d_end

        # Calculate average accessibility score
        overall_score = round(score_weighted_sum / (total_distance - d_start - d_end + 1e-6), 1)
        overall_score = max(0.0, min(100.0, overall_score))

        # Assign route letter grade
        if assessed_count == 0 and segments_traversed:
            route_grade = "U"
        elif overall_score >= 90.0:
            route_grade = "A"
        elif overall_score >= 75.0:
            route_grade = "B"
        elif overall_score >= 55.0:
            route_grade = "C"
        elif overall_score >= 35.0:
            route_grade = "D"
        else:
            route_grade = "F"

        # Estimated travel duration (manual wheelchair avg speed = 3.6 km/h = 1.0 m/s)
        duration_min = round(total_distance / 60.0, 1)

        routing_notes = [
            f"Optimized for {wheelchair_profile.replace('_', ' ')}.",
            f"Selected route avoids severe stairs and steep inclines.",
            f"Segment accessibility average: {overall_score:.1f}/100 (Grade {route_grade})."
        ]
        if has_unassessed:
            routing_notes.append("Notice: Route traverses unassessed pedestrian segments. On-the-ground accessibility is unverified for these sections.")

        # Build GeoJSON feature
        geojson_coords = [[pt[1], pt[0]] for pt in path_coords] # [[lon, lat], ...]
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": geojson_coords
                    },
                    "properties": {
                        "total_distance_meters": round(total_distance, 1),
                        "accessibility_score": overall_score,
                        "grade": route_grade,
                        "avoids_stairs": avoids_stairs
                    }
                }
            ]
        }

        # Optional alternative direct shortest route comparison
        direct_comparison = None
        if include_alternatives:
            try:
                direct_path = nx.dijkstra_path(self.graph, start_node, end_node, weight="distance")
                direct_dist = d_start + d_end
                direct_stairs_count = 0
                for j in range(len(direct_path) - 1):
                    e_data = self.graph.get_edge_data(direct_path[j], direct_path[j+1])
                    direct_dist += e_data["distance"]
                    if e_data.get("has_stairs"):
                        direct_stairs_count += 1
                tradeoff_dist = round(total_distance - direct_dist, 1)
                direct_comparison = {
                    "direct_distance_meters": round(direct_dist, 1),
                    "accessible_detour_meters": max(0.0, tradeoff_dist),
                    "direct_route_has_stairs": direct_stairs_count > 0,
                    "stairs_avoided_by_accessible_route": direct_stairs_count
                }
                if direct_stairs_count > 0:
                    routing_notes.append(
                        f"Accessibility tradeoff: Accessible route is {max(0.0, tradeoff_dist):.0f}m longer, avoiding {direct_stairs_count} staircase(s) present on the direct shortest path."
                    )
            except Exception:
                pass

        return RouteResponse(
            success=True,
            total_distance_meters=round(total_distance, 1),
            estimated_duration_minutes=duration_min,
            overall_accessibility_score=overall_score,
            route_grade=route_grade,
            avoids_stairs=avoids_stairs,
            barriers_encountered=barriers_encountered,
            path_coordinates=path_coords,
            segments_traversed=segments_traversed,
            routing_notes=routing_notes,
            geojson=geojson,
            algorithm_used=algo_choice,
            direct_route_comparison=direct_comparison,
            has_unassessed_segments=has_unassessed,
            unassessed_segments_count=getattr(self, "_temp_unassessed_count", 0),
            stale_segments_count=getattr(self, "_temp_stale_count", 0)
        )
