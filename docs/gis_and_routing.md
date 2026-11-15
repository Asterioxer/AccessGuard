# AccessGuard — GIS and Routing Intelligence Documentation

## 1. Geospatial Architecture & Coordinate Reference Systems (CRS)

AccessGuard standardizes on standard Open Geospatial Consortium (OGC) reference frameworks:

| Context | CRS / Datum | Representation | Rationale |
| :--- | :--- | :--- | :--- |
| **Database Storage (PostGIS / SQLite)** | **EPSG:4326 (WGS 84)** | Geographic decimal degrees `(latitude, longitude)` | Canonical global coordinates returned by consumer GPS receivers and EXIF tags. |
| **REST API & GeoJSON Exports** | **RFC 7946 / EPSG:4326** | `[longitude, latitude]` | Strict compliance with the official GeoJSON specification. |
| **Map Rendering (Leaflet)** | **EPSG:3857 (Web Mercator)** | Projected planar coordinates | Standard projection for tiled raster and vector basemaps (CartoDB, OpenStreetMap). |
| **Metric Spatial Calculations** | **Haversine Great-Circle** | Meters ($m$) | Accurate spherical geodesic distance computation between geographic coordinates without projection distortion. |

---

## 2. Spatial Indexing & Road Segment Snapping

Observations uploaded with GPS coordinates must be associated with the nearest pedestrian segment:

1. **Snapping Tolerance:** The spatial indexing engine ([backend/app/gis/spatial_index.py](file:///c:/Users/soham/Desktop/AccessGuard/backend/app/gis/spatial_index.py)) enforces a `SNAP_TOLERANCE_METERS = 25.0m` search radius. Points farther than $25\text{m}$ from any sidewalk remain unassociated or trigger a new segment boundary.
2. **Endpoint Maximum Distance Guard:** In [backend/app/gis/routing_engine.py](file:///c:/Users/soham/Desktop/AccessGuard/backend/app/gis/routing_engine.py), origin and destination coordinates must lie within `MAX_SNAP_METERS = 500.0m` of the mapped pedestrian network. Points beyond this threshold are rejected with explicit feedback.

---

## 3. NetworkX Accessibility Graph

The pedestrian network is represented as an undirected graph $G = (V, E)$:
- **Vertices ($V$):** Intersection nodes rounded to 5 decimal places ($\approx 1.1\text{m}$ resolution).
- **Edges ($E$):** Pedestrian sidewalk segments containing geometric length, accessibility score ($0.0 - 100.0$), barrier count, and hazard flags.

### Edge Cost Formula
Edge traversal cost is determined non-linearly based on verified physical barriers:

$$\text{Cost}(e) = \text{Length}(e) \times M(e)$$

Where $M(e)$ is the barrier cost multiplier:
- **Impassable Stairs Detected:** $M(e) = 100.0$ (Forces route search around the obstacle).
- **Grade A ($\ge 90.0$):** $M(e) = 1.0$ (Unimpeded baseline).
- **Grade B ($75.0 - 89.9$):** $M(e) = 1.25$ (Minor slope or surface irregularity).
- **Grade C ($55.0 - 74.9$):** $M(e) = 2.0$ (Noticeable roughness or moderate obstacle).
- **Grade U (Unassessed, 0 Observations):** $M(e) = 2.5$ (Caution multiplier prioritizing surveyed accessible corridors).
- **Grade D ($35.0 - 54.9$):** $M(e) = 4.5$ (Significant barriers; used only if no alternative exists).
- **Grade F ($< 35.0$):** $M(e) = 30.0$ (Severe obstacles; avoided whenever possible).

---

## 4. Pathfinding Algorithms: Dijkstra vs A*

AccessGuard supports interchangeable pathfinding algorithms selected via `RouteRequest.algorithm`:

### Dijkstra's Algorithm (`algorithm="dijkstra"`)
- **Characteristics:** Exhaustive, uniform-cost radial expansion.
- **Guarantee:** Globally optimal path minimizing total barrier-weighted cost.
- **Complexity:** $O(|E| + |V| \log |V|)$.

### A* Search (`algorithm="astar"`)
- **Characteristics:** Goal-directed search guided by an admissible heuristic function $h(n)$.
- **Heuristic:** Great-circle Haversine distance between current node $n$ and destination node $t$:
  $$h(n) = \text{Haversine}(n_{\text{lat}}, n_{\text{lon}}, t_{\text{lat}}, t_{\text{lon}})$$
  Since $M(e) \ge 1.0$, $\text{Haversine}(n, t) \le \text{Cost}^*(n, t)$, satisfying admissibility ($h(n) \le h^*(n)$).
- **Advantage:** Faster convergence and lower memory consumption on large metropolitan pedestrian graphs.

---

## 5. Alternative Route & Accessibility Tradeoff Analysis

When `include_alternatives=True` is requested, the engine computes:
1. **Primary Accessible Route:** Path minimizing barrier cost ($M(e) \ge 1.0$).
2. **Direct Shortest Route:** Path minimizing unweighted physical distance ($M(e) = 1.0$).

The response includes `direct_route_comparison`:
```json
{
  "direct_distance_meters": 412.5,
  "accessible_detour_meters": 87.5,
  "direct_route_has_stairs": true,
  "stairs_avoided_by_accessible_route": 2
}
```
Users and navigation aids receive explicit evidence: *"Accessible route is 88m longer, avoiding 2 staircases present on the direct shortest path."*
