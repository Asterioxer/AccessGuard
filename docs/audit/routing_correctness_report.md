# AccessGuard Routing & GIS Correctness Report
**Author**: Independent GIS & Spatial Routing Engineer  
**Audit Target**: `backend/app/gis/`  
**Execution Timestamp**: November 2026

---

## 1. Executive Summary

The accessibility routing engine was subjected to graph-theoretic edge case testing using `scripts/audit/audit_routing_gis.py`. The routing engine correctly penalizes unramped stairs (cost multiplier $100\times$) and selects ramped bypasses when alternatives exist. However, the audit uncovered **two high-severity spatial defects**:
1. **Silent Grade A Assumption for Unsurveyed Sidewalks**: Segments with zero observations default to an accessibility score of $100.0$ (Grade A), violating core pedestrian safety ethics.
2. **Missing Endpoint Snap Boundary**: Queries with origin or destination coordinates thousands of kilometers away from the pedestrian network silently snap to the nearest node across continents.

---

## 2. Graph Connectivity & Inaccessible Network Stress Tests

### 2.1. Disconnected Graph Test
- **Scenario**: Two disconnected pedestrian clusters separated by $2\text{ km}$ with no interconnecting edge.
- **Result**: **PASS**. `nx.dijkstra_path` raised `NetworkXNoPath`, which the engine caught and converted into a graceful `RouteResponse(success=False, routing_notes=["No accessible continuous pedestrian route connects these locations."])`.

### 2.2. Inaccessible Network (All Paths Blocked by Stairs)
- **Scenario**: Origin and destination connected only by a segment confirmed to have 5 unramped stairs.
- **Result**: **PASS WITH APPROPRIATE WARNING**.
  - `success`: `True` (identifies physical connectivity).
  - `avoids_stairs`: `False`.
  - `route_grade`: `F`.
  - `barriers_encountered`: Itemizes the 5-step concrete stair obstacle with coordinates.
  - `routing_notes`: Flags that the route includes unavoidable severe obstacles.

---

## 3. Critical Defect Analysis

### 3.1. Defect GIS-01: Silent Grade A Classification for Unsurveyed Segments
- **Severity**: **HIGH (Accessibility Risk)**
- **Location**: `backend/app/db/models.py:42`, `backend/app/gis/scoring.py:27`
- **Code Trace**:
  ```python
  # In PedestrianSegment model:
  accessibility_score = Column(Float, default=100.0)
  score_grade = Column(String, default="A")

  # In scoring.py:
  base_score = 100.0
  # If observations list is empty, penalties = 0, bonuses = 0 -> net_score = 100.0 (Grade A)
```
- **Auditor Verdict**:
  When OpenStreetMap data or municipal sidewalk geometries are imported, segments with zero field observations are assigned an accessibility score of $100.0$ (Grade A). This produces a **catastrophic false sense of safety**: a wheelchair user reviewing the map will see a green Grade A corridor, only to encounter an unmapped $40\text{cm}$ curb drop or open drain.
- **Required Remediation**:
  1. Add an explicit `evaluation_status` field: `"unassessed"`, `"tentative"`, `"assessed"`.
  2. For segments with zero observations, assign `score_grade = "U"` (Unassessed / Grey), with `accessibility_score = None` or a conservative baseline (e.g. $50.0$), displaying clear UI indicators that the segment has not been verified.

---

### 3.2. Defect GIS-02: Missing Snap Radius Threshold on Routing Endpoints
- **Severity**: **MEDIUM**
- **Location**: `backend/app/gis/routing_engine.py:90-101`
- **Code Trace**:
  ```python
  def _find_nearest_node(
      self, lat: float, lon: float
  ) -> Tuple[Tuple[float, float], float]:
    best_node = None
    min_dist = float("inf")
    for node in self.graph.nodes():
      dist = haversine_distance_meters(lat, lon, node[0], node[1])
      if dist < min_dist:
        min_dist = dist
        best_node = node
    return best_node, min_dist
```
- **Auditor Verdict**:
  In `scripts/audit/audit_routing_gis.py`, coordinates for Tokyo, Japan (`35.6762, 139.6503`) were passed as the origin against a San Francisco network. `_find_nearest_node` happily snapped Tokyo to the SF network, reporting a walking distance of $8,274,748\text{ meters}$ ($8,274\text{ km}$)!
- **Required Remediation**:
  Enforce `MAX_SNAP_DISTANCE_METERS = 500.0`. If $d_{\text{start}} > 500\text{m}$ or $d_{\text{end}} > 500\text{m}$, immediately return `success=False` with error message `"Origin/Destination is too far from the mapped pedestrian network (> 500m)."`.

---

## 4. Barrier-Aware Alternative Path Verification

The audit verified that the non-linear cost multipliers successfully redirect routes when safe alternatives exist:
- Direct path with stairs: Length $200\text{m}$, Cost $= 200 \times 100.0 = 20,000$.
- Ramp bypass path: Length $240\text{m}$, Cost $= 240 \times 1.0 = 240$.
- **Outcome**: The A* routing engine selected the ramp bypass with zero stairs, validating the core algorithmic objective.
