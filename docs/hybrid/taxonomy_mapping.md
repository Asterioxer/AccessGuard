# AccessGuard Canonical Taxonomy & Cross-Lineage Semantic Mapping
**AccessGuard Hybrid Vision Intelligence (Mission 006)**  
**Feature Branch:** `feature/hybrid-vision-intelligence`  
**Date:** November 2026  
**Status:** COMPLETE — CANONICAL ONTOLOGY

---

## 1. Executive Overview

A central failure mode of multi-model computer vision ensembles is semantic dissonance: different research teams train models with disparate labeling definitions, granularities, and spatial conventions. For example:
- A generic obstacle detector treats a fire hydrant or light pole as a standard obstacle bounding box.
- GuideCaneOD distinguishes `pole` (utility pole) from `short_post` (bollard) and `hydrant` (fire hydrant).
- CitySurfaces segments pavement into paving materials (`concrete`, `cobblestone`), which are not obstacles but continuous friction/vibration attributes.
- RampNet identifies curb ramps as continuous heatmaps and discrete apex coordinates, rather than rectangular bounding boxes.
- Project Sidewalk Validator evaluates candidates under 5 broad categories (`CurbRamp`, `NoCurbRamp`, `Obstacle`, `SurfaceProblem`, `Crosswalk`).

To reconcile these heterogeneous perspectives without semantic corruption, AccessGuard establishes a **Canonical Accessibility Ontology**.

---

## 2. AccessGuard Canonical Accessibility Ontology

AccessGuard organizes all perception outputs into five primary ontological super-categories:

1. **`ACCESSIBILITY_TRANSITION`** (Infrastructure enabling vertical/grade transitions):
   - `curb_ramp_present`: A compliant, functional curb ramp or blended transition.
   - `curb_ramp_missing`: An abrupt curb drop-off without a transition ramp.
   - `stairs`: Exterior steps or multi-level staircases requiring an alternate ramp route.
   - `steep_incline`: Ramp or walkway slope exceeding wheelchair gradient standards (>8.33% / 1:12).

2. **`PEDESTRIAN_FACILITY`** (Dedicated pedestrian wayfinding and crossing features):
   - `crosswalk_present`: Marked zebra crossing or pedestrian street crossing corridor.
   - `tactile_paving_present`: Truncated dome or directional blister pavers for visually impaired guidance (GuideCane `blind_road`).
   - `traffic_signal_pedestrian`: Pedestrian signal indicator (`traffic_signal_red`, `traffic_signal_green`).

3. **`PATHWAY_OBSTRUCTION`** (Physical objects encroaching on minimum clear width <90cm):
   - `pole_or_bollard`: Fixed vertical elements including utility poles, light posts, bollards (`short_post`), and fire hydrants.
   - `construction_obstacle`: Construction barriers, barricades, scaffolding, traffic cones, and excavations.
   - `street_furniture_obstacle`: Park benches, waste bins (`trash_bin`), postal boxes, outdoor dining.
   - `vegetation_overgrowth`: Low-hanging branches, tree trunks, and overgrown roots lifting pavers.
   - `vehicle_encroachment`: Parked two-wheelers (`bike_moto`), motor vehicles, or carts blocking sidewalk width.
   - `narrow_sidewalk`: Geometric constriction where structural sidewalk width is less than 90cm.

4. **`SURFACE_QUALITY`** (Pavement condition and rolling resistance):
   - `surface_smooth`: High-accessibility paving (`concrete`, smooth `asphalt`).
   - `surface_moderate`: Paving with mild texture/joint vibration (`bricks`, `hexagonal` pavers).
   - `surface_rough`: High vibration, snag hazard (`cobblestone`, `granite block-stone`, unpaved `mixed`).
   - `surface_damage`: Potholes, cracks, broken pavers, lifted slabs, open drains.

5. **`VERIFICATION_STATE`** (Meta-level assessment of candidate observations):
   - `VERIFIED`: Confirmed by multiple independent models or calibrated validator ($\ge 0.85$ confidence).
   - `TENTATIVE`: Single-model detection requiring second-stage verification.
   - `CONTRADICTED`: Conflicted detections (e.g. RampNet indicates ramp, but YOLO indicates stairs).

---

## 3. Cross-Lineage Semantic Mapping Matrix

```
+---------------------------------------------------------------------------------------------------------------+
|                                    CROSS-LINEAGE TAXONOMY CONCORDANCE TABLE                                    |
+--------------------------+-----------------------+--------------------+------------------+--------------------+
| AccessGuard Canonical    | GuideCaneOD           | RampNet            | CitySurfaces     | Sidewalk Validator |
+--------------------------+-----------------------+--------------------+------------------+--------------------+
| curb_ramp_present        | --                    | curb_ramp (peak)   | --               | CurbRamp           |
| curb_ramp_missing        | --                    | --                 | --               | NoCurbRamp         |
| stairs                   | stairs (15)           | --                 | --               | Obstacle           |
| crosswalk_present        | crosswalk (5)         | --                 | --               | Crosswalk          |
| tactile_paving_present   | blind_road (4)        | --                 | --               | --                 |
| traffic_signal_red       | red_light (6)         | --                 | --               | --                 |
| traffic_signal_green     | green_light (7)       | --                 | --               | --                 |
| pole_or_bollard          | pole (8),             | --                 | --               | Obstacle           |
|                          | short_post (10),      |                    |                  |                    |
|                          | hydrant (11)          |                    |                  |                    |
| construction_obstacle    | roadblock (9),        | --                 | --               | Obstacle           |
|                          | barrier (12)          |                    |                  |                    |
| street_furniture_obstacle| trash_bin (14),       | --                 | --               | Obstacle           |
|                          | sign (16)             |                    |                  |                    |
| vehicle_encroachment     | bike_moto (2),        | --                 | --               | Obstacle           |
|                          | vehicle (3)           |                    |                  |                    |
| surface_damage           | --                    | --                 | --               | SurfaceProblem     |
| surface_smooth           | --                    | --                 | concrete (0),    | --                 |
|                          |                       |                    | asphalt (3)      |                    |
| surface_moderate         | --                    | --                 | bricks (1),      | --                 |
|                          |                       |                    | hexagonal (8)    |                    |
| surface_rough            | --                    | --                 | cobblestone (9), | SurfaceProblem     |
|                          |                       |                    | granite stone (7)| (if severe)        |
+--------------------------+-----------------------+--------------------+------------------+--------------------+
```

---

## 4. Normalization Rules & Safety Constraints

1. **Non-Equivalence Rule:** Road cracks in vehicular asphalt are NEVER mapped to pedestrian sidewalk defects unless spatial intersection confirms sidewalk geometry.
2. **Keypoint Expansion:** RampNet keypoint apexes are stored with native $(x, y)$ coordinates. When interacting with bounding-box consumers, a nominal normalized bounding box of radius $r = 0.04$ is synthesized, annotated with `"is_keypoint": True`.
3. **Surface Roughness Penalty:** CitySurfaces masks do not produce discrete obstacles unless roughness exceeds $0.65$ (e.g. extensive cobblestone or unpaved mixed terrain), in which case a `surface_damage` barrier is synthesized for routing penalty calculation.
