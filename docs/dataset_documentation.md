# AccessGuard — Public Dataset & Research Expansion Guide

## 1. Relevant Open-Source Accessibility & Urban Datasets

To train deep learning models (`TorchvisionAdapter` and `YOLOAdapter`) for production deployment across global pedestrian infrastructure, AccessGuard integrates with and evaluates the following open datasets:

| Dataset | Primary Focus | License | Annotation Format | Geographic Coverage | Known Biases & Domain Gaps |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Project Sidewalk** (UW / Makeability Lab) | Wheelchair barriers (stairs, missing ramps, surface problems, obstacles) | CC-BY 4.0 | Polygons & point labels with severity ratings | US cities (Seattle, DC, Chicago, Columbus), Mexico, New Zealand | High coverage of North American standard curb cuts; under-represents South Asian / Indian informal infrastructure. |
| **Mapillary Vistas** | Fine-grained street scenes (curbs, sidewalks, crosswalks, potholes, utility poles) | CC-BY-NC-SA 4.0 (Research) / Commercial | Pixel-level semantic & instance segmentation | Global (6 continents, diverse lighting/weather) | Street-oriented camera viewpoints; pedestrian perspective sidewalk detail varies. |
| **Cityscapes** | Urban street scenes (sidewalks, pedestrians, roads, poles) | Custom Research Non-Commercial | Pixel-level semantic segmentation | 50 German & European cities | High-income European urban geometry; pristine paving, few severe potholes or unpaved paths. |
| **CrowdCurbs / RDD2022** (Road Damage Dataset) | Surface cracks, potholes, alligator cracking, road wear | CC-BY-SA 4.0 | Bounding boxes (VOC/YOLO) | India, Japan, Czech Republic, China, USA | Focuses primarily on motor vehicle roadways rather than pedestrian footpaths. |

---

## 2. Dataset Preparation & CVAT Annotation Pipeline

AccessGuard provides a standard data preparation pipeline to ingest and annotate field crowdsourced imagery:

```
Raw Field Imagery -> EXIF GPS Extraction -> Face & License Blur -> Quality Filter (Blur/Fog) -> CVAT Ingestion -> VOC/COCO Export -> Train/Val/Test Split
```

### Data Pipeline Principles:
1. **Zero Facial Identity Retention:** All images pass through `apply_privacy_redaction()` before annotation upload.
2. **Standard Label Schema (COCO JSON / Pascal VOC):**
   - `stairs`: Physical steps, stairwells, elevated stoops without ramp.
   - `curb_ramp_present`: Sloped transitions with or without truncated domes (tactile paving).
   - `curb_ramp_missing`: Unramped vertical curb drops $>5\text{cm}$ at street crossings.
   - `surface_damage`: Cracks $>2\text{cm}$ wide, heaved slabs, potholes, loose paver blocks.
   - `pole_or_bollard`: Bollards, utility poles, signposts obstructing $\ge 50\%$ of sidewalk width.
   - `construction_obstacle`: Scaffolding, debris, building material blocking path.

---

## 3. Active Learning & Uncertainty-Driven Annotation

Rather than labeling thousands of redundant clean sidewalk images, AccessGuard employs **Active Learning**:

1. **Uncertainty Query Strategy:**
   Observations where `is_uncertain == True` ($0.45 \le \text{confidence} < 0.70$ or $\text{uncertainty\_score} > 0.35$) are prioritized in the `/verifications/queue`.
2. **Community Consensus Triangulation:**
   When community reviewers dispute an AI detection or reach conflicting votes ($2 \text{ confirm}, 2 \text{ dispute}$), the sample is flagged for expert re-annotation in CVAT.
3. **Continuous Retraining Loop:**
   Validated edge cases are added to the validation benchmark to prevent regressions.
