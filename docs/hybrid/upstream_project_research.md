# Upstream Open-Source Research Analysis & Due Diligence
**AccessGuard Hybrid Vision Intelligence (Mission 006)**  
**Feature Branch:** `feature/hybrid-vision-intelligence`  
**Date:** November 2026  
**Status:** COMPLETE — VERIFIED AUDIT

---

## 1. Executive Summary

To elevate AccessGuard from a single-model pedestrian barrier detection system into a resilient, multi-expert hybrid intelligence platform, we conducted an exhaustive empirical due diligence and architectural assessment of seven candidate open-source upstream projects across four primary lineages:

1. **Project Sidewalk / RampNet** (`ProjectSidewalk/RampNet`)
2. **Project Sidewalk Validator & AI API** (`ProjectSidewalk/sidewalk-validator-ai`, `ProjectSidewalk/sidewalk-ai-api`, `ProjectSidewalk/SidewalkWebpage`)
3. **GuideCaneOD & GuideCaneOD-YOLO** (`yolo0731/GuideCaneOD`, `yolo0731/GuideCaneOD-YOLO`)
4. **CitySurfaces** (`VIDA-NYU/city-surfaces`)

Each upstream repository was acquired into an isolated research workspace (`AccessGuard-Research/upstream/`) to isolate foreign dependencies, prevent licensing contamination, and inspect models, evaluation harnesses, datasets, and inference pipelines.

```
+----------------------------------------------------------------------------------------------------+
|                                    ACCESSGUARD RESEARCH EVALUATION                                 |
+-----------------------------------+--------------------+--------------------+----------------------+
| Upstream Repository               | Primary License    | Model Architecture | Compatibility Status |
+-----------------------------------+--------------------+--------------------+----------------------+
| ProjectSidewalk/RampNet           | MIT                | ConvNeXt-V2 Base   | APPROVED (PROTOTYPE) |
| ProjectSidewalk/sidewalk-validator| MIT                | DINOv2-Large       | APPROVED (PROTOTYPE) |
| ProjectSidewalk/sidewalk-ai-api   | MIT                | Flask + DINOv2     | APPROVED (REFERENCE) |
| ProjectSidewalk/SidewalkWebpage   | MIT                | Scala / Play / PG  | APPROVED (REFERENCE) |
| yolo0731/GuideCaneOD              | CC BY-NC 4.0       | Dataset / YAML     | APPROVED (RESTRICTED)|
| yolo0731/GuideCaneOD-YOLO         | AGPL v3            | YOLOv8/v11 Train   | REJECTED (CODE COPY) |
| VIDA-NYU/city-surfaces            | BSD 3-Clause       | HRNet / OCRNet     | APPROVED (PROTOTYPE) |
+-----------------------------------+--------------------+--------------------+----------------------+
```

---

## 2. Lineage I: Project Sidewalk — RampNet

### 2.1 Overview & Academic Lineage
- **Origin:** Makeability Lab, University of Washington (O'Meara et al., ICCV 2025 Workshop on Vision Foundation Models and Generative AI for Accessibility).
- **Core Problem Addressed:** Scarcity of annotated curb ramp datasets in street-level imagery. Stage 1 bootstraps 210,000+ annotations from municipal GIS open data mapped to Google Street View panoramas; Stage 2 trains a high-capacity curb ramp heatmap detector.
- **Repository Commit:** `8d8d8eef58ad631b143c2a8e4d7224c2fe1221a5` (MIT License).

### 2.2 Model Architecture & Inference Pipeline
- **Backbone:** ConvNeXt-V2 Base (`convnextv2_base.fcmae_ft_in22k_in1k_384`).
- **Input Geometry:** Full equirectangular $360^\circ$ panorama resized to $2048 \times 4096$ with standard ImageNet normalization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).
- **Output:** Continuous heatmap of shape $512 \times 1024$ (nominal resolution; effective feature resolution is $64 \times 128$ due to stride-32 backbone followed by linear $1 \times 1$ head).
- **Decoding Mechanism:** Local peak extraction using `skimage.feature.peak_local_max(np.clip(heatmap, 0, 1), min_distance=10, threshold_abs=0.50)` scaled to original image dimensions.
- **Output Representation:** **Point Keypoints** $(x, y)$, NOT bounding boxes or polygon masks.

### 2.3 Critical Findings & Evaluation Errata
1. **ICCV 2025 Workshop Erratum (§3.3):**
   - The original paper implementation (`v1.0-iccv2025`) allowed a single prediction within the matching radius of two adjacent ground-truth curb ramps (e.g. dual-ramp street corners) to count as two True Positives. Redundant detections were also ignored rather than scored as False Positives.
   - **Correction (`v1.1-corrected-eval`):** Standard one-to-one Hungarian / greedy matching with duplicates penalised as False Positives shifts gold-set precision from 0.938 to 0.949, recall from 0.935 to 0.873, and AP from 0.9236 to 0.9205.
   - **AccessGuard Mandate:** Any integration or benchmarking of RampNet within AccessGuard MUST enforce greedy one-to-one matching (`v1.1-corrected-eval` rules).
2. **The $360^\circ$ Seam Artifact:**
   - Equirectangular panoramas wrap at the boundary ($x=0$ and $x=W$). Because `peak_local_max` is executed in planar 2D Cartesian coordinates, a ramp spanning the wrap boundary produces dual peaks on both margins. In the published dataset, 3.7% of panoramas carry seam duplicate labels.
3. **Perspective Distortion & Camera Geometry:**
   - RampNet is trained exclusively on equirectangular panoramas. Passing raw perspective camera frames (e.g., standard smartphone photos or vehicle dashcams) directly into RampNet leads to severe aspect ratio deformation and elevated false positive rates unless cropped/projected into spherical coordinates or evaluated on genuine panoramic imagery.

---

## 3. Lineage II: Project Sidewalk — Validator AI & AI API

### 3.1 Overview & System Function
- **Origin:** University of Washington Makeability Lab.
- **Repository Commits:**
  - `sidewalk-validator-ai`: `750b071100c5b725a091e813150a12c83f60ab30` (MIT License).
  - `sidewalk-ai-api`: `b834896630d14ab14a6a0db754f49278823b2ea0` (MIT License).
  - `SidewalkWebpage`: `main` branch (MIT License).
- **Core Function:** Project Sidewalk Validator is **NOT** an object detector. It is an automated second-stage verification engine that assesses human- or model-suggested candidate labels (`CurbRamp`, `NoCurbRamp`, `Obstacle`, `SurfaceProblem`, `Crosswalk`).

### 3.2 Model Architecture & Crop Preprocessing
- **Backbone:** Fine-tuned `DINOv2-Large` classifier.
- **Inputs:** Localized image crops generated from candidate label coordinates. The pipeline uses depth estimation (`Depth-Anything-V2`) to adjust crop window sizing dynamically according to object distance, and applies pitch/roll tilt correction to normalize viewing angles.
- **Outputs:** Binary classification verdict: `correct` vs `incorrect`, with uncalibrated logits converted to calibrated precision estimates via empirical calibration tables (`accuracy_mappings/{label_type}.json`).
- **Performance:** Round 3 models achieve macro-AUC of 0.915 (CurbRamp 0.953, Crosswalk 0.972, NoCurbRamp 0.920, Obstacle 0.873, SurfaceProblem 0.855).

### 3.3 AccessGuard Role & Integration Strategy
- **Architectural Role:** Secondary Verification & Confidence Calibration.
- When an object detector (such as AccessGuard's YOLO or Torchvision adapter) identifies a tentative barrier or curb ramp, the crop is passed to the Sidewalk Validator model. If the calibrated probability of correctness is $\ge 0.85$, the observation is promoted to `VERIFIED`; if low, it is marked `TENTATIVE_LOW_CONFIDENCE` or flagged for human review.

---

## 4. Lineage III: GuideCaneOD & GuideCaneOD-YOLO

### 4.1 Overview & Scope
- **Author:** yolo0731 (2025).
- **Repositories:**
  - `yolo0731/GuideCaneOD`: Dataset definitions, 17-class taxonomy, COCO/YOLO annotations, label guidelines (`1dbcdcae6dc421088474d0adfda7b593e265ffa4`).
  - `yolo0731/GuideCaneOD-YOLO`: Ultralytics training/evaluation harness, patch scripts (`17dade86115cb7a0788446739c36cc8d6dd05978`).
- **Core Purpose:** Real-time obstacle detection for visually impaired navigation using intelligent white canes (GuideCane).

### 4.2 17-Class Taxonomy Analysis
GuideCaneOD defines 17 classes specifically curated for blind/low-vision pedestrian navigation:
1. `person` (Pedestrian collision hazard)
2. `dog` (Moving obstacle / assistance animal)
3. `bike_moto` (Two-wheeler sidewalk obstruction)
4. `vehicle` (Motor vehicle encroachment)
5. `blind_road` (Tactile paving / warning / direction indicator)
6. `crosswalk` (Pedestrian zebra crossing)
7. `red_light` (Traffic signal state)
8. `green_light` (Traffic signal state)
9. `pole` (Utility/lighting pole on walkway)
10. `roadblock` (Construction barrier / traffic cone / blockade)
11. `short_post` (Bollard / parking post blocking wheelchair passage)
12. `hydrant` (Fire hydrant sidewalk obstruction)
13. `barrier` (Guardrail / fence / temporary railing)
14. `tree` (Tree trunk / overgrown roots lifting pavement)
15. `trash_bin` (Waste bin blocking pedestrian pathway)
16. `stairs` (Curb steps / staircases requiring ramp alternative)
17. `sign` (Low-hanging or ground-mounted traffic signage)

### 4.3 Strengths & Limitations
- **Strengths:** Unlike standard COCO (which has 80 general object classes, missing tactile paving and bollards), GuideCaneOD targets specific micro-barriers crucial to wheelchair users and visually impaired pedestrians (`short_post`, `blind_road`, `stairs`, `roadblock`).
- **Limitations:**
  - Does not segment surface damage, potholes, or unpaved terrain.
  - Does not distinguish compliant curb ramps from missing curb ramps (only detects `stairs` as a negative barrier).

### 4.4 Legal & Licensing Analysis
- **`GuideCaneOD` Dataset:** Released under **Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0)**. Permitted for non-commercial research and validation only.
- **`GuideCaneOD-YOLO` Software:** Released under **GNU Affero General Public License v3 (AGPL v3)**.
- **Contamination Risk:** Incorporating any source code from `GuideCaneOD-YOLO` into AccessGuard (MIT License) would violate license compatibility and trigger copyleft virality.
- **Remediation:** **REJECT CODE COPYING.** AccessGuard implements an independent, clean-room adapter using native Ultralytics YOLO APIs that consumes model weights or runs via decoupled execution, completely isolated from upstream AGPL code.

---

## 5. Lineage IV: CitySurfaces

### 5.1 Overview & Scope
- **Origin:** VIDA Lab, New York University (Hosseini et al., *Sustainable Cities and Society*, 2022).
- **Repository Commit:** `c57c823a84ae7158dfe26b97d1f4d4dda47b55f0` (BSD 3-Clause License).
- **Core Function:** City-scale high-resolution semantic segmentation of sidewalk paving materials from street-level perspective imagery.

### 5.2 Model Architecture & Paving Taxonomy
- **Architecture:** Hierarchical multi-scale attention combined with Object-Contextual Representations (OCRNet / HRNet-W48) based on NVIDIA Semantic Segmentation.
- **Input Geometry:** High-resolution perspective imagery ($1024 \times 2048$ or native street view crops).
- **Output:** Dense 2D semantic mask with 10 classes:
  - 8 Sidewalk Materials: `concrete` (0), `bricks` (1), `granite` (2), `asphalt` (3), `mixed` (4), `granite block-stone` (7), `hexagonal` (8), `cobblestone` (9).
  - 2 Non-Sidewalk Classes: `road` (5), `background` (6).

### 5.3 Semantic Taxonomy Alignment
- **Paving Materials Are Not Barriers:** In AccessGuard's ontology, cobblestone or bricks are not discrete obstacle bounding boxes; they represent continuous surface properties.
- **Mapping to Accessibility Metrics:**
  - `concrete`, `asphalt`: Smooth surface $\to$ High rolling index, wheelchair accessible (Roughness Index = 1.0).
  - `bricks`, `hexagonal`: Moderate vibration $\to$ Acceptable with caution (Roughness Index = 0.75).
  - `cobblestone`, `granite block-stone`, `mixed`: High vibration, wheel snag hazard $\to$ Poor accessibility (Roughness Index = 0.35).
- **Hardware Profile:** HRNet-W48 requires substantial GPU memory during inference (~2.8GB VRAM in FP32). On a 4GB GPU (RTX 3050 Laptop), it cannot execute concurrently with ConvNeXt-V2 or DINOv2-Large without memory eviction.

---

## 6. Synthesis: The AccessGuard Hybrid Vision Architecture

By synthesizing the distinct strengths of each evaluated upstream project, AccessGuard achieves an end-to-end, multi-expert accessibility perception pipeline:

```
+-------------------------------------------------------------------------------------------------------+
|                                    INPUT STREET-LEVEL IMAGE / PANORAMA                                |
+---------------------------------------------------+---------------------------------------------------+
                                                    |
                         +--------------------------+--------------------------+
                         |                                                     |
             [Perspective Camera Crop]                                [Equirectangular 360° Pano]
                         |                                                     |
                         v                                                     v
          +-------------------------------+                             +-------------------+
          |     PRIMARY DETECTOR POOL     |                             |   RAMPNET ADAPTER |
          |   - AccessGuard YOLO (COCO)   |                             |   ConvNeXt-V2     |
          |   - GuideCane Adapter (17 cls)|                             |   Heatmap Peaks   |
          +---------------+---------------+                             +---------+---------+
                          |                                                       |
                          |  Bounding Boxes                                       | Keypoints
                          v                                                       v
          +---------------------------------------------------------------------------------+
          |                               EVIDENCE FUSION ENGINE                            |
          |   1. Coordinate Frame Normalization (Spherical <-> Perspective Projection)      |
          |   2. Spatial Association & Clustering (IoU / Geodesic Distance Gating)          |
          |   3. Cross-Model Disagreement Detection (Consensus vs Contradiction)            |
          |   4. Non-Naive Confidence Calibration & Decay                                   |
          +---------------------------------------+-----------------------------------------+
                                                  |
                                                  v Candidate Obstacles / Ramps
          +---------------------------------------------------------------------------------+
          |                           SECONDARY VERIFICATION POOL                           |
          |   - Sidewalk Validator Adapter (DINOv2-Large Crop Classifier)                   |
          |   - CitySurfaces Adapter (HRNet Sidewalk Material Segmentation)                 |
          +---------------------------------------+-----------------------------------------+
                                                  |
                                                  v Fused Accessibility Evidence
          +---------------------------------------------------------------------------------+
          |                POSTGIS GEOSPATIAL REGISTRATION & BARRIER-AWARE ROUTING          |
          +---------------------------------------------------------------------------------+
```

### Key Architectural Tenets
1. **Separation of Concerns:** Detectors generate candidates; segmenters assess surface texture; validators calibrate correctness; fusion combines evidence.
2. **Failure Transparency:** Every adapter reports explicit lifecycle status (`READY`, `NOT_LOADED`, `INFERENCE_FAILED`, `DEGRADED`). Inference failure NEVER emits empty detections.
3. **Strict Memory Budgeting:** Supports `RESOURCE_CONSTRAINED` mode with sequential execution and CPU fallback for 4GB VRAM environments.
