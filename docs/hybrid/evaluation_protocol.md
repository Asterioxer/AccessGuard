# Hybrid Vision Evaluation & Benchmarking Protocol
**AccessGuard Hybrid Vision Intelligence (Mission 006)**  
**Feature Branch:** `feature/hybrid-vision-intelligence`  
**Date:** November 2026  
**Status:** COMPLETE — RIGOROUS EVALUATION METHODOLOGY

---

## 1. Overview & Evaluation Principles

To guarantee that reported performance metrics reflect actual pedestrian infrastructure safety rather than synthetic overfitting, all model evaluations in AccessGuard adhere to four strict principles:
1. **Unfabricated Evidence:** Every metric must be computed directly by executing automated verification scripts against physical image arrays. No numbers may be estimated or manually inserted.
2. **Standard One-to-One Matching:** Predictions and ground-truth annotations are matched using greedy one-to-one Hungarian-style matching. Duplicate detections claiming the same barrier are penalized as False Positives (incorporating the RampNet ICCV'25 erratum).
3. **Failure Distinction:** Uninitialized models, CUDA memory errors, or adapter crashes are explicitly tracked as `execution_errors`. They must never be converted to empty detection lists or counted as successful zero-barrier classifications.
4. **Representative Environmental Slicing:** Models are evaluated across realistic pedestrian challenge conditions characteristic of Indian urban infrastructure (broken pavers, open trenches, dynamic obstacles, monsoon puddles, low illumination).

---

## 2. Benchmark Dataset: Indian Streetscape Suite (25 Scenes)

The primary evaluation suite comprises 25 high-resolution streetscape scenes (`data/benchmarks/indian_streetscapes/`) representing real-world pedestrian environments:
- **Spatial Geometry:** Mix of standard perspective camera views and panoramic angles.
- **Annotated Ground Truths:** 85+ total ground-truth instances spanning:
  - `curb_ramp_present` / `curb_ramp_missing`
  - `stairs` (steep steps, uneven risers)
  - `pole_or_bollard` (utility poles, short bollards, hydrants)
  - `construction_obstacle` (barricades, construction debris, traffic cones)
  - `surface_damage` (broken flagstones, potholes, cobblestone roughness)
  - `crosswalk_present`
- **Environmental Slices:**
  - **Illumination:** Daylight, Twilight, Night / Low Light ($<30\text{ lux}$).
  - **Weather:** Clear, Rain / Water-logged, Fog / Low Contrast.
  - **Camera Angle:** Eye-level pedestrian, Vehicle dashcam, Low-angle wheelchair perspective.

---

## 3. Mathematical Metric Formulations

### 3.1 Precision, Recall & F1-Score
Given True Positives ($TP$), False Positives ($FP$), and False Negatives ($FN$) at IoU threshold $\theta = 0.50$:
$$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}$$
$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

### 3.2 Mean Average Precision (mAP@0.50)
Interpolated average precision across 11 recall points for each class $c$:
$$\text{AP}_c = \frac{1}{11} \sum_{r \in \{0, 0.1, \dots, 1.0\}} \max_{\tilde{r} \ge r} p(\tilde{r})$$
$$\text{mAP} = \frac{1}{C} \sum_{c=1}^C \text{AP}_c$$

### 3.3 Expected Calibration Error (ECE)
Predictions are grouped into $M = 10$ confidence bins $B_m \subset (0, 1]$.
$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

---

## 4. Benchmark Models & Experimental Configurations

The benchmarking suite evaluates eight distinct operational configurations:

1. **`M1-HEURISTIC`:** Rule-based computer vision (Canny edges, color segmentation, Hough lines).
2. **`M2-TORCHVISION`:** Faster R-CNN with ResNet-50-FPN backbone.
3. **`M3-YOLO`:** Ultralytics YOLOv8n baseline model.
4. **`M4-RAMPNET`:** Project Sidewalk ConvNeXt-V2 curb ramp heatmap detector.
5. **`M5-GUIDECANE`:** GuideCane 17-class micro-barrier detector.
6. **`M6-CITYSURFACES`:** High-resolution surface material and roughness segmentation.
7. **`M7-FUSED-HYBRID`:** Multi-Model Evidence Fusion Engine combining M3 + M4 + M5 + M6.
8. **`M8-FULL-HYBRID`:** Complete Hybrid Architecture (M7 + Sidewalk Validator DINOv2 verification).

---

## 5. Ablation Studies Protocol

Four controlled ablation experiments are executed to isolate individual architectural components:
- **Ablation 1 (Ensemble Synergy):** Full Hybrid vs Single Best Model (YOLO or GuideCane).
- **Ablation 2 (Validator Impact):** Full Hybrid with Validator vs Full Hybrid without Validator (measuring false positive suppression and precision gain).
- **Ablation 3 (Seam Suppression):** RampNet with seam suppression vs RampNet without seam suppression on panoramic boundaries.
- **Ablation 4 (Hardware Profiles):** `FULL_HYBRID` (concurrent) vs `RESOURCE_CONSTRAINED` (sequential offloading) measuring peak VRAM usage and inference latency on the RTX 3050.
