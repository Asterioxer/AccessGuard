# Model Card: AccessGuard Heuristic Computer Vision Adapter

## Model Details
- **Model Name:** AccessGuard Classical Heuristic CV Adapter (`HeuristicCVAdapter`)
- **Version:** 1.1.0
- **Model Type:** Deterministic Classical Computer Vision (OpenCV)
- **License:** MIT License
- **Frameworks:** OpenCV 4.10, NumPy
- **Hardware Targets:** Any standard CPU (x86_64, ARM64), zero GPU/CUDA dependency.

---

## Intended Use & Capabilities
- **Primary Use Case:** Automated detection of wheelchair accessibility barriers in street-level imagery and sampled keyframes from pedestrian videos.
- **Recognized Barrier Categories:**
  1. `stairs`: Multi-step staircases, curb steps, unramped changes in elevation.
  2. `curb_ramp_present`: Detects presence of pedestrian curb cuts and flared transition ramps.
  3. `curb_ramp_missing`: Abrupt unramped roadway curb drop-offs.
  4. `surface_damage`: Fractures, potholes, buckling, severe sidewalk cracking.
  5. `pole_or_bollard`: Vertical poles, street bollards, central walkway obstructions.
- **Out-of-Scope:**
  - Facial recognition or identification of individuals (explicitly forbidden and redacted).
  - License plate transcription.
  - Indoor navigation or building interiors.

---

## Technical Pipeline & Mechanics
1. **Preprocessing & Quality Verification:**
   - Blur detection via Laplacian variance ($Var(\nabla^2 I) < 80.0$).
   - Contrast check ($std < 22.0$) with adaptive CLAHE (Contrast Limited Adaptive Histogram Equalization) applied to the luminance ($L$) channel in LAB color space.
2. **Stair Detection:**
   - Sobel horizontal gradient filter with morphological horizontal kernel.
   - Hough line transform (`cv2.HoughLinesP`).
   - Signed vertical gradient disambiguation to reject flat painted pavement markings (e.g., pedestrian zebra crosswalks).
3. **Pavement Damage Detection:**
   - High-frequency edge density and contour fragmentation in lower $60\%$ pedestrian region.
4. **Obstacle & Bollard Detection:**
   - Vertical contour aspect ratio analysis ($AR > 2.0$) in pedestrian transit path.
5. **Curb Ramp Detection:**
   - Lower plane boundary transition with diagonal flare contour matching ($15^\circ - 45^\circ$).

---

## Empirical Benchmark Performance (Held-Out Test Split)

Evaluated on 10 held-out unseen test scenes ($640\times 480$, diverse lighting and obstacle conditions):

| Metric | Measured Value | Notes |
| :--- | :--- | :--- |
| **Mean Average Precision (mAP@0.5)** | **$0.6182$ ($61.8\%$)** | PASCAL VOC 11-point interpolation |
| **Expected Calibration Error (ECE)** | **$0.2683$** | 5-bin probability calibration |
| **Inference Latency (CPU)** | **$16.82\text{ ms}$** | Sub-20ms per frame |
| **Zebra Crosswalk False Positive Rate** | **$0.0\%$** | Verified across multiple painted stripe patterns |

### Per-Class Performance
- `curb_ramp_present`: Precision $1.0$, Recall $1.0$, AP $1.0$
- `stairs`: Precision $1.0$, Recall $0.50$, AP $0.545$
- `pole_or_bollard`: Precision $1.0$, Recall $0.50$, AP $0.545$
- `surface_damage`: Precision $0.33$, Recall $1.0$, AP $1.0$
- `curb_ramp_missing`: Precision $0.0$, Recall $0.0$, AP $0.0$ (Requires semantic contextual segmentation)

---

## Limitations & Domain Gaps
1. **Low-Contrast Fog / Deep Shadows:** Despite CLAHE enhancement, subtle stair risers under severe fog or rain glare can produce false negatives.
2. **Indian Pedestrian Infrastructure:** Unpaved dirt shoulders, open drainage culverts, two-wheeler encroachments, and monsoon waterlogging are outside the classical geometric rules and require deep multi-task segmentation models (e.g., Mask2Former / SegFormer).
3. **Absence of Evidence:** Detections are assigned explicit uncertainty scores ($0.08 - 0.65$), and predictions with confidence $< 0.70$ or uncertainty $> 0.35$ are flagged as tentative.
