# AccessGuard Computer Vision Evaluation & Validation Report
**Author**: Independent Senior ML Engineer  
**Audit Target**: `backend/app/cv/`  
**Execution Timestamp**: November 2026

---

## 1. Model Adapters Forensic Inspection

| Adapter | Source File | Inspection Verdict |
| :--- | :--- | :--- |
| **`DummyAdapter`** | `adapters/dummy_adapter.py` | **Pass**: Deterministic mock correctly simulating standard, empty, and uncertain modes. |
| **`HeuristicCVAdapter`** | `adapters/heuristic_adapter.py` | **Pass with Technical Debt**: Classical computer vision using morphological filters, Sobel rungs, Laplacian variance, and contour ratios. Operates with high execution speed ($16.8\text{ ms/frame}$), but lacks semantic scene understanding. |
| **`TorchvisionAdapter`** | `adapters/torchvision_adapter.py` | **Incomplete**: Intended for Mask R-CNN. However, line 34 checks `if self.weights_path and torch.cuda.is_available():`. In standard CPU deployments or fresh installs without weights, `self.model` is `None` and silently falls back to `HeuristicCVAdapter`. |
| **`YOLOAdapter`** | `adapters/yolo_adapter.py` | **Incomplete**: Stub implementation. Lines 29-31 unconditionally fall back to `HeuristicCVAdapter` when `model_path` is empty. |

---

## 2. Preprocessing & Channel Consistency Audit

1. **Color Space Handling**:
   - `HeuristicCVAdapter` expects standard OpenCV BGR arrays and converts via `cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)`.
   - Executable testing (`scripts/audit/audit_cv_validation.py`) confirmed that passing inverted RGB arrays does not cause catastrophic failures for edge detection because gray luminance weights preserve edge contrast. However, color-dependent detections (e.g. tactile yellow paving) will produce inconsistent results if RGB/BGR conventions are not strictly enforced at the ingestion boundary.
2. **Coordinate Normalization**:
   - Bounding boxes are consistently normalized to $[0.0, 1.0]$ in $(y_{\min}, x_{\min}, y_{\max}, x_{\max})$ order across all adapters.
3. **Decompression Bomb Protection**:
   - **VULNERABILITY**: `cv2.imdecode` and `Image.open` decode incoming images into memory without checking raw pixel dimensions ($W \times H$). An attacker can upload a compressed $2\text{MB}$ JPEG that decompresses into $50,000 \times 50,000$ pixels ($7.5\text{ GB}$ uncompressed RAM), causing Out-Of-Memory container termination.

---

## 3. Inference Confidence & Uncertainty Semantics

In `HeuristicCVAdapter`, confidence scores are **not probabilistic posteriors** generated from training distributions or softmax calibration; they are **hand-crafted piecewise linear heuristics**:
- Stairs: $\text{Confidence} = \min(0.94, 0.65 + 0.06 \times \text{step\_count})$.
- Surface Damage: $\text{Confidence} = \min(0.92, 0.55 + 5.0 \times \text{edge\_density})$.
- Vertical Obstacles: Fixed constant $0.84$.
- Curb Ramps: Fixed constant $0.86$.

### Mathematical Verification of ECE
Expected Calibration Error (ECE) is implemented in `metrics.py`:
$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
While the mathematical formula in `metrics.py` is accurate, calculating ECE over hand-crafted heuristic scores is conceptually misaligned because the confidence is not produced by a calibrated statistical model.

---

## 4. Evaluation Dataset Split & Data Leakage Audit

```
┌────────────────────────────────────────────────────────────────────────┐
│ CRITICAL AUDIT FINDING: EVALUATION DATA LEAKAGE                        │
│                                                                        │
│ In backend/app/cv/evaluation/run_eval.py:                              │
│ 1. generate_synthetic_benchmark_dataset() generates exactly 5 scenes. │
│ 2. Heuristic filter thresholds (e.g. edge_density > 0.025,             │
│    laplacian_var > 100.0, step count >= 3) were explicitly             │
│    tuned to match those exact 5 synthetic scenes.                      │
│ 3. Zero independent test/validation split exists.                      │
│ 4. Reported 1.0 mAP@0.5 is an artifact of test-set overfitting.        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Empirical Stress Testing (False Positives & Edge Cases)

The audit executed empirical stress tests using `scripts/audit/audit_cv_validation.py`:

### 5.1. Zebra Crosswalk False-Positive Anomaly
- **Test**: Synthetic image of standard pedestrian zebra crossing (5 parallel white stripes on asphalt).
- **Result**: `Crosswalk detected as stairs? TRUE (Confidence: 0.94, Detections: 2)`.
- **Root Cause**: The Hough line transform and horizontal Sobel filter cannot differentiate between horizontal concrete step treads and painted road crossing stripes.
- **Impact**: **High Severity**. A wheelchair user approaching a safe pedestrian zebra crosswalk would be warned by the system that the path is blocked by impassable stairs!

### 5.2. Adverse Lighting & Environmental Stress
- **Severe Underexposure (Night, mean brightness < 20)**: Detected stairs with higher uncertainty ($0.41$).
- **Overexposure / Glare (mean brightness > 230)**: Detected stairs with higher uncertainty ($0.39$).
- **Low Contrast / Fog ($\sigma < 10$)**: **False Negative**. Failed to detect step rungs due to threshold failure.

---

## 6. Indian Pedestrian Infrastructure Reality Check

The current evaluation benchmark is based on idealized Western sidewalks and fails completely to represent the reality of Indian urban infrastructure.

### Unhandled High-Impact Indian Barriers:
1. **Open & Broken Storm-Water Drains**: Missing concrete slab covers along pedestrian footpaths (severe life-safety hazard for wheelchair users).
2. **Dislodged Interlocking Paver Blocks**: Widespread uneven settling, loose tiles, and missing blocks creating wheel-trapping chasms.
3. **Severe Footpath Encroachment**: Parked two-wheelers, auto-rickshaws, chai stalls, and street vendor carts forcing pedestrians onto active carriageways.
4. **Non-Standard High Curbs**: Footpath curbs in Indian cities frequently exceed $30\text{cm} - 50\text{cm}$ in height (2 to 3 times higher than Western $15\text{cm}$ curbs) with completely absent curb cuts.
5. **Mid-Sidewalk Utility Obstacles**: Electric utility poles, distribution transformers, and telecom boxes installed directly in the center of $1.2\text{m}$ footpaths.
6. **Monsoon Waterlogging**: Open puddles concealing road defects and sewer grates.

**Recommendation**: The taxonomy in `backend/app/cv/base.py` must be expanded to include:
- `open_drain`
- `paver_damage`
- `two_wheeler_obstruction`
- `vendor_obstruction`
- `waterlogged_hazard`
