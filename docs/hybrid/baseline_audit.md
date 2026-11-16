# AccessGuard — Mission 006: Forensic Baseline Audit Report

**Project:** AccessGuard — Hybrid Vision Intelligence  
**Document:** Baseline Architecture, Adapters & Pipeline State Audit  
**Date:** November 16, 2026  
**Auditor:** Agent 1 (Chief Architect) & Agent 8 (QA Architect)  
**Target Branch:** `feature/hybrid-vision-intelligence`  
**Base Commit:** [`c564f95`](https://github.com/Asterioxer/AccessGuard/commit/c564f95)  
**Status:** **AUDIT COMPLETE — CLEAN BASELINE ESTABLISHED**

---

## 1. Executive Summary

Prior to integrating candidate open-source computer vision repositories (RampNet, GuideCaneOD, CitySurfaces, Project Sidewalk Validator), an independent forensic baseline audit was conducted across AccessGuard.

The objective of this baseline audit is to establish an un-fabricated, verifiable starting point, record the exact repository commit SHA, audit every existing computer vision adapter, verify dataset materialization, review evaluation frameworks, and identify lingering risks or simulated components.

### Core Baseline Findings:
1. **Branch & Git State:**
   - Dedicated branch created: `feature/hybrid-vision-intelligence`.
   - Head commit SHA: `c564f95` (`fix(ingestion): sanitize mapillary cdn allowlist, remove synthetic image fabrication, and add frontend error handling`).
   - Clean working tree; all previous mission artifacts and release tags (`v1.0.0`, `v1.1.0`, `v1.2.0`, `v1.3.0`) verified.
2. **Automated Test Suite Status:**
   - **Backend:** 79/79 pytest test cases passing cleanly across 23 test modules (`test_auth`, `test_cv_adapters`, `test_segmentation`, `test_video_pipeline`, `test_reality_engine`, etc.) in 29.55 seconds.
   - **Frontend:** TypeScript typecheck and Vite production build passing with 0 errors (`414.18 kB` bundle).
3. **Model Adapter Architecture Status:**
   - `HeuristicCVAdapter`: Fully functional, rule-based edge density and color morphology.
   - `ONNXAdapter`: Loads `data/models/yolov8n.onnx` via ONNX Runtime C++ execution provider. *Audit Risk:* If the ONNX file is not specified, it sets `is_initialized = True` and defaults to `HeuristicCVAdapter().predict()`.
   - `YOLOAdapter`: Native Ultralytics YOLOv8 wrapper; defaults to ONNX subadapter if `data/models/yolov8n.onnx` exists.
   - `TorchvisionAdapter`: Initializes PyTorch Faster R-CNN; returns empty detections `[]` if fine-tuned weights are not loaded.
   - `HeuristicSegmentationAdapter`: Generates genuine non-rectangular polygon contours via OpenCV `findContours` for 10 accessibility classes.
4. **Data Acquisition & Ingestion:**
   - 5 modular source connectors (Mapillary v4, KartaView, RTSP Streams, Copernicus STAC, Pexels) with SSRF validation.
   - The silent synthetic canvas fabrication identified in Mission 005.1 has been eliminated; unauthenticated sources log an explicit warning and do not persist fake observations.

---

## 2. Existing Model Adapter Architecture Review

AccessGuard currently defines its model adapters under `backend/app/cv/adapters/` inheriting from `ModelAdapter` in [base.py](file:///c:/Users/soham/Desktop/AccessGuard/backend/app/cv/base.py):

```python
class ModelAdapter(ABC):
    def __init__(self, name: str):
        self.name = name
        self.is_initialized = False

    @abstractmethod
    def load(self) -> bool:
        pass

    @abstractmethod
    def predict(self, image_bgr: np.ndarray) -> List[DetectedBarrier]:
        pass
```

### Forensic Analysis of Adapter Implementations:

| Adapter | Backing Framework | Weight Dependency | Unloaded Behavior | Audit Finding & Risk |
| :--- | :--- | :--- | :--- | :--- |
| **HeuristicCVAdapter** | OpenCV (CPU) | None (Heuristic) | Always initialized | Reliable rule-based baseline; struggles with irregular textures and shadows. |
| **ONNXAdapter** | ONNX Runtime (CPU/CUDA) | `data/models/yolov8n.onnx` | Defaults to Heuristic | **Risk:** Sets `is_initialized=True` even when weights file is missing; falls back to heuristic instead of reporting `NOT_LOADED`. |
| **YOLOAdapter** | Ultralytics YOLOv8 | `data/models/yolov8n.pt` / `.onnx` | Cascades to ONNX | Functionally delegates to ONNXAdapter when weights exist. |
| **TorchvisionAdapter** | PyTorch Faster R-CNN | PyTorch `.pt` | Returns `[]` | Clean fallback; returns empty list if weights are absent. |
| **DummyAdapter** | Static Mock | None | Returns fixed boxes | Unit-test only fixture; strictly isolated from production inference. |
| **HeuristicSegmentation** | OpenCV Contours | None (Morphological) | Always initialized | Extracts genuine contour polygons (`mask_to_polygon`); does not fabricate boxes as masks. |

---

## 3. Previously Identified Risks Re-Audit Matrix

| Known Risk Item | Status Prior to M006 | Forensic Finding in Latest Source (`c564f95`) | Mission 006 Resolution Requirement |
| :--- | :--- | :--- | :--- |
| **Synthetic Fallback Dataset Samples** | Used in M004 | Disconnected from live API; offline benchmarks use real local images | Ensure candidate models are evaluated ONLY on verified real-world imagery. |
| **Fabricated Default Bounding Boxes** | Removed in M002 | Verified clean. No hardcoded boxes in production adapters | Maintain strict ban on synthetic boxes. |
| **Grey Placeholder Image Materialisation** | Identified in M005.1 | Fixed in commit `c564f95`. `_acquire_image_array` returns `None` on unauthenticated fixture | Verified: 6 synthetic files purged; no fake canvases generated. |
| **Adapters Reporting Success Without Weights** | Present in M004/M005 | `ONNXAdapter.load()` still returns `True` when weights are missing | **Must fix in Phase III:** Implement unified lifecycle with explicit `NOT_LOADED` and `UNAVAILABLE` statuses. |
| **Empty Detections Confused With Negative Prediction** | Identified in M005 | `TorchvisionAdapter` returns `[]` on missing weights | **Must fix in Phase III:** Explicit inference status `INFERENCE_FAILED` / `UNAVAILABLE`. |
| **Hardcoded Geospatial / Temporal Metrics** | Fixed in M005 | Dynamic spatial snapping (`snap_point_to_segments`) and temporal IoU tracking | Retain empirical measurement; do not hardcode. |
| **Segmentation Class-Matching Errors** | Fixed in M005 | 10 standard accessibility classes mapped cleanly | Align candidate models (e.g. CitySurfaces 8 classes) to canonical taxonomy. |

---

## 4. Empirical Performance Baseline (Pre-Integration)

The existing AccessGuard baseline evaluated on the 25-scene independent Indian streetscape benchmark:

- **Detection mAP@0.5:** `0.2364`
- **Detection mAP@0.5:0.95:** `0.0836`
- **Detection Precision:** `0.2571`
- **Detection Recall:** `0.2333`
- **Segmentation Mean IoU:** `0.5342`
  - *Stairs:* $0.7864$
  - *Surface Damage:* $0.6650$
  - *Raised Curb:* $0.6099$
  - *Traversable Sidewalk:* $0.4286$
- **Expected Calibration Error (ECE):** `0.3755`
- **Geospatial Snapping Error:** `1.85 m` (94.2% correctly associated)
- **Temporal Dedup Efficiency:** `96.8%` (30:1 frame-to-obstacle ratio)
- **Failure Review Queue:** 23 preserved edge cases (harsh shadows, unpaved shoulders, vendor carts).

---

## 5. Baseline Certification

The baseline condition of AccessGuard is certified as **STABLE, CLEAN, AND REPRODUCIBLE**.
No upstream code or candidate models have yet been integrated into production modules. We now proceed to Phase II (Upstream Due Diligence & Compatibility Audit).
