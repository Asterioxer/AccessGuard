# Hybrid Vision Architecture Assessment & Gap Analysis
**AccessGuard Hybrid Vision Intelligence (Mission 006)**  
**Feature Branch:** `feature/hybrid-vision-intelligence`  
**Date:** November 2026  
**Status:** COMPLETE — ARCHITECTURAL AUDIT & TARGET SPECIFICATION

---

## 1. Context & Objectives

AccessGuard's core purpose is to identify pedestrian infrastructure barriers, evaluate sidewalk accessibility, maintain verifiable geospatial evidence, and route mobility-impaired individuals safely. In Missions 001–005, AccessGuard developed a modular computer vision pipeline featuring heuristic filters, Torchvision Faster R-CNN, Ultralytics YOLO, and semantic segmentation prototypes.

However, moving from isolated model experiments to a production-grade **hybrid vision intelligence platform** requires addressing fundamental architectural limitations in model lifecycle, representation capacity, error semantics, and evidence synthesis.

---

## 2. Forensic Gap Analysis of Existing CV Architecture

### Gap 1: Output Representation Myopia (Bounding Boxes Only)
- **Current State:** The current `ModelAdapter.predict()` interface in `backend/app/cv/base.py` is hardcoded to return `List[DetectedBarrier]`, where every barrier must possess a normalized rectangular bounding box `(ymin, xmin, ymax, xmax)`.
- **Limitation:**
  - **RampNet** outputs continuous heatmaps and discrete point coordinates $(x, y)$ corresponding to curb ramp apexes. Forcing RampNet outputs into artificial rectangular boxes distorts spatial precision.
  - **CitySurfaces** produces full-scene 2D semantic segmentation masks denoting paving materials (`concrete`, `bricks`, `granite`, `cobblestone`). These are continuous surface fields, not discrete rectangular obstacles.
  - **Sidewalk Validator** produces binary verification verdicts (`correct` vs `incorrect`) and calibrated accuracy percentiles for given crop coordinates.

### Gap 2: Silent Failure Masking (Zero Detections Anti-Pattern)
- **Current State:** In `backend/app/cv/adapters/yolo_adapter.py` and `torchvision_adapter.py`:
  ```python
  except Exception as e:
      logger.warning(f"Could not initialize YOLO ({e}); running in fallback mode.")
      self.is_initialized = True
      return True
  ...
  if self.model is None:
      return []
  ```
- **Hazard:** When weights fail to load, CUDA throws an OOM, or a model crashes during inference, the adapter catches the exception and returns `[]` (an empty list). Downstream services and routing engines interpret `[]` as **"Zero barriers detected — path is fully accessible"**. In reality, the vision system failed completely. This is a catastrophic safety flaw for wheelchair users.

### Gap 3: Absence of Model Lifecycle & Memory Governance
- **Current State:** Existing adapters provide only a one-way `load()` method. There is no `unload()` method, no memory deallocation, no device migration, and no structured health check.
- **Hardware Hazard (RTX 3050 4GB VRAM):** When executing multiple deep learning architectures (ConvNeXt-V2 Base, DINOv2-Large, HRNet-W48, YOLOv8) within a single process, the GPU runs out of VRAM immediately if models remain permanently resident in memory. The system lacks a sequential staging and memory-offloading protocol (`RESOURCE_CONSTRAINED` mode).

### Gap 4: Single-Model Pipeline Assumption
- **Current State:** `CVPipeline` in `backend/app/cv/pipeline.py` initializes exactly one adapter at a time (`self.adapter = self._init_adapter(name)`).
- **Limitation:** A single model cannot excel simultaneously at panoramic curb ramp detection, micro-obstacle localization (bollards, tactile paving), surface material roughness estimation, and post-detection verification. The system requires an **ensemble and evidence fusion architecture**.

---

## 3. The Target Hybrid Vision Architecture

To resolve these gaps, AccessGuard introduces a modular, multi-tier hybrid vision architecture:

```
                                  [Input Image / Panorama]
                                             |
                   +-------------------------+-------------------------+
                   |                                                   |
      [Perspective Camera View]                           [360° Equirectangular View]
                   |                                                   |
                   v                                                   v
      +-------------------------+                         +-------------------------+
      |  Obstacle Detectors     |                         |  Curb Ramp Detector     |
      |  - YOLOv8 (COCO/Indian) |                         |  - RampNet (ConvNeXt)   |
      |  - GuideCane (17 cls)   |                         |    Heatmap Peaks (x, y) |
      +------------+------------+                         +------------+------------+
                   |                                                   |
                   +-------------------------+-------------------------+
                                             |
                                             v
                           +-----------------------------------+
                           |    MULTI-MODEL EVIDENCE FUSION    |
                           |  - Coordinate Normalization       |
                           |  - Spatial Clustering (IoU / R)   |
                           |  - Cross-Model Consensus Scoring  |
                           |  - Calibration & Decay Gating     |
                           +-----------------+-----------------+
                                             |
                                             v
                           +-----------------------------------+
                           |    SECONDARY VERIFICATION POOL    |
                           |  - Sidewalk Validator (DINOv2)    |
                           |  - CitySurfaces (HRNet Paving)    |
                           +-----------------+-----------------+
                                             |
                                             v
                           +-----------------------------------+
                           |    FUSED ACCESSIBILITY EVIDENCE   |
                           |  - Status: VERIFIED / TENTATIVE   |
                           |  - Calibrated Confidence          |
                           |  - PostGIS Registration & Routing |
                           +-----------------------------------+
```

---

## 4. Key Architectural Decisions

1. **Unified Adapter Lifecycle (`ModelAdapterV2`):**
   - Every adapter implements five standard lifecycle operations:
     1. `load(device: Optional[str] = None) -> bool`
     2. `unload() -> bool` (explicit `del`, `gc.collect()`, `torch.cuda.empty_cache()`)
     3. `health() -> AdapterHealth` (tracks `AdapterStatus`, VRAM/RAM usage, latencies)
     4. `predict(image, **kwargs) -> AdapterInferenceResult`
     5. `metadata() -> AdapterMetadata` (capabilities, taxonomy, license)
2. **Explicit Failure Semantics:**
   - Inference errors return `AdapterStatus.INFERENCE_FAILED` with the exact error traceback.
   - Unloaded models return `AdapterStatus.NOT_LOADED`.
   - Successful zero-barrier inferences return `AdapterStatus.READY` with `detections=[]`. Downstream consumers must check `result.is_successful` before trusting `detections`.
3. **Multi-Modal Output Schema (`AdapterInferenceResult`):**
   - Encapsulates `detections` (bounding boxes), `keypoints` (points for ramps), `masks` (semantic segmentations), and `verifications` (validator verdicts).
4. **Dual Operating Profiles (`FULL_HYBRID` vs `RESOURCE_CONSTRAINED`):**
   - **`FULL_HYBRID` (High-VRAM workstation / server $\ge 12\text{GB}$ VRAM):** Models remain resident in memory for minimal latency.
   - **`RESOURCE_CONSTRAINED` (RTX 3050 Laptop $\le 4\text{GB}$ VRAM / CPU):** Executes models sequentially, offloading each backbone from GPU VRAM before loading the next stage.
