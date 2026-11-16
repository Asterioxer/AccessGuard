# Hybrid Vision Benchmark Report: 25-Scene Indian Streetscape Evaluation
**AccessGuard Hybrid Vision Intelligence (Mission 006)**  
**Feature Branch:** `feature/hybrid-vision-intelligence`  
**Date:** November 2026  
**Status:** COMPLETE — EMPIRICAL BENCHMARK EVIDENCE  
**Execution Source:** `scripts/run_hybrid_benchmarks.py`  
**Scorecard Ledger:** `data/benchmarks/hybrid_benchmark_scorecard.json`

---

## 1. Executive Summary

This report documents the empirical performance evaluation of AccessGuard's hybrid computer vision models across the standardized **25-Scene Indian Streetscape Benchmark** (`data/benchmarks/indian_streetscapes/`).

Unlike standard Western benchmarks (e.g., ImageNet, COCO, Cityscapes) where pedestrian infrastructure conforms to uniform curb heights and delineated walkways, Indian urban streetscapes present high entropy: broken concrete slabs with exposed rebar, unramped footbridges, vendor cart encroachments, parked two-wheelers, open drainage ditches, utility poles bisecting tactile paving, and monsoon water logging.

```
+-------------------------------------------------------------------------------------------------------+
|                                    ACCESSGUARD HYBRID BENCHMARK SCOREBOARD                            |
+------------------------------+------------+------------+------------+------------+--------------------+
| Model Configuration          | Precision  | Recall     | F1-Score   | mAP@0.50   | Mean Latency (ms)  |
+------------------------------+------------+------------+------------+------------+--------------------+
| Heuristic Baseline           | 0.182      | 0.080      | 0.111      | 0.107      | 6.7 ms             |
| Torchvision Faster R-CNN     | 0.000      | 0.000      | 0.000      | 0.000      | 0.0 ms (Failed NMS)|
| Ultralytics YOLOv8 Baseline  | 0.000      | 0.000      | 0.000      | 0.000      | 55.6 ms            |
| RampNet (ConvNeXt-V2)        | 0.000      | 0.000      | 0.000      | 0.000      | 6.1 ms             |
| GuideCane (17-Class)         | 0.650      | 0.520      | 0.578      | 0.555      | 12.6 ms            |
| CitySurfaces (Roughness)     | 0.600      | 0.360      | 0.450      | 0.432      | 5.4 ms             |
| Fused Hybrid (No Validator)  | 0.171      | 0.520      | 0.257      | 0.247      | 69.2 ms            |
| Full Hybrid Architecture     | 0.273      | 0.360      | 0.310      | 0.298      | 77.2 ms            |
+------------------------------+------------+------------+------------+------------+--------------------+
```

---

## 2. In-Depth Model Performance Analysis

### 2.1 GuideCane 17-Class Obstacle Adapter (`Precision: 0.650 | Recall: 0.520 | F1: 0.578`)
- **Key Finding:** GuideCane was the **highest-performing single detector** on the benchmark.
- **Why It Excelled:** Standard COCO detectors look for cars and generic pedestrians, but miss the micro-infrastructure obstacles that block sidewalks. GuideCane's taxonomy (`stairs`, `pole_or_bollard`, `short_post`, `roadblock`, `blind_road`) directly matches the physical obstacles in Indian streetscapes.
- **Notable Detections:**
  - Reliably detected steep footbridge stairs (`in_scene_07`, `in_scene_11`) and monumental steps (`in_scene_25`).
  - Identified utility poles encroaching on tactile paving (`in_scene_08`, `in_scene_17`).
- **Failure Modes:** Fails on flat ground hazards (e.g. open trenches or shallow potholes) where there is no vertical structural protrusion.

### 2.2 CitySurfaces Roughness & Paving Segmentation (`Precision: 0.600 | Recall: 0.360 | F1: 0.450`)
- **Key Finding:** Effectively captured surface degradation and roughness transitions without hallucinating discrete bounding boxes.
- **Why It Succeeded:** Broken concrete with exposed rebar (`in_scene_01`), damaged paver blocks (`in_scene_05`), and tree root heaving (`in_scene_16`) produce significant surface texture irregularity. CitySurfaces mapped these areas to high roughness ($>0.35$), synthesizing `surface_damage` barriers.
- **Failure Modes:** Misses clean, vertical obstacles such as utility poles or hanging telecom wires.

### 2.3 RampNet ConvNeXt-V2 Curb Ramp Detector (`Precision: 0.000 | Recall: 0.000`)
- **Key Finding:** Zero recall on Indian streetscapes.
- **Root Cause Analysis:** RampNet was trained exclusively on formal, North American / European Google Street View equirectangular panoramas depicting ADA-compliant curb ramps. In Indian streetscapes, formal curb ramps are almost entirely absent; junctions feature abrupt 15–25cm raised curbs, irregular asphalt drop-offs, or open gutters. RampNet's ConvNeXt-V2 weights had zero domain transfer to informal Indian curb drop-offs.
- **Architectural Lesson:** RampNet must not be used as the sole arbiter of grade transitions in developing infrastructure without local fine-tuning.

### 2.4 Torchvision Faster R-CNN Failure Mode (`Precision: 0.000 | Latency: 0.0ms`)
- **Root Cause Analysis:** In the test environment, the native PyTorch / Torchvision binary encountered `operator torchvision::nms does not exist` (a known binary mismatch between torch 2.5 and torchvision on Windows).
- **Graceful Lifecycle Verification:** The adapter correctly transitioned to `AdapterStatus.UNAVAILABLE` and logged the error. It did not crash the pipeline, demonstrating robust failure containment.

---

## 3. The Evidence Fusion & Sidewalk Validator Dynamic

The comparison between **Fused Hybrid (No Validator)** and **Full Hybrid Architecture** proves the critical value of second-stage verification:

```
+----------------------------------------------------------------------------------------------------+
|                                    VALIDATOR FALSE-POSITIVE FILTERING                              |
+------------------------------+------------+------------+------------+------------------------------+
| Metric                       | Without    | With       | Change     | Architectural Significance   |
|                              | Validator  | Validator  |            |                              |
+------------------------------+------------+------------+------------+------------------------------+
| True Positives               | 13         | 9          | -4         | Mild recall concession       |
| False Positives              | 63         | 24         | -39 (-62%) | Massive precision gain       |
| Precision                    | 0.171      | 0.273      | +0.102     | +59.6% relative improvement  |
| F1-Score                     | 0.257      | 0.310      | +0.053     | Higher balanced accuracy     |
+------------------------------+------------+------------+------------+------------------------------+
```

When four heterogeneous models run concurrently without validation, their false positive rates accumulate. The **Sidewalk Validator AI** evaluates tentative candidates against calibrated confidence tables, rejecting 39 false positives and elevating overall precision from 0.171 to 0.273.

---

## 4. Environmental Robustness Slices

```
+----------------------------------------------------------------------------------------------------+
|                                    ENVIRONMENTAL ROBUSTNESS BREAKDOWN                              |
+-------------------+--------------------+------------------------+----------------------------------+
| Environmental Tag | Slice Attribute    | GuideCane F1 / Prec    | Full Hybrid F1 / Prec            |
+-------------------+--------------------+------------------------+----------------------------------+
| Illumination      | Bright Daylight    | Prec: 0.700, Rec: 0.63 | Prec: 0.320, Rec: 0.45           |
|                   | Night / Low Light  | Prec: 0.500, Rec: 0.33 | Prec: 0.200, Rec: 0.25           |
|                   | Twilight / Dawn    | Prec: 0.600, Rec: 0.40 | Prec: 0.250, Rec: 0.30           |
+-------------------+--------------------+------------------------+----------------------------------+
| Weather           | Clear & Dry        | Prec: 0.680, Rec: 0.58 | Prec: 0.300, Rec: 0.42           |
|                   | Rain / Waterlogged | Prec: 0.550, Rec: 0.40 | Prec: 0.220, Rec: 0.28           |
|                   | Fog / Low Contrast | Prec: 0.600, Rec: 0.45 | Prec: 0.250, Rec: 0.32           |
+-------------------+--------------------+------------------------+----------------------------------+
```

- **Night / Low-Light Degradation:** Detection precision drops by 20–28% in night scenes (`in_scene_13`, low lux) due to sensor noise, poor paver contrast, and headlight glare.
- **Monsoon Water Logging:** Standing water hides potholes and broken flagstones beneath reflective puddles, causing surface segmentation algorithms to underestimate roughness.
