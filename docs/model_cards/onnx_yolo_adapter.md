# Model Card: ONNX Runtime YOLOv8n Adapter

**Model Identifier:** `accessguard-onnx-yolov8n`  
**Framework:** ONNX Runtime 1.26.0 (CPU + CUDA Execution Providers)  
**Input Resolution:** $640 \times 640 \times 3$ (Float32 BCHW normalized $[0, 1]$)  
**Architecture:** YOLOv8-nano fused convolutional backbone with decoupled detection head  
**Model Size on Disk:** $12.3\text{ MB}$ (`data/models/yolov8n.onnx`)  

---

## 1. Intended Use

- **Task:** Real-time pedestrian accessibility barrier detection for street and sidewalk imagery.
- **Deployment Targets:** Browser live webcam inference, low-power mobile sidecars, edge servers, and cloud FastAPI instances.
- **Target Accessibility Classes:**
  1. `curb_ramp_present`: Dropped curbs meeting wheelchair gradient standards.
  2. `stairs`: Stepped vertical changes presenting physical barriers.
  3. `surface_damage`: Severe pavement cracking, missing slabs, potholes.
  4. `pole_or_bollard`: Vertical bollards, narrow posts, utility poles.
  5. `construction_obstacle`: Roadwork barriers, vendor pushcarts, temporary obstructions.

---

## 2. Benchmark Performance

Evaluated on the independent AccessGuard held-out test split and Indian pedestrian streetscape collection:

| Metric | Measured Value | Standard Target | Status |
| :--- | :--- | :--- | :--- |
| **mAP@0.5 (General Accessibility)** | 0.7420 (74.2%) | $\ge 0.70$ | **PASS** |
| **mAP@0.5:0.95 (COCO Metric)** | 0.5180 (51.8%) | $\ge 0.45$ | **PASS** |
| **Expected Calibration Error (ECE)** | 0.1420 (14.2%) | $\le 0.20$ | **PASS** |
| **Mean Inference Latency (CPU)** | $25.06\text{ ms} \pm 2.1\text{ ms}$ | $< 50\text{ ms}$ | **PASS** |
| **Throughput (FPS)** | $39.9\text{ FPS}$ (Single thread) | $\ge 15\text{ FPS}$ | **PASS** |
| **Peak VRAM Allocated** | $412\text{ MB}$ (CUDA) | $< 1000\text{ MB}$ | **PASS** |

---

## 3. Operational Limitations & Edge Cases

1. **Indian Infrastructure Bias:** Pretrained baseline YOLO models achieve low accuracy ($\approx 7.27\%$ mAP) on unpaved shoulders, open stormwater drains, and street vendor carts without active-learning domain adaptation.
2. **Extreme Motion Blur:** When camera angular velocity exceeds $45^\circ/\text{s}$, Laplacian variance drops below $80.0$, applying a 0.35 uncertainty penalty.
3. **Monocular Scale Ambiguity:** Bounding box size cannot definitively establish wheelchair clearance width without calibrated ground-plane homography.
