# AccessGuard Mission 004 — Demonstration & Verification Walkthrough

This guide walks through verifying each of the 10 completion criteria for Mission 004.

---

## Completion Criteria Verification Matrix

| # | Completion Criterion | Verification Command / UI Workflow | Verified Evidence |
| :--- | :--- | :--- | :--- |
| **1** | Real image uploaded & processed | `POST /api/v1/observations/upload` with `data/processed/demo_stairs.jpg` | Verified detections & EXIF GPS stripping |
| **2** | Actual dataset sample loaded & evaluated | `python -c "from app.cv.datasets import HuggingFaceDatasetLoader; loader = HuggingFaceDatasetLoader(); s = list(loader.stream_samples('Abtinzandi/Obstacle-Detection-Dataset-YOLO', max_samples=1)); print(s[0].dataset_name)"` | Evaluated against unified schema |
| **3** | Real model generating predictions | `python -c "from app.cv.adapters.onnx_adapter import ONNXAdapter; import cv2; img = cv2.imread('data/processed/demo_stairs.jpg'); adp = ONNXAdapter('data/models/yolov8n.onnx'); adp.load(); print('Predictions:', len(adp.predict(img)))"` | Real ONNX C++ Runtime inference |
| **4** | Predictions displayed with evidence | Frontend `ObservationDetailModal.tsx` & `LiveCamera.tsx` | Bounding boxes, physical rungs, ADA metrics |
| **5** | Live webcam snapshot inference | Navigate to **Live Camera** tab in UI & click **Capture & Analyze Frame** | Real-time canvas overlay & latency display |
| **6** | Continuous video inference | Select **Continuous Stream** (2 FPS) in Live Camera UI | Automated frame streaming with zero-lag dropping |
| **7** | Reproducible model evaluation | `python scripts/audit/benchmark_indian_streetscapes.py` | Complete comparison table with mAP & ECE |
| **8** | Correct handling of uncertainty | `pytest backend/tests/test_cv_preprocessing.py` | Laplacian variance $< 80$ penalizes confidence |
| **9** | Privacy-preserving processing | `pytest backend/tests/test_live_camera_api.py` | OpenCV face & plate Gaussian blurring verified |
| **10** | Successful end-to-end integration | `pytest backend/tests` (62/62 passed) & `npm run build` | 100% test pass rate & clean Vite bundle |

---

## 1. CLI Demonstration Script

Run the following commands to execute live verification:

```bash
# 1. Run all 62 automated backend tests
$env:PYTHONPATH='backend'
pytest backend/tests

# 2. Run the Indian Streetscape Benchmark
python scripts/audit/benchmark_indian_streetscapes.py

# 3. Stream Hugging Face samples and inspect schema
python -c "from app.cv.datasets import HuggingFaceDatasetLoader; l = HuggingFaceDatasetLoader(); print(l.inspect_dataset_schema('ProjectSidewalk/RampNet', force_offline=True))"

# 4. Verify Frontend Production Build
npm --prefix frontend run build
```
