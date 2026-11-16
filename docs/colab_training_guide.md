# Google Colab Training and Model Packaging Guide

This guide details the end-to-end workflow for training, benchmarking, exporting, and packaging real-world accessibility barrier detection models using Google Colab GPUs (Nvidia T4/A100) or local hardware, and transferring the optimized artifacts into AccessGuard.

---

## 1. Notebook Organization

AccessGuard provides 5 self-contained, reproducible Jupyter notebooks in the `notebooks/` directory:

| Notebook | Purpose | Key Outputs |
| :--- | :--- | :--- |
| `01_dataset_exploration.ipynb` | Streams and inspects Hugging Face datasets | Image schemas, bounding boxes, class distribution |
| `02_dataset_normalization.ipynb` | Maps source labels to AccessGuard unified categories | `dataset/accessguard.yaml`, normalized YOLO splits |
| `03_model_training_yolo.ipynb` | Trains lightweight YOLOv8 on GPU with CPU fallback | Checkpoints: `best.pt`, `last.pt`, loss curves |
| `04_model_validation_and_benchmarking.ipynb` | Held-out validation, mAP@0.5:0.95, ECE, confusion matrix | Comparative benchmark table, ECE plots |
| `05_onnx_export_and_packaging.ipynb` | Exports weights to ONNX with dynamic batching & packaging | `yolov8n.onnx`, `accessguard_yolov8n_onnx.tar.gz` |

---

## 2. Running in Google Colab

1. **Upload Notebooks**:
   - Go to [colab.research.google.com](https://colab.research.google.com).
   - Select **Upload** and upload any notebook from `notebooks/`.
2. **Enable GPU Hardware Accelerator**:
   - Click **Runtime** > **Change runtime type**.
   - Hardware accelerator: Select **T4 GPU** (or V100/A100 if Colab Pro).
3. **Execute Pipeline**:
   - Follow the steps sequentially from data streaming to model training.
   - Training runs with `yolov8n.pt` take approximately 12–25 minutes for 25 epochs on a standard T4 GPU.

---

## 3. Exporting & Transferring Model Weights to AccessGuard

Once training completes in `05_onnx_export_and_packaging.ipynb`:

1. Download the exported ONNX model artifact:
   ```python
   from google.colab import files
   files.download('yolov8n.onnx')
   ```
2. Place the file into your local AccessGuard workspace:
   ```bash
   cp ~/Downloads/yolov8n.onnx data/models/yolov8n.onnx
   ```
3. Restart or reload the backend:
   - AccessGuard's `ONNXAdapter` and `YOLOAdapter` will automatically detect `data/models/yolov8n.onnx` and activate C++ hardware-accelerated ONNX Runtime inference!
