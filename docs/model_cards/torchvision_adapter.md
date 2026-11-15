# Model Card: AccessGuard Torchvision Deep Learning Adapter

## Model Details
- **Model Name:** AccessGuard PyTorch Vision Adapter (`TorchvisionAdapter`)
- **Version:** 1.1.0
- **Model Architecture:** Faster R-CNN / Mask R-CNN with ResNet-50-FPN backbone
- **Framework:** PyTorch 2.4, Torchvision
- **Execution Mode:** Hybrid: Executes PyTorch tensor forward pass if weights and CUDA/CPU resources are available; automatically falls back to `HeuristicCVAdapter` when weights are absent.
- **Security Guard:** Enforces `weights_only=True` on all `torch.load()` operations to prevent arbitrary code execution via pickle deserialization (neutralizing Bandit B614).

---

## Intended Use
- Deep learning object detection and instance segmentation for accessibility obstacles.
- Enables community and municipal deployments with GPU resources to run high-capacity neural networks trained on custom sidewalk datasets (e.g., Cityscapes, Mapillary Vistas, Project Sidewalk).

---

## Preprocessing Contract
1. Input images are decoded to BGR format via OpenCV.
2. Conversion to RGB tensor normalized to $[0.0, 1.0]$.
3. Standard ImageNet mean ($[0.485, 0.456, 0.406]$) and standard deviation ($[0.229, 0.224, 0.225]$) normalization applied during model forward pass.
4. Output bounding boxes are normalized to $(ymin, xmin, ymax, xmax) \in [0.0, 1.0]$.

---

## Current Status & Deployment Guidance
- **Default State in Clean Repository:** Standby / Fallback mode. Without external pre-trained checkpoint files mounted to `data/models/torchvision_weights.pt`, the adapter delegates execution to `HeuristicCVAdapter`.
- **Benchmarking Notice:** Zero accuracy or benchmark numbers are fabricated for this adapter when weights are absent. Once fine-tuned weights are placed in the configured path, `run_benchmark_evaluation(adapter_name="torchvision")` executes evaluation on the held-out test suite.
