"""
AccessGuard Model Adapters.
Interchangeable CV adapters for heuristic, PyTorch torchvision, YOLO, and ONNX Runtime.
"""

from app.cv.adapters.dummy_adapter import DummyAdapter
from app.cv.adapters.heuristic_adapter import HeuristicCVAdapter
from app.cv.adapters.torchvision_adapter import TorchvisionAdapter
from app.cv.adapters.yolo_adapter import YOLOAdapter
from app.cv.adapters.onnx_adapter import ONNXAdapter

__all__ = [
    "DummyAdapter",
    "HeuristicCVAdapter",
    "TorchvisionAdapter",
    "YOLOAdapter",
    "ONNXAdapter"
]
