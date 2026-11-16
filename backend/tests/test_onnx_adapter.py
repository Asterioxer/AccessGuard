"""
Tests for ONNX Runtime adapter and YOLO-ONNX fallback.
"""

import os
import numpy as np
import pytest
from app.cv.adapters.onnx_adapter import ONNXAdapter
from app.cv.adapters.yolo_adapter import YOLOAdapter

def test_onnx_adapter_initialization():
    adapter = ONNXAdapter()
    assert adapter.name == "onnx_adapter"
    assert adapter.load() is True
    assert adapter.is_initialized is True

def test_onnx_adapter_preprocessing():
    adapter = ONNXAdapter(input_size=(640, 640))
    sample_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
    tensor, scale, (dx, dy) = adapter.preprocess(sample_bgr)

    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.float32
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0

def test_onnx_adapter_predict():
    model_path = "data/models/yolov8n.onnx"
    adapter = ONNXAdapter(model_path=model_path if os.path.exists(model_path) else None)
    adapter.load()
    sample_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
    preds = adapter.predict(sample_bgr)
    assert isinstance(preds, list)

def test_yolo_adapter_with_onnx_backend():
    yolo_adp = YOLOAdapter()
    assert yolo_adp.load() is True
    sample_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
    preds = yolo_adp.predict(sample_bgr)
    assert isinstance(preds, list)
