"""
Deterministic Tests for Torchvision Adapter Lifecycle and Inference Integrity.
Verifies:
1. Missing weights under STRICT_GENUINE reports UNAVAILABLE without false success.
2. Missing weights under DEVELOPMENT_FALLBACK reports DEGRADED with heuristic_fallback.
3. Genuine loaded model returns READY with model_inference (including zero detections).
4. OOM exceptions report RESOURCE_EXHAUSTED.
5. Runtime inference exceptions report INFERENCE_FAILED.
6. ImageProcessingService routing with requested_adapter='torchvision'.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from app.cv.adapters.torchvision_adapter import TorchvisionAdapter
from app.cv.adapters.base_v2 import AdapterStatus, ExecutionPolicy
from app.cv.orchestration.image_processing_service import (
    ImageProcessingService,
    ImageProcessingRequest
)


def test_torchvision_missing_weights_strict_mode():
    """Missing weights in strict mode must report UNAVAILABLE, not READY or model_inference."""
    adapter = TorchvisionAdapter(weights_path="/nonexistent/weights.pt")
    img = np.zeros((100, 100, 3), dtype=np.uint8)

    # load() should return False
    assert adapter.load(allow_fallback=False) is False
    assert adapter.is_initialized is False
    assert adapter._status == AdapterStatus.UNAVAILABLE

    # predict_v2() should report UNAVAILABLE
    res = adapter.predict_v2(img, allow_fallback=False)
    assert res.status == AdapterStatus.UNAVAILABLE
    assert res.execution_type == "unavailable"
    assert res.execution_attempted is False
    assert res.execution_succeeded is False
    assert len(res.detections) == 0

    # legacy predict() should return empty list
    assert adapter.predict(img) == []


def test_torchvision_missing_weights_fallback_mode():
    """Missing weights in fallback mode must report DEGRADED with heuristic_fallback."""
    adapter = TorchvisionAdapter(weights_path="/nonexistent/weights.pt")
    img = np.zeros((100, 100, 3), dtype=np.uint8)

    assert adapter.load(allow_fallback=True) is True
    assert adapter.is_initialized is True
    assert adapter._status == AdapterStatus.DEGRADED

    res = adapter.predict_v2(img, allow_fallback=True)
    assert res.status == AdapterStatus.DEGRADED
    assert res.execution_type == "heuristic_fallback"
    assert res.execution_attempted is True
    assert res.execution_succeeded is True
    assert len(res.detections) == 0


def test_torchvision_invalid_image_input():
    """Corrupt or non-array inputs must fail validation with INFERENCE_FAILED."""
    adapter = TorchvisionAdapter()
    res = adapter.predict_v2(None, allow_fallback=True)
    assert res.status == AdapterStatus.INFERENCE_FAILED
    assert res.error_code == "INVALID_IMAGE_INPUT"
    assert res.execution_succeeded is False


def test_torchvision_genuine_execution_and_zero_detections():
    """Loaded model running successfully with zero detections must report READY and model_inference."""
    adapter = TorchvisionAdapter(weights_path=None)
    adapter.is_initialized = True
    adapter._status = AdapterStatus.READY
    adapter.device = "cpu"
    adapter.transforms = MagicMock(return_value=MagicMock(unsqueeze=MagicMock(return_value=MagicMock(to=MagicMock()))))

    # Mock PyTorch model output with zero boxes
    mock_model = MagicMock()
    mock_model.return_value = [{
        'boxes': MagicMock(cpu=MagicMock(return_value=MagicMock(numpy=MagicMock(return_value=np.empty((0, 4)))))),
        'labels': MagicMock(cpu=MagicMock(return_value=MagicMock(numpy=MagicMock(return_value=np.empty((0,)))))),
        'scores': MagicMock(cpu=MagicMock(return_value=MagicMock(numpy=MagicMock(return_value=np.empty((0,))))))
    }]
    adapter.model = mock_model

    img = np.zeros((200, 200, 3), dtype=np.uint8)
    res = adapter.predict_v2(img, allow_fallback=False)

    assert res.status == AdapterStatus.READY
    assert res.execution_type == "model_inference"
    assert res.execution_succeeded is True
    assert res.execution_attempted is True
    assert len(res.detections) == 0
    assert res.metadata.get("output_count") == 0


def test_torchvision_oom_failure():
    """CUDA OOM must be classified as RESOURCE_EXHAUSTED."""
    adapter = TorchvisionAdapter(weights_path=None)
    adapter.is_initialized = True
    adapter._status = AdapterStatus.READY
    adapter.device = "cpu"
    adapter.transforms = MagicMock(return_value=MagicMock(unsqueeze=MagicMock(return_value=MagicMock(to=MagicMock()))))

    mock_model = MagicMock(side_effect=RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"))
    adapter.model = mock_model

    img = np.zeros((200, 200, 3), dtype=np.uint8)
    res = adapter.predict_v2(img, allow_fallback=False)

    assert res.status == AdapterStatus.RESOURCE_EXHAUSTED
    assert res.error_code == "RESOURCE_EXHAUSTED"
    assert res.execution_succeeded is False


def test_torchvision_runtime_inference_failure():
    """General tensor errors during forward pass must report INFERENCE_FAILED."""
    adapter = TorchvisionAdapter(weights_path=None)
    adapter.is_initialized = True
    adapter._status = AdapterStatus.READY
    adapter.device = "cpu"
    adapter.transforms = MagicMock(return_value=MagicMock(unsqueeze=MagicMock(return_value=MagicMock(to=MagicMock()))))

    mock_model = MagicMock(side_effect=RuntimeError("Dimension mismatch in conv2d kernel"))
    adapter.model = mock_model

    img = np.zeros((200, 200, 3), dtype=np.uint8)
    res = adapter.predict_v2(img, allow_fallback=False)

    assert res.status == AdapterStatus.INFERENCE_FAILED
    assert res.error_code == "INFERENCE_FAILED"
    assert res.execution_succeeded is False


def test_image_processing_service_torchvision_routing():
    """ImageProcessingService routing requested_adapter='torchvision' enforces correct lifecycle records."""
    service = ImageProcessingService()

    # Synthetic 50x50 white image
    import cv2
    img = np.full((50, 50, 3), 255, dtype=np.uint8)
    _, enc = cv2.imencode(".jpg", img)
    img_bytes = enc.tobytes()

    # 1. Under STRICT_GENUINE without weights -> UNAVAILABLE and overall_status == FAILED
    req_strict = ImageProcessingRequest(
        image_bytes=img_bytes,
        filename="test_tv_strict.jpg",
        source="user_upload",
        requested_adapter="torchvision",
        execution_policy=ExecutionPolicy.STRICT_GENUINE,
        enable_privacy_redaction=False,
        save_media=False
    )
    res_strict = service.process_image(req_strict)
    tv_record = next(r for r in res_strict.model_execution_records if r.adapter_name == "torchvision_adapter")
    assert tv_record.status == "UNAVAILABLE"
    assert tv_record.execution_type == "unavailable"
    assert res_strict.overall_status == "FAILED"

    # 2. Under DEVELOPMENT_FALLBACK without weights -> DEGRADED and overall_status == SUCCESS
    req_fallback = ImageProcessingRequest(
        image_bytes=img_bytes,
        filename="test_tv_fallback.jpg",
        source="user_upload",
        requested_adapter="torchvision",
        execution_policy=ExecutionPolicy.DEVELOPMENT_FALLBACK,
        enable_privacy_redaction=False,
        save_media=False
    )
    res_fallback = service.process_image(req_fallback)
    tv_record_fb = next(r for r in res_fallback.model_execution_records if r.adapter_name == "torchvision_adapter")
    assert tv_record_fb.status == "DEGRADED"
    assert tv_record_fb.execution_type == "heuristic_fallback"
    assert res_fallback.overall_status == "SUCCESS"
