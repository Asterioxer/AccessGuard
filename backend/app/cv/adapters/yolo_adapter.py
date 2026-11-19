"""
Ultralytics YOLO Model Adapter for AccessGuard.

Provides native PyTorch and Ultralytics YOLOv8/v11 inference for
detecting accessibility obstacles, curb ramps, stairs, and surface damage.
"""

import os
from typing import List, Optional
import numpy as np

from app.cv.base import ModelAdapter, DetectedBarrier, ACCESSIBILITY_CLASSES
from app.cv.adapters.base_v2 import AdapterStatus
from app.core.logging import logger

class YOLOAdapter(ModelAdapter):
    """
    Ultralytics YOLO model adapter.
    Runs native YOLOv8/v11 inference with class mapping and evidence extraction.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.45
    ):
        super().__init__(name="yolo_adapter")
        self.version = "1.2.0"
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.model = None
        self._device = "cpu"
        self._weight_sha256: Optional[str] = None
        self._status: AdapterStatus = AdapterStatus.NOT_LOADED
        self._is_mock: bool = False
        self._last_error: Optional[str] = None
        self.onnx_subadapter = None

    def load(self, allow_fallback: bool = False) -> bool:
        """Loads Ultralytics YOLO model from weights file or ONNX runtime."""
        import hashlib
        try:
            onnx_default = "data/models/yolov8n.onnx"
            pt_default = "data/models/yolov8n.pt"

            # If explicit model path provided
            if self.model_path:
                if not os.path.exists(self.model_path):
                    self._last_error = f"UNAVAILABLE: MODEL_WEIGHTS_NOT_FOUND ({self.model_path})"
                    if allow_fallback:
                        self.is_initialized = True
                        self._is_mock = True
                        self._status = AdapterStatus.DEGRADED
                        return True
                    else:
                        self.is_initialized = False
                        self._is_mock = False
                        self._status = AdapterStatus.UNAVAILABLE
                        return False

                if self.model_path.endswith(".onnx"):
                    from app.cv.adapters.onnx_adapter import ONNXAdapter
                    self.onnx_subadapter = ONNXAdapter(self.model_path)
                    if self.onnx_subadapter.load():
                        self.is_initialized = True
                        self._status = AdapterStatus.READY
                        self._is_mock = False
                        with open(self.model_path, "rb") as f_w:
                            self._weight_sha256 = hashlib.sha256(f_w.read()).hexdigest()
                        return True
                    else:
                        self._last_error = f"UNAVAILABLE: ONNX_LOAD_FAILED ({self.model_path})"
                        self.is_initialized = False
                        self._is_mock = False
                        self._status = AdapterStatus.UNAVAILABLE
                        return False
                else:
                    from ultralytics import YOLO
                    self.model = YOLO(self.model_path)
                    self.is_initialized = True
                    self._status = AdapterStatus.READY
                    self._is_mock = False
                    with open(self.model_path, "rb") as f_w:
                        self._weight_sha256 = hashlib.sha256(f_w.read()).hexdigest()
                    return True

            # Default search order: ONNX first, then PyTorch weights
            if os.path.exists(onnx_default):
                from app.cv.adapters.onnx_adapter import ONNXAdapter
                self.onnx_subadapter = ONNXAdapter(onnx_default)
                if self.onnx_subadapter.load():
                    self.model_path = onnx_default
                    self.is_initialized = True
                    self._status = AdapterStatus.READY
                    self._is_mock = False
                    with open(onnx_default, "rb") as f_w:
                        self._weight_sha256 = hashlib.sha256(f_w.read()).hexdigest()
                    return True

            if os.path.exists(pt_default):
                from ultralytics import YOLO
                self.model = YOLO(pt_default)
                self.model_path = pt_default
                self.is_initialized = True
                self._status = AdapterStatus.READY
                self._is_mock = False
                with open(pt_default, "rb") as f_w:
                    self._weight_sha256 = hashlib.sha256(f_w.read()).hexdigest()
                return True

            # If neither onnx nor pt weights exist
            self._last_error = "UNAVAILABLE: MODEL_WEIGHTS_NOT_FOUND (yolov8n.onnx / yolov8n.pt)"
            if allow_fallback:
                logger.info("YOLO weights not found locally; running in fallback mode.")
                self.is_initialized = True
                self._is_mock = True
                self._status = AdapterStatus.DEGRADED
                return True
            else:
                self.is_initialized = False
                self._is_mock = False
                self._status = AdapterStatus.UNAVAILABLE
                logger.warning("YOLO weights not found; model marked UNAVAILABLE.")
                return False

        except Exception as e:
            self._last_error = f"UNAVAILABLE: MODEL_LOAD_FAILED ({e})"
            if allow_fallback:
                logger.warning(f"Could not initialize YOLO ({e}); running in fallback mode.")
                self.is_initialized = True
                self._is_mock = True
                self._status = AdapterStatus.DEGRADED
                return True
            else:
                self.is_initialized = False
                self._is_mock = False
                self._status = AdapterStatus.UNAVAILABLE
                logger.warning(f"Could not load YOLO ({e}); marked UNAVAILABLE.")
                return False

    def predict(self, image_bgr: np.ndarray) -> List[DetectedBarrier]:
        """Runs inference on a single BGR image array."""
        if not self.is_initialized:
            self.load()

        if hasattr(self, "onnx_subadapter") and self.onnx_subadapter is not None:
            return self.onnx_subadapter.predict(image_bgr)

        if self.model is None:
            logger.info("YOLOAdapter: no active model weights; returning empty detections.")
            return []

        h, w = image_bgr.shape[:2]
        try:
            results = self.model.predict(
                source=image_bgr,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False
            )

            detections: List[DetectedBarrier] = []
            if results and len(results) > 0:
                boxes = results[0].boxes
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    xyxy = box.xyxy[0].tolist() # x1, y1, x2, y2

                    x1, y1, x2, y2 = xyxy
                    ymin = max(0.0, min(1.0, y1 / h))
                    xmin = max(0.0, min(1.0, x1 / w))
                    ymax = max(0.0, min(1.0, y2 / h))
                    xmax = max(0.0, min(1.0, x2 / w))

                    # Map YOLO class names
                    names = self.model.names if hasattr(self.model, "names") else {}
                    raw_name = names.get(cls_id, f"class_{cls_id}").lower()

                    # Semantic mapping to AccessGuard classes
                    if "stair" in raw_name or "step" in raw_name:
                        b_type = "stairs"
                    elif "ramp" in raw_name:
                        b_type = "curb_ramp_present"
                    elif "pothole" in raw_name or "crack" in raw_name or "damage" in raw_name:
                        b_type = "surface_damage"
                    elif "bollard" in raw_name or "pole" in raw_name:
                        b_type = "pole_or_bollard"
                    else:
                        b_type = "construction_obstacle"

                    detections.append(
                        DetectedBarrier(
                            barrier_type=b_type,
                            confidence=round(conf, 4),
                            uncertainty_score=round(1.0 - conf, 4),
                            bbox=(round(ymin, 4), round(xmin, 4), round(ymax, 4), round(xmax, 4)),
                            evidence={
                                "source": "ultralytics_yolo",
                                "raw_class": raw_name,
                                "class_id": cls_id,
                                "model_confidence": round(conf, 4)
                            }
                        )
                    )

            return detections
        except Exception as e:
            logger.warning(f"Error during YOLO prediction ({e}); returning empty detections.")
            return []

    def predict_v2(self, image_bgr: np.ndarray, allow_fallback: bool = False, **kwargs):
        """Unified AdapterInferenceResult return for Evidence Fusion."""
        from app.cv.adapters.base_v2 import AdapterInferenceResult, AdapterStatus
        import time

        start_time = time.perf_counter()

        if self._status not in (AdapterStatus.READY, AdapterStatus.DEGRADED):
            if not self.load(allow_fallback=allow_fallback):
                return AdapterInferenceResult(
                    status=self._status,
                    adapter_name=self.name,
                    execution_type="heuristic_fallback" if self._is_mock else "unavailable",
                    error_message=f"Model not initialized ({self._last_error or 'load failed'})",
                    execution_time_ms=round((time.perf_counter() - start_time) * 1000.0, 2),
                    device_used=self._device
                )

        detections = self.predict(image_bgr)
        latency = (time.perf_counter() - start_time) * 1000.0

        exec_type = "heuristic_fallback" if self._is_mock else "model_inference"

        return AdapterInferenceResult(
            status=self._status,
            adapter_name=self.name,
            detections=detections,
            execution_time_ms=round(latency, 2),
            device_used=self._device,
            execution_type=exec_type,
            weight_path=self.model_path if not self._is_mock else None,
            weight_sha256=self._weight_sha256 if not self._is_mock else None,
            upstream_repo="https://github.com/ultralytics/ultralytics"
        )
