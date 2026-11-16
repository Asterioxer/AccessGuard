"""
Ultralytics YOLO Model Adapter for AccessGuard.

Provides native PyTorch and Ultralytics YOLOv8/v11 inference for
detecting accessibility obstacles, curb ramps, stairs, and surface damage.
"""

import os
from typing import List, Optional
import numpy as np

from app.cv.base import ModelAdapter, DetectedBarrier, ACCESSIBILITY_CLASSES
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
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.model = None

    def load(self) -> bool:
        """Loads Ultralytics YOLO model from weights file or pretrained baseline."""
        try:
            # Check if ONNX model exists first for pure C++ hardware acceleration without torchvision::nms issues
            onnx_default = "data/models/yolov8n.onnx"
            if self.model_path and self.model_path.endswith(".onnx") and os.path.exists(self.model_path):
                from app.cv.adapters.onnx_adapter import ONNXAdapter
                self.onnx_subadapter = ONNXAdapter(self.model_path)
                self.onnx_subadapter.load()
                self.is_initialized = True
                return True
            elif os.path.exists(onnx_default):
                from app.cv.adapters.onnx_adapter import ONNXAdapter
                self.onnx_subadapter = ONNXAdapter(onnx_default)
                self.onnx_subadapter.load()
                self.is_initialized = True
                return True

            from ultralytics import YOLO
            if self.model_path and os.path.exists(self.model_path):
                self.model = YOLO(self.model_path)
                logger.info(f"Loaded custom YOLO model from {self.model_path}")
            else:
                logger.info("YOLOAdapter initialized with lightweight YOLOv8n baseline.")
                self.model = YOLO("data/models/yolov8n.pt" if os.path.exists("data/models/yolov8n.pt") else "yolov8n.pt")

            self.is_initialized = True
            return True
        except Exception as e:
            logger.warning(f"Could not initialize Ultralytics YOLO ({e}); running in fallback mode.")
            self.is_initialized = True
            return True

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
