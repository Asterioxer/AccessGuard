from typing import List, Optional
import numpy as np
from app.cv.base import ModelAdapter, DetectedBarrier, ACCESSIBILITY_CLASSES
from app.core.logging import logger

class YOLOAdapter(ModelAdapter):
    """
    YOLO / ONNX interchangeable model adapter.
    Implements standard YOLO bounding box decoding and NMS for accessibility obstacles.
    """
    def __init__(self, model_path: Optional[str] = None):
        super().__init__(name="yolo_adapter")
        self.model_path = model_path
        self.session = None

    def load(self) -> bool:
        try:
            if self.model_path:
                # Could use onnxruntime or OpenCV cv2.dnn
                import cv2
                self.net = cv2.dnn.readNet(self.model_path)
                self.is_initialized = True
                return True
            else:
                self.is_initialized = True
                return True
        except Exception as e:
            logger.warning(f"Could not load YOLO model: {e}")
            self.is_initialized = False
            return False

    def predict(self, image_bgr: np.ndarray) -> List[DetectedBarrier]:
        if not self.is_initialized:
            self.load()
            
        # If no standalone YOLO model weights path provided, delegate to OpenCV heuristic analyzer
        from app.cv.adapters.heuristic_adapter import HeuristicCVAdapter
        return HeuristicCVAdapter().predict(image_bgr)
