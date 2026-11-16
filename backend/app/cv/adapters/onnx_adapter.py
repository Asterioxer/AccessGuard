"""
ONNX Runtime Model Adapter for AccessGuard.

Provides hardware-accelerated, lightweight cross-platform inference
using ONNX Runtime CPU and CUDA execution providers.
"""

import os
import time
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import cv2

from app.cv.base import ModelAdapter, DetectedBarrier, ACCESSIBILITY_CLASSES
from app.core.logging import logger

class ONNXAdapter(ModelAdapter):
    """
    ONNX Runtime model adapter for accessibility barrier detection.
    Supports ONNX models exported from YOLO or PyTorch Faster R-CNN.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        input_size: Tuple[int, int] = (640, 640),
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.45
    ):
        super().__init__(name="onnx_adapter")
        self.model_path = model_path
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.session = None
        self.input_name = None
        self.output_names = []

    def load(self) -> bool:
        """Initializes the ONNX Runtime inference session."""
        try:
            import onnxruntime as ort

            # Select providers: CUDA if available, fallback to CPU
            available_providers = ort.get_available_providers()
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in available_providers else ["CPUExecutionProvider"]

            if self.model_path and os.path.exists(self.model_path):
                self.session = ort.InferenceSession(self.model_path, providers=providers)
                self.input_name = self.session.get_inputs()[0].name
                self.output_names = [o.name for o in self.session.get_outputs()]
                self.is_initialized = True
                logger.info(f"ONNX session loaded from {self.model_path} with providers {self.session.get_providers()}")
                return True
            else:
                # If no weights file specified, create or use fallback
                self.is_initialized = True
                logger.info("ONNXAdapter initialized in test/fallback mode.")
                return True
        except Exception as e:
            logger.warning(f"Failed to initialize ONNXAdapter: {e}")
            self.is_initialized = False
            return False

    def preprocess(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Letterbox image preprocessing to standard square input size (e.g. 640x640).
        Returns normalized float32 tensor of shape (1, 3, H, W) and scaling factors.
        """
        h_orig, w_orig = image_bgr.shape[:2]
        target_w, target_h = self.input_size

        scale = min(target_w / w_orig, target_h / h_orig)
        new_w = int(w_orig * scale)
        new_h = int(h_orig * scale)

        resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)

        dx = (target_w - new_w) // 2
        dy = (target_h - new_h) // 2
        canvas[dy:dy + new_h, dx:dx + new_w] = resized

        # BGR to RGB, HWC to CHW, normalize to [0, 1]
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        tensor = np.expand_dims(tensor, axis=0) # (1, 3, 640, 640)

        return tensor, scale, (dx, dy)

    def predict(self, image_bgr: np.ndarray) -> List[DetectedBarrier]:
        """Runs ONNX inference and post-processes output boxes."""
        if not self.is_initialized:
            if not self.load():
                return []

        h_orig, w_orig = image_bgr.shape[:2]

        if self.session is None or self.model_path is None or not os.path.exists(self.model_path):
            # Fallback to heuristic adapter predictions formatted cleanly
            from app.cv.adapters.heuristic_adapter import HeuristicCVAdapter
            return HeuristicCVAdapter().predict(image_bgr)

        # Preprocess
        tensor, scale, (dx, dy) = self.preprocess(image_bgr)

        # Inference
        outputs = self.session.run(self.output_names, {self.input_name: tensor})
        raw_preds = outputs[0] # Typically shape (1, 84, 8400) or (1, 8400, 84)

        # Post-process standard YOLO ONNX format
        return self._postprocess_yolo_onnx(raw_preds, w_orig, h_orig, scale, dx, dy)

    def _postprocess_yolo_onnx(
        self,
        raw_preds: np.ndarray,
        orig_w: int,
        orig_h: int,
        scale: float,
        dx: int,
        dy: int
    ) -> List[DetectedBarrier]:
        """Decodes raw YOLO ONNX outputs with non-maximum suppression (NMS)."""
        # Shape adjustment if (1, 84, 8400) -> transpose to (1, 8400, 84)
        if raw_preds.ndim == 3 and raw_preds.shape[1] < raw_preds.shape[2]:
            raw_preds = np.transpose(raw_preds, (0, 2, 1))

        preds = raw_preds[0] # (num_boxes, 4 + num_classes)
        boxes = []
        confidences = []
        class_ids = []

        num_classes = preds.shape[1] - 4
        for row in preds:
            scores = row[4:]
            class_id = int(np.argmax(scores))
            conf = float(scores[class_id])
            if conf >= self.confidence_threshold:
                cx, cy, w, h = row[:4]
                # Convert from canvas coordinates back to original image
                x1 = (cx - w / 2.0 - dx) / scale
                y1 = (cy - h / 2.0 - dy) / scale
                x2 = (cx + w / 2.0 - dx) / scale
                y2 = (cy + h / 2.0 - dy) / scale

                boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
                confidences.append(conf)
                class_ids.append(class_id)

        # NMS
        if not boxes:
            return []

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.iou_threshold)
        detections: List[DetectedBarrier] = []

        if len(indices) > 0:
            for idx in np.array(indices).flatten():
                x, y, bw, bh = boxes[idx]
                conf = confidences[idx]
                cid = class_ids[idx]

                # Map class ID to taxonomy
                cat_name = ACCESSIBILITY_CLASSES[cid % len(ACCESSIBILITY_CLASSES)]

                ymin = max(0.0, min(1.0, y / orig_h))
                xmin = max(0.0, min(1.0, x / orig_w))
                ymax = max(0.0, min(1.0, (y + bh) / orig_h))
                xmax = max(0.0, min(1.0, (x + bw) / orig_w))

                detections.append(
                    DetectedBarrier(
                        barrier_type=cat_name,
                        confidence=round(conf, 4),
                        uncertainty_score=round(1.0 - conf, 4),
                        bbox=(ymin, xmin, ymax, xmax),
                        evidence={"source": "onnx_inference", "model": "yolo_onnx", "input_resolution": self.input_size}
                    )
                )

        return detections
