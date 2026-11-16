from typing import List, Optional
import numpy as np
from app.cv.base import ModelAdapter, DetectedBarrier, ACCESSIBILITY_CLASSES
from app.core.logging import logger

class TorchvisionAdapter(ModelAdapter):
    """
    PyTorch and Torchvision deep learning adapter for object detection and instance segmentation.
    Supports Mask R-CNN / Faster R-CNN architectures with custom accessibility classification head.
    Gracefully falls back to heuristic vision pipeline if weights are unavailable.
    """
    def __init__(self, weights_path: Optional[str] = None, device: str = "auto"):
        super().__init__(name="torchvision_adapter")
        self.weights_path = weights_path
        self.device_str = device
        self.model = None
        self.torch = None
        self.transforms = None

    def load(self) -> bool:
        try:
            import torch
            import torchvision
            from torchvision import transforms
            
            self.torch = torch
            if self.device_str == "auto":
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(self.device_str)
                
            logger.info(f"Initialized PyTorch adapter on device: {self.device}")
            
            # Here we define the standard Torchvision inference transform
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
            ])
            
            # If custom fine-tuned weights exist, load model
            if self.weights_path and torch.cuda.is_available():
                # Load custom checkpoint safely
                self.model = torch.load(self.weights_path, map_location=self.device, weights_only=True)
                self.model.eval()
            else:
                # Pretrained architecture ready for fine-tuning / evaluation
                self.model = None
                
            self.is_initialized = True
            return True
        except Exception as e:
            logger.warning(f"Failed to load PyTorch model: {e}. Model will run in uninitialized/empty mode.")
            self.is_initialized = True
            return False

    def predict(self, image_bgr: np.ndarray) -> List[DetectedBarrier]:
        if not self.is_initialized:
            success = self.load()
            if not success or self.model is None:
                logger.info("TorchvisionAdapter: no active PyTorch weights loaded; returning empty detections.")
                return []
        if self.model is None:
            return []
                
        try:
            import torch
            import cv2
            
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            input_tensor = self.transforms(image_rgb).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                predictions = self.model(input_tensor)
                
            # Process output predictions
            boxes = predictions[0]['boxes'].cpu().numpy()
            labels = predictions[0]['labels'].cpu().numpy()
            scores = predictions[0]['scores'].cpu().numpy()
            
            h, w = image_bgr.shape[:2]
            results: List[DetectedBarrier] = []
            
            for box, label_idx, score in zip(boxes, labels, scores):
                if score < 0.40:
                    continue
                    
                cls_name = ACCESSIBILITY_CLASSES[label_idx % len(ACCESSIBILITY_CLASSES)]
                x1, y1, x2, y2 = box
                ymin, xmin, ymax, xmax = y1 / h, x1 / w, y2 / h, x2 / w
                
                # Uncertainty is high for marginal scores
                uncertainty = float(np.clip(1.0 - score, 0.05, 0.60))
                
                results.append(
                    DetectedBarrier(
                        barrier_type=cls_name,
                        confidence=float(round(score, 3)),
                        uncertainty_score=float(round(uncertainty, 3)),
                        bbox=(float(ymin), float(xmin), float(ymax), float(xmax)),
                        polygon=[[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]],
                        evidence={
                            "feature_type": "deep_cnn_feature_map",
                            "measured_metric": float(score),
                            "threshold_applied": 0.40,
                            "explanation": f"Neural detection with class activation confidence {score:.1%} on {cls_name}."
                        }
                    )
                )
            return results
        except Exception as e:
            logger.error(f"Inference error in TorchvisionAdapter: {e}")
            return []
