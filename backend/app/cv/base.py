from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np

# Accessibility barrier taxonomy
ACCESSIBILITY_CLASSES = [
    "stairs",
    "curb_ramp_missing",
    "curb_ramp_present",
    "surface_damage",
    "construction_obstacle",
    "narrow_sidewalk",
    "steep_incline",
    "pole_or_bollard"
]

@dataclass
class DetectedBarrier:
    barrier_type: str
    confidence: float # 0.0 to 1.0
    uncertainty_score: float # 0.0 to 1.0
    bbox: Tuple[float, float, float, float] # (ymin, xmin, ymax, xmax) normalized [0.0, 1.0]
    polygon: Optional[List[List[float]]] = None # [[x1, y1], [x2, y2], ...]
    evidence: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_uncertain(self) -> bool:
        """Determines if prediction is borderline or carries high uncertainty."""
        return self.uncertainty_score > 0.35 or (0.45 <= self.confidence < 0.70)
        
    @property
    def is_confirmed(self) -> bool:
        """Obstacles are NOT confirmed unless evidence is high and uncertainty is low."""
        return self.confidence >= 0.85 and self.uncertainty_score <= 0.20

class ModelAdapter(ABC):
    """Abstract base class for interchangeable CV model adapters."""
    
    def __init__(self, name: str):
        self.name = name
        self.is_initialized = False

    @abstractmethod
    def load(self) -> bool:
        """Load weights, pipelines, or configurations."""
        pass

    @abstractmethod
    def predict(self, image_bgr: np.ndarray) -> List[DetectedBarrier]:
        """
        Run inference on a single BGR image array.
        Returns list of DetectedBarrier objects with bounding boxes, confidence, uncertainty, and evidence.
        """
        pass

    def predict_batch(self, images: List[np.ndarray]) -> List[List[DetectedBarrier]]:
        """Default batch inference calling predict iteratively or vectorially."""
        return [self.predict(img) for img in images]
