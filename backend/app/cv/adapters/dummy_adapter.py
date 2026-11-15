from typing import List
import numpy as np
from app.cv.base import ModelAdapter, DetectedBarrier

class DummyAdapter(ModelAdapter):
    """
    Deterministic adapter used in unit testing and verification benchmarks.
    Simulates detection of stairs and surface damage with configurable behavior.
    """
    def __init__(self, mode: str = "standard"):
        super().__init__(name="dummy_adapter")
        self.mode = mode
        self.is_initialized = True

    def load(self) -> bool:
        self.is_initialized = True
        return True

    def predict(self, image_bgr: np.ndarray) -> List[DetectedBarrier]:
        if not self.is_initialized:
            self.load()

        if self.mode == "empty":
            return []

        if self.mode == "uncertain":
            # Borderline confidence and high uncertainty to test uncertainty flagging
            return [
                DetectedBarrier(
                    barrier_type="stairs",
                    confidence=0.55,
                    uncertainty_score=0.48,
                    bbox=(0.4, 0.2, 0.8, 0.8),
                    polygon=[[0.2, 0.4], [0.8, 0.4], [0.8, 0.8], [0.2, 0.8]],
                    evidence={
                        "feature_type": "edge_rungs",
                        "measured_metric": 2.0,
                        "threshold_applied": 3.0,
                        "explanation": "Weak periodic horizontal edges detected; ambiguous step geometry."
                    }
                )
            ]

        # Standard test detection
        return [
            DetectedBarrier(
                barrier_type="stairs",
                confidence=0.92,
                uncertainty_score=0.12,
                bbox=(0.35, 0.15, 0.75, 0.85),
                polygon=[[0.15, 0.35], [0.85, 0.35], [0.85, 0.75], [0.15, 0.75]],
                evidence={
                    "feature_type": "edge_rungs",
                    "measured_metric": 5.0,
                    "threshold_applied": 3.0,
                    "explanation": "5 distinct parallel horizontal step rungs detected with high contrast."
                }
            ),
            DetectedBarrier(
                barrier_type="curb_ramp_missing",
                confidence=0.88,
                uncertainty_score=0.15,
                bbox=(0.70, 0.05, 0.95, 0.45),
                polygon=[[0.05, 0.70], [0.45, 0.70], [0.45, 0.95], [0.05, 0.95]],
                evidence={
                    "feature_type": "vertical_curb_drop",
                    "measured_metric": 18.5,
                    "threshold_applied": 5.0,
                    "explanation": "Abrupt 18.5cm vertical curb drop without flared slope or detectable tactile paving."
                }
            )
        ]
