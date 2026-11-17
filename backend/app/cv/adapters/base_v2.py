"""
Unified Model Adapter Interface V2 for AccessGuard.
Provides lifecycle management (load, unload, health), multi-modal output support
(bounding boxes, keypoints, semantic masks, verifications), and strict failure semantics.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import time
import numpy as np

from app.cv.base import DetectedBarrier, ModelAdapter


class AdapterStatus(str, Enum):
    """Lifecycle and operational status of a model adapter."""
    NOT_LOADED = "NOT_LOADED"           # Weights not in memory
    LOADING = "LOADING"                 # Currently initializing or transferring weights
    READY = "READY"                     # Ready for inference; health checks pass
    DEGRADED = "DEGRADED"               # Operational under fallback (e.g. CPU or stub)
    INFERENCE_FAILED = "INFERENCE_FAILED" # Inference error (e.g. CUDA OOM, corrupt tensor)
    UNAVAILABLE = "UNAVAILABLE"         # Required weights, libraries, or licenses missing
    FAILED = "FAILED"                   # Fatal initialization or runtime failure


class OutputModality(str, Enum):
    """Output representation produced by the adapter."""
    BOUNDING_BOX = "BOUNDING_BOX"       # Rectangular boxes (ymin, xmin, ymax, xmax)
    POINT_KEYPOINT = "POINT_KEYPOINT"   # Discrete (x, y) coordinates (RampNet apexes)
    SEMANTIC_MASK = "SEMANTIC_MASK"     # Dense 2D integer or probability arrays (CitySurfaces)
    VERIFICATION = "VERIFICATION"       # Binary correctness verdict & calibrated accuracy (Validator)


@dataclass
class PointKeypoint:
    """Discrete point coordinate representation (e.g. curb ramp apexes)."""
    x: float                            # Normalized horizontal coordinate [0.0, 1.0]
    y: float                            # Normalized vertical coordinate [0.0, 1.0]
    confidence: float                   # Detection confidence score [0.0, 1.0]
    uncertainty_score: float = 0.1      # Uncertainty score [0.0, 1.0]
    label: str = "curb_ramp_present"    # Semantic class
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_detected_barrier(self, box_radius: float = 0.04) -> DetectedBarrier:
        """Converts keypoint to a synthetic DetectedBarrier bounding box for legacy consumers."""
        ymin = max(0.0, self.y - box_radius)
        xmin = max(0.0, self.x - box_radius)
        ymax = min(1.0, self.y + box_radius)
        xmax = min(1.0, self.x + box_radius)
        return DetectedBarrier(
            barrier_type=self.label,
            confidence=round(self.confidence, 4),
            uncertainty_score=round(self.uncertainty_score, 4),
            bbox=(round(ymin, 4), round(xmin, 4), round(ymax, 4), round(xmax, 4)),
            evidence={
                **self.evidence,
                "is_keypoint": True,
                "keypoint_center": (round(self.x, 4), round(self.y, 4))
            }
        )


@dataclass
class SemanticMask:
    """Dense semantic segmentation mask (e.g. sidewalk surface materials)."""
    mask: np.ndarray                    # 2D array of class IDs (uint8)
    classes: List[str]                  # List of class names indexed by integer ID
    resolution: Tuple[int, int]         # (height, width)
    roughness_score: float = 0.0        # Estimated surface roughness index [0.0, 1.0]
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationVerdict:
    """Second-stage verification assessment of a candidate label."""
    is_valid: bool                      # True if model endorses candidate label
    verdict: str                        # "correct", "incorrect", or "unsure"
    calibrated_accuracy: float          # Calibrated probability of correctness [0.0, 1.0]
    raw_confidence: float               # Raw softmax or sigmoid confidence score [0.0, 1.0]
    label_type: str                     # Evaluated label category
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AdapterInferenceResult:
    """Unified inference output container across all adapter types."""
    status: AdapterStatus
    adapter_name: str
    detections: List[DetectedBarrier] = field(default_factory=list)
    keypoints: List[PointKeypoint] = field(default_factory=list)
    mask: Optional[SemanticMask] = None
    verification: Optional[VerificationVerdict] = None
    execution_time_ms: float = 0.0
    device_used: str = "cpu"
    execution_type: str = "model_inference" # "model_inference" or "heuristic_fallback"
    weight_path: Optional[str] = None
    weight_sha256: Optional[str] = None
    upstream_repo: Optional[str] = None
    input_dimensions: Optional[Tuple[int, int]] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_successful(self) -> bool:
        """True only if the inference completed successfully without error."""
        return self.status in (AdapterStatus.READY, AdapterStatus.DEGRADED)


@dataclass
class AdapterMetadata:
    """Static metadata describing an adapter's capabilities and origin."""
    name: str
    version: str
    model_family: str
    modalities: List[OutputModality]
    supported_classes: List[str]
    license: str
    upstream_repo: Optional[str] = None
    weight_sha256: Optional[str] = None
    precision: str = "fp32"
    vram_budget_mb: int = 1000
    default_device: str = "cpu"
    citation: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class AdapterHealth:
    """Runtime telemetry and health status of an adapter."""
    name: str
    status: AdapterStatus
    device: str
    vram_allocated_mb: float = 0.0
    ram_used_mb: float = 0.0
    total_inferences: int = 0
    last_latency_ms: float = 0.0
    error_message: Optional[str] = None


class ModelAdapterV2(ModelAdapter, ABC):
    """
    Unified abstract interface for AccessGuard Computer Vision Model Adapters.
    Subclasses must implement load, unload, predict, health, and metadata.
    Backward-compatible with legacy ModelAdapter predict(image_bgr).
    """

    def __init__(self, name: str, version: str = "1.0.0"):
        super().__init__(name=name)
        self.version = version
        self._status: AdapterStatus = AdapterStatus.NOT_LOADED
        self._device: str = "cpu"
        self._inference_count: int = 0
        self._last_latency_ms: float = 0.0
        self._last_error: Optional[str] = None

    @property
    def status(self) -> AdapterStatus:
        return self._status

    @abstractmethod
    def load(self, device: Optional[str] = None) -> bool:
        """
        Loads model weights into memory on the target device.
        Must transition self._status to READY on success, or DEGRADED/UNAVAILABLE on error.
        """
        pass

    @abstractmethod
    def unload(self) -> bool:
        """
        Frees loaded weights and tensors from memory.
        Transitions self._status to NOT_LOADED.
        """
        pass

    @abstractmethod
    def predict_v2(self, image: Union[np.ndarray, str], **kwargs) -> AdapterInferenceResult:
        """
        Executes inference and returns a comprehensive AdapterInferenceResult.
        MUST catch internal errors and report status=INFERENCE_FAILED on failure.
        """
        pass

    def predict(self, image_bgr: np.ndarray) -> List[DetectedBarrier]:
        """
        Legacy ModelAdapter compatibility method.
        Executes predict_v2 and returns list of DetectedBarrier.
        If keypoints were detected, converts them to synthetic DetectedBarrier bounding boxes.
        """
        res = self.predict_v2(image_bgr)
        if not res.is_successful:
            return []
        
        all_detections = list(res.detections)
        if res.keypoints and not res.detections:
            all_detections.extend([kp.to_detected_barrier() for kp in res.keypoints])
        return all_detections

    @abstractmethod
    def health(self) -> AdapterHealth:
        """Returns structured health telemetry."""
        pass

    @abstractmethod
    def metadata(self) -> AdapterMetadata:
        """Returns static capabilities, taxonomy, and licensing metadata."""
        pass
