# Hybrid Inference Interface Specification
**AccessGuard Hybrid Vision Intelligence (Mission 006)**  
**Feature Branch:** `feature/hybrid-vision-intelligence`  
**Date:** November 2026  
**Status:** COMPLETE — FORMAL INTERFACE CONTRACT

---

## 1. Scope & Design Philosophy

This document defines the formal specification for the **AccessGuard Model Adapter V2** interface. The specification establishes standard lifecycle governance, multi-modal output representations, explicit failure semantics, and memory management across all current and future computer vision models.

---

## 2. Type Definitions & Enums

### 2.1 Adapter Lifecycle Status (`AdapterStatus`)
```python
from enum import Enum

class AdapterStatus(str, Enum):
    NOT_LOADED = "NOT_LOADED"           # Weights not in memory; idle state
    LOADING = "LOADING"                 # Currently initializing or transferring weights to device
    READY = "READY"                     # Model ready for inference; health checks passing
    DEGRADED = "DEGRADED"               # Operational under fallback (e.g., CPU fallback, missing auxiliary models)
    INFERENCE_FAILED = "INFERENCE_FAILED" # Inference error occurred (e.g. CUDA OOM, tensor shape mismatch)
    UNAVAILABLE = "UNAVAILABLE"         # Required weight files, libraries, or licenses missing
```

### 2.2 Output Modalities (`OutputModality`)
```python
class OutputModality(str, Enum):
    BOUNDING_BOX = "BOUNDING_BOX"       # Rectangular boxes (ymin, xmin, ymax, xmax)
    POINT_KEYPOINT = "POINT_KEYPOINT"   # Discrete (x, y) coordinates (RampNet apexes)
    SEMANTIC_MASK = "SEMANTIC_MASK"     # Dense 2D integer or probability arrays (CitySurfaces)
    VERIFICATION = "VERIFICATION"       # Binary correctness verdict & calibrated accuracy (Validator)
```

---

## 3. Data Transfer Objects (DTOs)

### 3.1 Point Keypoints (`PointKeypoint`)
```python
@dataclass
class PointKeypoint:
    x: float                            # Normalized horizontal coordinate [0.0, 1.0] or pixel coordinate
    y: float                            # Normalized vertical coordinate [0.0, 1.0] or pixel coordinate
    confidence: float                   # Detection confidence score [0.0, 1.0]
    uncertainty_score: float            # Uncertainty score [0.0, 1.0]
    label: str                          # Semantic class (e.g. "curb_ramp_present")
    evidence: Dict[str, Any] = field(default_factory=dict)
```

### 3.2 Semantic Mask (`SemanticMask`)
```python
@dataclass
class SemanticMask:
    mask: np.ndarray                    # 2D array of class IDs (uint8) or probability maps
    classes: List[str]                  # List of class names indexed by integer ID
    resolution: Tuple[int, int]         # (height, width)
    roughness_score: float              # Estimated accessibility rolling resistance index [0.0, 1.0]
    evidence: Dict[str, Any] = field(default_factory=dict)
```

### 3.3 Verification Verdict (`VerificationVerdict`)
```python
@dataclass
class VerificationVerdict:
    is_valid: bool                      # True if model endorses candidate label
    verdict: str                        # "correct", "incorrect", or "unsure"
    calibrated_accuracy: float          # Calibrated probability of correctness [0.0, 1.0]
    raw_confidence: float               # Raw softmax / sigmoid confidence score [0.0, 1.0]
    label_type: str                     # Evaluated label category (e.g. "CurbRamp", "Obstacle")
    evidence: Dict[str, Any] = field(default_factory=dict)
```

### 3.4 Inference Result (`AdapterInferenceResult`)
```python
@dataclass
class AdapterInferenceResult:
    status: AdapterStatus
    adapter_name: str
    detections: List[DetectedBarrier] = field(default_factory=list)
    keypoints: List[PointKeypoint] = field(default_factory=list)
    mask: Optional[SemanticMask] = None
    verification: Optional[VerificationVerdict] = None
    execution_time_ms: float = 0.0
    device_used: str = "cpu"
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_successful(self) -> bool:
        """True only if the inference completed successfully without error."""
        return self.status in (AdapterStatus.READY, AdapterStatus.DEGRADED)
```

---

## 4. Abstract Base Class: `ModelAdapterV2`

```python
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Union
import numpy as np

class ModelAdapterV2(ABC):
    """
    Unified abstract interface for AccessGuard Computer Vision Model Adapters.
    Enforces complete lifecycle management, structured health reporting,
    multi-modal inference, and strict failure semantics.
    """

    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version
        self._status: AdapterStatus = AdapterStatus.NOT_LOADED
        self._device: str = "cpu"
        self._inference_count: int = 0
        self._last_latency_ms: float = 0.0
        self._last_error: Optional[str] = None

    @abstractmethod
    def load(self, device: Optional[str] = None) -> bool:
        """
        Loads model weights into memory on the target device.
        Must transition self._status to READY upon success, or UNAVAILABLE / INFERENCE_FAILED on error.
        Returns True if successfully ready, False otherwise.
        """
        pass

    @abstractmethod
    def unload(self) -> bool:
        """
        Frees all loaded weights and tensors from RAM/VRAM.
        Invokes garbage collection and device cache flushing (e.g., torch.cuda.empty_cache()).
        Transitions self._status to NOT_LOADED.
        Returns True if cleanly deallocated.
        """
        pass

    @abstractmethod
    def predict(self, image: Union[np.ndarray, str], **kwargs) -> AdapterInferenceResult:
        """
        Executes model inference on the provided input image.
        MUST catch internal exceptions and return an AdapterInferenceResult with
        status=AdapterStatus.INFERENCE_FAILED and error details on failure.
        NEVER return an empty list masked as successful inference.
        """
        pass

    @abstractmethod
    def health(self) -> Dict[str, Any]:
        """
        Returns structured telemetry including status, device, allocated memory,
        inference count, and last latency.
        """
        pass

    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """
        Returns static metadata: supported modalities, taxonomy, license,
        recommended hardware budgets, and citation details.
        """
        pass
```

---

## 5. Failure Semantics & Error Protocol

1. **Failure vs Zero Detections:**
   - If an image contains zero accessibility barriers and inference succeeds, `predict()` returns:
     `AdapterInferenceResult(status=AdapterStatus.READY, detections=[], keypoints=[])`
   - If model weights are missing or inference encounters a CUDA Out-of-Memory error, `predict()` returns:
     `AdapterInferenceResult(status=AdapterStatus.INFERENCE_FAILED, error_message="CUDA out of memory", detections=[])`
   - Downstream consumers MUST check:
     ```python
     if not result.is_successful:
         raise VisionServiceDegradedError(result.error_message)
     ```
2. **Graceful Fallback:**
   - If CUDA memory allocation fails during `load()`, the adapter may attempt CPU loading and set `status = AdapterStatus.DEGRADED`.
