import os
import hashlib
from typing import List, Optional
import numpy as np

from app.cv.base import ModelAdapter, DetectedBarrier, ACCESSIBILITY_CLASSES
from app.cv.adapters.base_v2 import AdapterStatus, AdapterInferenceResult
from app.core.logging import logger


class TorchvisionAdapter(ModelAdapter):
    """
    PyTorch and Torchvision deep learning adapter for object detection and instance segmentation.
    Supports Mask R-CNN / Faster R-CNN architectures with custom accessibility classification head.
    Truthfully reports lifecycle status, enforces genuine weights requirement under strict mode,
    and isolates inference exceptions.
    """

    def __init__(self, weights_path: Optional[str] = None, device: str = "auto"):
        super().__init__(name="torchvision_adapter")
        self.version = "1.2.0"
        self.weights_path = weights_path
        self.device_str = device
        self.model = None
        self.torch = None
        self.transforms = None
        self._status: AdapterStatus = AdapterStatus.NOT_LOADED
        self._weight_sha256: Optional[str] = None
        self._is_mock: bool = False
        self._last_error: Optional[str] = None
        self.is_initialized = False

    def _is_oom_exception(self, exc: Exception) -> bool:
        """Determines if an exception was caused by CUDA out of memory or host memory exhaustion."""
        msg = str(exc).lower()
        name = type(exc).__name__.lower()
        return any(tok in msg or tok in name for tok in (
            "out of memory",
            "cuda oom",
            "cudadrivererror",
            "resourceexhausted",
            "memoryerror",
            "cuda error: out of memory",
            "default cpu allocator: can't allocate memory"
        ))

    def load(self, allow_fallback: bool = False) -> bool:
        """
        Loads PyTorch model from weights file.
        Under STRICT_GENUINE (allow_fallback=False), missing weights reports UNAVAILABLE.
        """
        try:
            import torch
            import torchvision
            from torchvision import transforms

            self.torch = torch
            if self.device_str == "auto":
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(self.device_str)

            self.transforms = transforms.Compose([
                transforms.ToTensor(),
            ])

            if self.weights_path and os.path.exists(self.weights_path):
                hasher = hashlib.sha256()
                with open(self.weights_path, "rb") as f:
                    while chunk := f.read(65536):
                        hasher.update(chunk)
                self._weight_sha256 = hasher.hexdigest()

                self.model = torch.load(self.weights_path, map_location=self.device, weights_only=True)
                self.model.eval()
                self.is_initialized = True
                self._status = AdapterStatus.READY
                self._is_mock = False
                self._last_error = None
                logger.info(f"Loaded Torchvision model from {self.weights_path} on {self.device} (sha256: {self._weight_sha256[:12]}...)")
                return True
            else:
                self.model = None
                self._weight_sha256 = None
                if allow_fallback:
                    self.is_initialized = True
                    self._is_mock = True
                    self._status = AdapterStatus.DEGRADED
                    self._last_error = "Torchvision running in heuristic fallback mode."
                    logger.info("Torchvision weights not found; operating in DEGRADED fallback mode.")
                    return True
                else:
                    self.is_initialized = False
                    self._is_mock = False
                    self._status = AdapterStatus.UNAVAILABLE
                    self._last_error = f"UNAVAILABLE: MODEL_WEIGHTS_NOT_FOUND ({self.weights_path or 'No weights path provided'})"
                    logger.info(f"Torchvision weights not found; marked UNAVAILABLE: {self._last_error}")
                    return False
        except Exception as e:
            logger.warning(f"Failed to load Torchvision model: {e}")
            self.model = None
            self.is_initialized = False
            self._is_mock = False
            self._status = AdapterStatus.UNAVAILABLE
            self._last_error = f"Torchvision model load error: {e}"
            return False

    def predict_v2(self, image_bgr: np.ndarray, allow_fallback: bool = False, **kwargs) -> AdapterInferenceResult:
        """
        Unified AdapterInferenceResult return for evidence fusion and orchestration.
        Strictly distinguishes UNAVAILABLE, DEGRADED, READY, and INFERENCE_FAILED.
        """
        import time
        start_time = time.perf_counter()
        device_name = str(getattr(self, "device", "cpu"))

        # 1. Input validation
        if image_bgr is None or not isinstance(image_bgr, np.ndarray) or image_bgr.size == 0 or image_bgr.ndim not in (2, 3):
            err_msg = f"Invalid image input: expected non-empty numpy ndarray, got {type(image_bgr)}"
            logger.error(f"TorchvisionAdapter input validation failed: {err_msg}")
            return AdapterInferenceResult.create_failed(
                adapter_name=self.name,
                error_message=err_msg,
                status=AdapterStatus.INFERENCE_FAILED,
                error_code="INVALID_IMAGE_INPUT",
                execution_time_ms=round((time.perf_counter() - start_time) * 1000.0, 2),
                device_used=device_name,
                weight_path=self.weights_path if not self._is_mock else None,
                weight_sha256=self._weight_sha256 if not self._is_mock else None,
                model_version=self.version,
                upstream_repo="https://github.com/pytorch/vision"
            )

        # 2. Lifecycle and readiness verification
        if self._status not in (AdapterStatus.READY, AdapterStatus.DEGRADED):
            if not self.load(allow_fallback=allow_fallback):
                return AdapterInferenceResult.create_unavailable(
                    adapter_name=self.name,
                    error_message=f"Model not initialized ({self._last_error or 'load failed'})",
                    status=self._status,
                    error_code="MODEL_WEIGHTS_NOT_FOUND",
                    execution_time_ms=round((time.perf_counter() - start_time) * 1000.0, 2),
                    device_used=device_name,
                    model_version=self.version,
                    upstream_repo="https://github.com/pytorch/vision"
                )

        # 3. Explicit development fallback branch
        if self._is_mock or self._status == AdapterStatus.DEGRADED:
            latency = (time.perf_counter() - start_time) * 1000.0
            return AdapterInferenceResult(
                status=AdapterStatus.DEGRADED,
                adapter_name=self.name,
                detections=[],
                execution_time_ms=round(latency, 2),
                device_used=device_name,
                execution_type="heuristic_fallback",
                execution_succeeded=True,
                execution_attempted=True,
                weight_path=None,
                weight_sha256=None,
                model_identifier=self.name,
                model_version=self.version,
                upstream_repo="https://github.com/pytorch/vision",
                metadata={"is_fallback": True, "output_count": 0}
            )

        # 4. Genuine model inference execution
        if self.model is not None:
            try:
                import torch
                import cv2

                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                input_tensor = self.transforms(image_rgb).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    predictions = self.model(input_tensor)

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

                latency = (time.perf_counter() - start_time) * 1000.0
                return AdapterInferenceResult(
                    status=AdapterStatus.READY,
                    adapter_name=self.name,
                    detections=results,
                    execution_time_ms=round(latency, 2),
                    device_used=device_name,
                    execution_type="model_inference",
                    execution_succeeded=True,
                    execution_attempted=True,
                    weight_path=self.weights_path,
                    weight_sha256=self._weight_sha256,
                    model_identifier=self.name,
                    model_version=self.version,
                    upstream_repo="https://github.com/pytorch/vision",
                    metadata={"output_count": len(results)}
                )
            except Exception as e:
                is_oom = self._is_oom_exception(e)
                status = AdapterStatus.RESOURCE_EXHAUSTED if is_oom else AdapterStatus.INFERENCE_FAILED
                err_code = "RESOURCE_EXHAUSTED" if is_oom else "INFERENCE_FAILED"
                err_msg = f"Torchvision inference exception: {type(e).__name__}: {e}"
                logger.error(err_msg)
                return AdapterInferenceResult.create_failed(
                    adapter_name=self.name,
                    error_message=err_msg,
                    status=status,
                    error_code=err_code,
                    execution_time_ms=round((time.perf_counter() - start_time) * 1000.0, 2),
                    device_used=device_name,
                    weight_path=self.weights_path,
                    weight_sha256=self._weight_sha256,
                    model_version=self.version,
                    upstream_repo="https://github.com/pytorch/vision"
                )

        return AdapterInferenceResult.create_unavailable(
            adapter_name=self.name,
            error_message="Model weights not available.",
            status=AdapterStatus.UNAVAILABLE,
            error_code="MODEL_WEIGHTS_NOT_FOUND",
            execution_time_ms=round((time.perf_counter() - start_time) * 1000.0, 2),
            device_used=device_name,
            model_version=self.version,
            upstream_repo="https://github.com/pytorch/vision"
        )

    def predict(self, image_bgr: np.ndarray) -> List[DetectedBarrier]:
        """Runs inference on a single BGR image array (legacy interface)."""
        res = self.predict_v2(image_bgr, allow_fallback=False)
        return res.detections if res.is_successful else []
