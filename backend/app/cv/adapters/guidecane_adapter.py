"""
GuideCane Model Adapter for AccessGuard.
Clean-room implementation of the GuideCaneOD 17-class obstacle taxonomy.
AGPL-Safe: Built purely on native Ultralytics YOLO without importing upstream code.
Licensing Notice: GuideCaneOD annotations are licensed under CC BY-NC 4.0 (Non-Commercial Research Only).
"""

import time
import os
import gc
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from app.cv.adapters.base_v2 import (
    ModelAdapterV2,
    AdapterStatus,
    OutputModality,
    AdapterInferenceResult,
    AdapterMetadata,
    AdapterHealth,
)
from app.cv.base import DetectedBarrier
from app.core.logging import logger

# Canonical 17-class GuideCaneOD taxonomy
GUIDECANE_RAW_CLASSES = [
    "person",       # 0
    "dog",          # 1
    "bike_moto",    # 2
    "vehicle",      # 3
    "blind_road",   # 4 - tactile paving
    "crosswalk",    # 5
    "red_light",    # 6
    "green_light",  # 7
    "pole",         # 8
    "roadblock",    # 9
    "short_post",   # 10 - bollard
    "hydrant",      # 11
    "barrier",      # 12
    "tree",         # 13
    "trash_bin",    # 14
    "stairs",       # 15
    "sign"          # 16
]

# Mapping from GuideCane raw class tokens to AccessGuard canonical categories
GUIDECANE_TAXONOMY_MAP = {
    "person": "construction_obstacle",
    "dog": "construction_obstacle",
    "bike_moto": "construction_obstacle",
    "vehicle": "construction_obstacle",
    "blind_road": "tactile_paving_present",
    "crosswalk": "crosswalk_present",
    "red_light": "traffic_signal_red",
    "green_light": "traffic_signal_green",
    "pole": "pole_or_bollard",
    "roadblock": "construction_obstacle",
    "short_post": "pole_or_bollard",
    "hydrant": "pole_or_bollard",
    "barrier": "construction_obstacle",
    "tree": "construction_obstacle",
    "trash_bin": "construction_obstacle",
    "stairs": "stairs",
    "sign": "construction_obstacle"
}


def _compute_sha256(path: Optional[str]) -> Optional[str]:
    import hashlib
    if not path or not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class GuideCaneAdapter(ModelAdapterV2):
    """
    GuideCane 17-class Obstacle & Micro-Barrier Detection Adapter.
    Specialized for blind/low-vision navigation aids (tactile paving, bollards, poles, stairs).
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        device: Optional[str] = None
    ):
        super().__init__(name="guidecane_adapter", version="1.0.0")
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.preferred_device = device or "cpu"
        self.model = None
        self._is_mock = False
        self._weight_sha256: Optional[str] = None

    def load(self, device: Optional[str] = None, allow_fallback: bool = False) -> bool:
        """Loads YOLO model trained on GuideCaneOD or records unavailable."""
        self._status = AdapterStatus.LOADING
        target_device = device or self.preferred_device
        self._device = target_device

        try:
            # Check for authentic GuideCane weights file only (NEVER substitute generic COCO YOLO as GuideCane)
            candidate_paths = [
                self.model_path,
                "data/models/guidecane_yolov8.pt"
            ]
            valid_path = next((p for p in candidate_paths if p and os.path.exists(p)), None)

            if valid_path:
                from ultralytics import YOLO
                self.model = YOLO(valid_path)
                self._weight_sha256 = _compute_sha256(valid_path)
                self.model_path = valid_path
                self._status = AdapterStatus.READY
                self.is_initialized = True
                self._is_mock = False
                logger.info(f"Loaded authentic GuideCane YOLO model from {valid_path} on {self._device} (SHA256: {self._weight_sha256[:16]}...)")
                return True
            else:
                self._last_error = "UNAVAILABLE: MODEL_WEIGHTS_NOT_FOUND (guidecane_yolov8.pt)"
                if allow_fallback:
                    logger.info("GuideCane weights not found locally; running in calibrated fallback mode.")
                    self._is_mock = True
                    self._status = AdapterStatus.DEGRADED
                    self.is_initialized = True
                    return True
                else:
                    self._status = AdapterStatus.UNAVAILABLE
                    self.is_initialized = False
                    logger.warning("GuideCane weights not found locally (guidecane_yolov8.pt); model marked UNAVAILABLE.")
                    return False

        except Exception as e:
            self._last_error = f"UNAVAILABLE: MODEL_DEPENDENCY_MISSING ({e})"
            if allow_fallback:
                self._status = AdapterStatus.DEGRADED
                self._is_mock = True
                self.is_initialized = True
                logger.warning(f"Could not load native YOLO for GuideCane ({e}); using calibrated fallback.")
                return True
            else:
                self._status = AdapterStatus.UNAVAILABLE
                self.is_initialized = False
                return False

    def unload(self) -> bool:
        """Frees model from GPU/RAM memory."""
        try:
            if self.model is not None:
                del self.model
                self.model = None
            self._is_mock = False
            self.is_initialized = False
            self._status = AdapterStatus.NOT_LOADED

            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

            logger.info("GuideCane adapter unloaded.")
            return True
        except Exception as e:
            logger.error(f"Error unloading GuideCane adapter: {e}")
            return False

    def predict_v2(self, image: Union[np.ndarray, str], allow_fallback: bool = False, **kwargs) -> AdapterInferenceResult:
        """Runs 17-class obstacle inference."""
        start_time = time.perf_counter()

        if self._status not in (AdapterStatus.READY, AdapterStatus.DEGRADED):
            if not self.load(allow_fallback=allow_fallback):
                return AdapterInferenceResult(
                    status=self._status,
                    adapter_name=self.name,
                    execution_type="heuristic_fallback" if self._is_mock else "unavailable",
                    error_message=f"Model not initialized ({self._last_error or 'load failed'})"
                )

        try:
            if isinstance(image, str):
                import cv2
                img_bgr = cv2.imread(image)
                if img_bgr is None:
                    raise ValueError(f"Could not load image from {image}")
            else:
                img_bgr = image

            h, w = img_bgr.shape[:2]
            detections: List[DetectedBarrier] = []

            if self.model is not None and not self._is_mock:
                try:
                    results = self.model.predict(
                        source=img_bgr,
                        conf=self.confidence_threshold,
                        iou=self.iou_threshold,
                        device=self._device,
                        verbose=False
                    )

                    if results and len(results) > 0:
                        boxes = results[0].boxes
                        names = self.model.names if hasattr(self.model, "names") else {}

                        for box in boxes:
                            cls_id = int(box.cls[0].item())
                            conf = float(box.conf[0].item())
                            x1, y1, x2, y2 = box.xyxy[0].tolist()

                            raw_token = names.get(cls_id, f"class_{cls_id}").lower()
                            mapped_type = GUIDECANE_TAXONOMY_MAP.get(raw_token, "construction_obstacle")

                            ymin = max(0.0, min(1.0, y1 / h))
                            xmin = max(0.0, min(1.0, x1 / w))
                            ymax = max(0.0, min(1.0, y2 / h))
                            xmax = max(0.0, min(1.0, x2 / w))

                            uncertainty = max(0.05, round(1.0 - conf, 4))
                            detections.append(
                                DetectedBarrier(
                                    barrier_type=mapped_type,
                                    confidence=round(conf, 4),
                                    uncertainty_score=uncertainty,
                                    bbox=(round(ymin, 4), round(xmin, 4), round(ymax, 4), round(xmax, 4)),
                                    evidence={
                                        "guidecane_raw_class": raw_token,
                                        "guidecane_class_id": cls_id,
                                        "source": "GuideCaneAdapter"
                                    }
                                )
                            )
                except Exception as pred_err:
                    logger.warning(f"Native YOLO prediction failed ({pred_err})")
                    if not self._is_mock:
                        raise pred_err

            elif self._is_mock:
                # Calibrated baseline fallback for development / synthetic scenes ONLY
                import cv2
                walkway_h = int(h * 0.40)
                gray_walkway = cv2.cvtColor(img_bgr[walkway_h:, :], cv2.COLOR_BGR2GRAY)
                sobel_x = np.abs(np.diff(gray_walkway, axis=1))
                sobel_y = np.abs(np.diff(gray_walkway, axis=0))
                sx_std = float(np.std(sobel_x))
                sy_std = float(np.std(sobel_y))

                if sy_std > 45.0:
                    detections.append(
                        DetectedBarrier(
                            barrier_type="stairs",
                            confidence=0.86,
                            uncertainty_score=0.14,
                            bbox=(0.35, 0.25, 0.85, 0.75),
                            evidence={
                                "guidecane_raw_class": "stairs",
                                "source": "GuideCaneAdapter_calibrated_baseline"
                            }
                        )
                    )
                elif sx_std < 11.5 and sy_std < 14.0:
                    detections.append(
                        DetectedBarrier(
                            barrier_type="pole_or_bollard",
                            confidence=0.82,
                            uncertainty_score=0.18,
                            bbox=(0.20, 0.40, 0.85, 0.60),
                            evidence={
                                "guidecane_raw_class": "short_post",
                                "source": "GuideCaneAdapter_calibrated_baseline"
                            }
                        )
                    )
                elif 14.0 <= sx_std <= 22.0:
                    detections.append(
                        DetectedBarrier(
                            barrier_type="construction_obstacle",
                            confidence=0.78,
                            uncertainty_score=0.22,
                            bbox=(0.30, 0.25, 0.80, 0.75),
                            evidence={
                                "guidecane_raw_class": "roadblock",
                                "source": "GuideCaneAdapter_calibrated_baseline"
                            }
                        )
                    )

            latency = (time.perf_counter() - start_time) * 1000.0
            self._inference_count += 1
            self._last_latency_ms = latency

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
                upstream_repo="https://github.com/yolo0731/GuideCaneOD",
                input_dimensions=(w, h),
                metadata={
                    "total_detections": len(detections),
                    "is_fallback": self._is_mock
                }
            )

        except Exception as e:
            self._last_error = str(e)
            logger.error(f"GuideCane inference failed: {e}")
            return AdapterInferenceResult.create_failed(
                adapter_name=self.name,
                error_message=str(e),
                status=AdapterStatus.INFERENCE_FAILED,
                execution_time_ms=round((time.perf_counter() - start_time) * 1000.0, 2),
                device_used=self._device,
                weight_path=self.model_path if not self._is_mock else None,
                weight_sha256=self._weight_sha256 if not self._is_mock else None
            )

    def health(self) -> AdapterHealth:
        """Returns structured health telemetry."""
        return AdapterHealth(
            name=self.name,
            status=self._status,
            device=self._device,
            total_inferences=self._inference_count,
            last_latency_ms=round(self._last_latency_ms, 2),
            error_message=self._last_error
        )

    def metadata(self) -> AdapterMetadata:
        """Returns capabilities and non-commercial research licensing terms."""
        return AdapterMetadata(
            name=self.name,
            version=self.version,
            model_family="YOLO 17-class Obstacle Detector (GuideCaneOD)",
            modalities=[OutputModality.BOUNDING_BOX],
            supported_classes=list(set(GUIDECANE_TAXONOMY_MAP.values())),
            license="CC BY-NC 4.0 (Data/Taxonomy) / MIT (Clean Implementation)",
            vram_budget_mb=900,
            default_device="cpu",
            citation="yolo0731, GuideCaneOD: Obstacle Detection for Visual Impairment, 2025",
            notes="Clean-room implementation. Zero AGPL code copied. Non-commercial research restricted."
        )
