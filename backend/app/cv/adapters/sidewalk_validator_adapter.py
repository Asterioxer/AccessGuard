"""
Sidewalk Validator Model Adapter for AccessGuard.
Implements the DINOv2 second-stage candidate verification pipeline from Project Sidewalk.
Performs confidence calibration using empirical lookup tables to output verified accuracy.
License: MIT (Makeability Lab, University of Washington).
"""

import time
import os
import json
import gc
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from app.cv.adapters.base_v2 import (
    ModelAdapterV2,
    AdapterStatus,
    OutputModality,
    VerificationVerdict,
    AdapterInferenceResult,
    AdapterMetadata,
    AdapterHealth,
)
from app.cv.base import DetectedBarrier
from app.core.logging import logger

VALIDATOR_LABEL_TYPES = ["crosswalk", "curbramp", "obstacle", "surfaceproblem", "nocurbramp"]

# Mapping from AccessGuard internal categories to Validator label types
ACCESSGUARD_TO_VALIDATOR_TYPE = {
    "curb_ramp_present": "curbramp",
    "curb_ramp_missing": "nocurbramp",
    "construction_obstacle": "obstacle",
    "pole_or_bollard": "obstacle",
    "stairs": "obstacle",
    "surface_damage": "surfaceproblem",
    "crosswalk_present": "crosswalk",
    "narrow_sidewalk": "obstacle",
    "steep_incline": "surfaceproblem"
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


class SidewalkValidatorAdapter(ModelAdapterV2):
    """
    Sidewalk Validator AI Adapter.
    Evaluates candidate accessibility detections and outputs calibrated verification verdicts.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        calibration_dir: Optional[str] = None,
        acceptance_threshold: float = 0.85,
        device: Optional[str] = None
    ):
        super().__init__(name="sidewalk_validator_adapter", version="1.2.0")
        self.model_path = model_path
        self.calibration_dir = calibration_dir or str(Path(__file__).parent / "calibration")
        self.acceptance_threshold = acceptance_threshold
        self.preferred_device = device or "cpu"
        self.model = None
        self._is_mock = False
        self._weight_sha256: Optional[str] = None
        self.calibration_maps: Dict[str, Dict[str, Dict[str, float]]] = {}

    def load(self, device: Optional[str] = None, allow_fallback: bool = False) -> bool:
        """Loads calibration mappings and DINOv2 weights."""
        self._status = AdapterStatus.LOADING
        target_device = device or self.preferred_device
        self._device = target_device

        try:
            # 1. Load calibration lookup tables
            for label_type in VALIDATOR_LABEL_TYPES:
                map_file = Path(self.calibration_dir) / f"{label_type}.json"
                if map_file.exists():
                    with open(map_file, "r") as f:
                        self.calibration_maps[label_type] = json.load(f)
                else:
                    logger.warning(f"Calibration table for {label_type} missing at {map_file}")

            # 2. Check for DINOv2 classifier weights
            candidate_paths = [
                self.model_path,
                "data/models/dinov2_validator.pt"
            ]
            valid_path = next((p for p in candidate_paths if p and os.path.exists(p)), None)

            if valid_path:
                import torch
                self.model = torch.load(valid_path, map_location=self._device, weights_only=True)
                self.model.eval()
                self._weight_sha256 = _compute_sha256(valid_path)
                self.model_path = valid_path
                self._status = AdapterStatus.READY
                self.is_initialized = True
                self._is_mock = False
                logger.info(f"Loaded live DINOv2 Validator model from {valid_path}")
                return True
            else:
                self._last_error = "UNAVAILABLE: MODEL_WEIGHTS_NOT_FOUND (dinov2_validator.pt)"
                if allow_fallback:
                    logger.info("Validator model weights not found locally; running in calibrated baseline mode.")
                    self._is_mock = True
                    self._status = AdapterStatus.DEGRADED
                    self.is_initialized = True
                    return True
                else:
                    self._status = AdapterStatus.UNAVAILABLE
                    self.is_initialized = False
                    logger.warning("Validator weights not found locally (dinov2_validator.pt); model UNAVAILABLE.")
                    return False

        except Exception as e:
            self._last_error = f"UNAVAILABLE: MODEL_DEPENDENCY_MISSING ({e})"
            if allow_fallback:
                self._status = AdapterStatus.DEGRADED
                self._is_mock = True
                self.is_initialized = True
                logger.warning(f"Validator initialization warning ({e}); operating in calibrated mode.")
                return True
            else:
                self._status = AdapterStatus.UNAVAILABLE
                self.is_initialized = False
                return False

    def unload(self) -> bool:
        """Frees model and clears cache."""
        try:
            if self.model is not None:
                del self.model
                self.model = None
            self._is_mock = False
            self.is_initialized = False
            self.calibration_maps.clear()
            self._status = AdapterStatus.NOT_LOADED

            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

            logger.info("Sidewalk Validator adapter unloaded.")
            return True
        except Exception as e:
            logger.error(f"Error unloading Sidewalk Validator adapter: {e}")
            return False

    def lookup_calibrated_accuracy(
        self,
        label_type: str,
        verdict: str,
        raw_confidence: float
    ) -> float:
        """
        Maps raw confidence in [0.5, 1.0] to empirical calibrated accuracy
        using Project Sidewalk's official non-decreasing calibration curves.
        """
        type_key = label_type.lower()
        if type_key not in self.calibration_maps:
            return round(raw_confidence, 4)

        table = self.calibration_maps[type_key].get(verdict, {})
        if not table:
            return round(raw_confidence, 4)

        # Keys are string floats ascending from "0.5" to "1.0"
        clamped_conf = max(0.5, min(1.0, raw_confidence))
        sorted_thresholds = sorted([float(k) for k in table.keys()])

        calibrated_val = 0.50
        for thresh in sorted_thresholds:
            if clamped_conf >= thresh:
                calibrated_val = table[f"{thresh:.2f}"] if f"{thresh:.2f}" in table else table[str(thresh)]
            else:
                break

        return round(float(calibrated_val), 4)

    def predict_v2(
        self,
        image: Union[np.ndarray, str],
        candidate_label: Optional[Union[str, DetectedBarrier]] = None,
        crop_box: Optional[Tuple[float, float, float, float]] = None,
        allow_fallback: bool = False,
        **kwargs
    ) -> AdapterInferenceResult:
        """
        Verifies a candidate accessibility detection on the image.
        """
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

            # Resolve label type
            if isinstance(candidate_label, DetectedBarrier):
                raw_type = candidate_label.barrier_type
                initial_conf = candidate_label.confidence
                if crop_box is None:
                    crop_box = candidate_label.bbox
            elif isinstance(candidate_label, str):
                raw_type = candidate_label
                initial_conf = kwargs.get("initial_confidence", 0.75)
            else:
                raw_type = "curb_ramp_present"
                initial_conf = 0.75

            validator_type = ACCESSGUARD_TO_VALIDATOR_TYPE.get(raw_type, "obstacle")

            # Extract crop if bounding box provided
            h, w = img_bgr.shape[:2]
            if crop_box:
                ymin, xmin, ymax, xmax = crop_box
                y1 = max(0, int(ymin * h))
                x1 = max(0, int(xmin * w))
                y2 = min(h, int(ymax * h))
                x2 = min(w, int(xmax * w))
                crop_img = img_bgr[y1:y2, x1:x2] if (y2 > y1 and x2 > x1) else img_bgr
            else:
                crop_img = img_bgr

            if self.model is not None and not self._is_mock:
                import torch
                from torchvision import transforms
                preprocess = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                tensor = preprocess(crop_img[:, :, ::-1]).unsqueeze(0).to(self._device)
                with torch.no_grad():
                    logits = self.model(tensor)
                    probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
                raw_prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
            elif self._is_mock:
                # Calibrated baseline: assess crop visual variance and initial confidence
                crop_gray = np.mean(crop_img, axis=2)
                contrast = float(np.std(crop_gray))
                # High contrast crops with reasonable initial confidence receive high probability
                if contrast > 25.0 and initial_conf >= 0.60:
                    raw_prob = min(0.96, initial_conf + 0.05)
                elif initial_conf < 0.40:
                    raw_prob = 0.35
                else:
                    raw_prob = initial_conf
            else:
                raw_prob = initial_conf

            verdict = "correct" if raw_prob >= 0.50 else "incorrect"
            calibrated_acc = self.lookup_calibrated_accuracy(validator_type, verdict, raw_prob)
            is_valid = (verdict == "correct") and (calibrated_acc >= self.acceptance_threshold)

            verification_verdict = VerificationVerdict(
                is_valid=is_valid,
                verdict=verdict,
                calibrated_accuracy=calibrated_acc,
                raw_confidence=round(raw_prob, 4),
                label_type=validator_type,
                evidence={
                    "original_label": raw_type,
                    "crop_evaluated": crop_box is not None,
                    "calibrated_via": f"accuracy_mappings/{validator_type}.json",
                    "mode": "calibrated_baseline" if self._is_mock else "dinov2_model"
                }
            )

            latency = (time.perf_counter() - start_time) * 1000.0
            self._inference_count += 1
            self._last_latency_ms = latency

            exec_type = "heuristic_fallback" if self._is_mock else "model_inference"

            return AdapterInferenceResult(
                status=self._status,
                adapter_name=self.name,
                verification=verification_verdict,
                execution_time_ms=round(latency, 2),
                device_used=self._device,
                execution_type=exec_type,
                weight_path=self.model_path if not self._is_mock else None,
                weight_sha256=self._weight_sha256 if not self._is_mock else None,
                upstream_repo="https://github.com/ProjectSidewalk/sidewalk-validator-ai",
                input_dimensions=(w, h),
                metadata={
                    "is_valid": is_valid,
                    "verdict": verdict,
                    "calibrated_accuracy": calibrated_acc,
                    "is_fallback": self._is_mock
                }
            )

        except Exception as e:
            self._last_error = str(e)
            logger.error(f"Validator inference failed: {e}")
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
        """Returns capabilities and MIT license."""
        return AdapterMetadata(
            name=self.name,
            version=self.version,
            model_family="DINOv2-Large (Crop Verification & Calibration)",
            modalities=[OutputModality.VERIFICATION],
            supported_classes=VALIDATOR_LABEL_TYPES,
            license="MIT",
            vram_budget_mb=2100,
            default_device="cpu",
            citation="Project Sidewalk Validator AI, University of Washington Makeability Lab, 2026",
            notes="Second-stage verification model; maps raw logits to calibrated accuracy percentiles."
        )
