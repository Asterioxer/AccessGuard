"""
CitySurfaces Model Adapter for AccessGuard.
Implements the high-resolution sidewalk surface material segmentation model from VIDA-NYU.
Segments 8 paving materials and computes continuous sidewalk roughness / rolling resistance metrics.
License: BSD 3-Clause (NVIDIA / NYU).
"""

import time
import os
import gc
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import cv2

from app.cv.adapters.base_v2 import (
    ModelAdapterV2,
    AdapterStatus,
    OutputModality,
    SemanticMask,
    AdapterInferenceResult,
    AdapterMetadata,
    AdapterHealth,
)
from app.cv.base import DetectedBarrier
from app.core.logging import logger

CITYSURFACES_CLASSES = [
    "concrete",            # 0
    "bricks",              # 1
    "granite",             # 2
    "asphalt",             # 3
    "mixed",               # 4
    "road",                # 5
    "background",          # 6
    "granite block-stone", # 7
    "hexagonal",           # 8
    "cobblestone"          # 9
]

# Surface smoothness index [0.0 = impassable / extreme vibration, 1.0 = smooth concrete]
MATERIAL_SMOOTHNESS_INDEX = {
    "concrete": 1.00,
    "asphalt": 0.95,
    "hexagonal": 0.75,
    "bricks": 0.70,
    "granite": 0.65,
    "mixed": 0.40,
    "granite block-stone": 0.35,
    "cobblestone": 0.20,
    "road": 0.90,
    "background": 0.00
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


class CitySurfacesAdapter(ModelAdapterV2):
    """
    CitySurfaces Sidewalk Paving Material & Roughness Segmentation Adapter.
    Outputs dense 2D masks of paving materials and calculates wheelchair rolling resistance.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        target_resolution: Tuple[int, int] = (512, 1024),
        device: Optional[str] = None
    ):
        super().__init__(name="citysurfaces_adapter", version="1.0.0")
        self.model_path = model_path
        self.target_resolution = target_resolution
        self.preferred_device = device or "cpu"
        self.model = None
        self._is_mock = False
        self._weight_sha256: Optional[str] = None

    def load(self, device: Optional[str] = None, allow_fallback: bool = False) -> bool:
        """Loads HRNet / OCRNet weights or records unavailable."""
        self._status = AdapterStatus.LOADING
        target_device = device or self.preferred_device
        self._device = target_device

        try:
            candidate_paths = [
                self.model_path,
                "data/models/block_c_10classes.pth"
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
                logger.info(f"Loaded authentic CitySurfaces model from {valid_path} on {self._device}")
                return True
            else:
                self._last_error = "UNAVAILABLE: MODEL_WEIGHTS_NOT_FOUND (block_c_10classes.pth)"
                if allow_fallback:
                    logger.info("CitySurfaces weights not found locally; running in calibrated baseline mode.")
                    self._is_mock = True
                    self._status = AdapterStatus.DEGRADED
                    self.is_initialized = True
                    return True
                else:
                    self._status = AdapterStatus.UNAVAILABLE
                    self.is_initialized = False
                    logger.warning("CitySurfaces weights not found locally (block_c_10classes.pth); model marked UNAVAILABLE.")
                    return False

        except Exception as e:
            self._last_error = f"UNAVAILABLE: MODEL_DEPENDENCY_MISSING ({e})"
            if allow_fallback:
                self._status = AdapterStatus.DEGRADED
                self._is_mock = True
                self.is_initialized = True
                logger.warning(f"Could not load native CitySurfaces network ({e}); activated baseline.")
                return True
            else:
                self._status = AdapterStatus.UNAVAILABLE
                self.is_initialized = False
                return False

    def unload(self) -> bool:
        """Deallocates model and clears memory."""
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

            logger.info("CitySurfaces adapter unloaded.")
            return True
        except Exception as e:
            logger.error(f"Error unloading CitySurfaces adapter: {e}")
            return False

    def _compute_roughness_and_distribution(
        self,
        mask_2d: np.ndarray
    ) -> Tuple[float, Dict[str, float]]:
        """Calculates material area distribution and overall rolling resistance index."""
        total_pixels = mask_2d.size
        distribution: Dict[str, float] = {}
        weighted_smoothness = 0.0
        sidewalk_pixels = 0

        for class_id, class_name in enumerate(CITYSURFACES_CLASSES):
            count = int(np.sum(mask_2d == class_id))
            if count > 0:
                fraction = count / total_pixels
                distribution[class_name] = round(fraction, 4)

                # Only include pedestrian walkway classes in roughness score
                if class_name not in ("road", "background"):
                    sidewalk_pixels += count
                    weighted_smoothness += count * MATERIAL_SMOOTHNESS_INDEX[class_name]

        if sidewalk_pixels > 0:
            overall_roughness = 1.0 - (weighted_smoothness / sidewalk_pixels)
        else:
            overall_roughness = 0.50  # Unknown / unpaved default

        return round(overall_roughness, 4), distribution

    def predict_v2(self, image: Union[np.ndarray, str], allow_fallback: bool = False, **kwargs) -> AdapterInferenceResult:
        """Executes surface material segmentation and roughness calculation."""
        start_time = time.perf_counter()

        if self._status not in (AdapterStatus.READY, AdapterStatus.DEGRADED):
            if not self.load(allow_fallback=allow_fallback):
                return AdapterInferenceResult(
                    status=self._status,
                    adapter_name=self.name,
                    execution_type="heuristic_fallback" if self._is_mock else "model_inference",
                    error_message=f"Model not initialized ({self._last_error or 'load failed'})"
                )

        try:
            if isinstance(image, str):
                img_bgr = cv2.imread(image)
                if img_bgr is None:
                    raise ValueError(f"Could not load image from {image}")
            else:
                img_bgr = image

            h, w = img_bgr.shape[:2]
            th, tw = self.target_resolution

            if self.model is not None and not self._is_mock:
                import torch

                resized = cv2.resize(img_bgr, (tw, th))
                rgb = resized[:, :, ::-1].astype(np.float32) / 255.0
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                norm = (rgb - mean) / std
                tensor = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).to(self._device)

                with torch.no_grad():
                    logits = self.model(tensor)
                    preds = torch.argmax(logits, dim=1).squeeze().cpu().numpy().astype(np.uint8)
                mask_2d = preds
            elif self._is_mock:
                # Calibrated baseline: identify lower-third walkway region
                mask_2d = np.full((th, tw), 6, dtype=np.uint8) # background
                walkway_h_start = int(th * 0.50)
                mask_2d[walkway_h_start:, :] = 0 # concrete default

                # Inspect color / texture variation for rough surface simulation
                gray_walkway = cv2.cvtColor(img_bgr[int(h * 0.50):, :], cv2.COLOR_BGR2GRAY)
                lap_var = float(cv2.Laplacian(gray_walkway, cv2.CV_64F).var())
                sx_std = float(np.std(np.abs(np.diff(gray_walkway, axis=1))))

                if lap_var > 100.0 or sx_std > 20.0:
                    # Patch of cobblestone / mixed rough / damaged surface detected
                    mask_2d[walkway_h_start + 40:, tw//6: 5*tw//6] = 9 # cobblestone
                    mask_2d[walkway_h_start + 80:, tw//4: 3*tw//4] = 4 # mixed
            else:
                mask_2d = np.full((th, tw), 6, dtype=np.uint8)

            roughness_score, material_dist = self._compute_roughness_and_distribution(mask_2d)

            semantic_mask = SemanticMask(
                mask=mask_2d,
                classes=CITYSURFACES_CLASSES,
                resolution=(th, tw),
                roughness_score=roughness_score,
                evidence={
                    "material_distribution": material_dist,
                    "dominant_material": max(material_dist, key=material_dist.get) if material_dist else "unknown",
                    "source": "CitySurfaces_model" if not self._is_mock else "CitySurfaces_calibrated_baseline"
                }
            )

            # If surface roughness exceeds threshold (> 0.35), synthesize surface_damage barrier
            detections: List[DetectedBarrier] = []
            if roughness_score > 0.35:
                detections.append(
                    DetectedBarrier(
                        barrier_type="surface_damage",
                        confidence=round(min(0.95, roughness_score + 0.40), 4),
                        uncertainty_score=0.15,
                        bbox=(0.50, 0.20, 0.85, 0.80),
                        evidence={
                            "roughness_score": roughness_score,
                            "reason": "Severe surface roughness (cobblestone / damaged paving) exceeds wheelchair safety threshold."
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
                mask=semantic_mask,
                execution_time_ms=round(latency, 2),
                device_used=self._device,
                execution_type=exec_type,
                weight_path=self.model_path if not self._is_mock else None,
                weight_sha256=self._weight_sha256 if not self._is_mock else None,
                upstream_repo="https://github.com/VIDA-NYU/city-surfaces",
                input_dimensions=(w, h),
                metadata={
                    "roughness_score": roughness_score,
                    "dominant_material": semantic_mask.evidence["dominant_material"],
                    "is_fallback": self._is_mock
                }
            )

        except Exception as e:
            self._last_error = str(e)
            logger.error(f"CitySurfaces inference failed: {e}")
            return AdapterInferenceResult(
                status=AdapterStatus.INFERENCE_FAILED,
                adapter_name=self.name,
                error_message=str(e),
                execution_time_ms=round((time.perf_counter() - start_time) * 1000.0, 2),
                device_used=self._device
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
        """Returns capabilities, paving taxonomy, and BSD licensing."""
        return AdapterMetadata(
            name=self.name,
            version=self.version,
            model_family="HRNet-W48 / OCRNet (Semantic Segmentation)",
            modalities=[OutputModality.SEMANTIC_MASK, OutputModality.BOUNDING_BOX],
            supported_classes=CITYSURFACES_CLASSES,
            license="BSD 3-Clause",
            vram_budget_mb=2200,
            default_device="cpu",
            citation="Hosseini et al., CitySurfaces, Sustainable Cities and Society 2022",
            notes="Segments 8 paving materials; provides continuous sidewalk roughness metrics."
        )
