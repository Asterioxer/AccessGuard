"""
Hybrid Vision Adapter for AccessGuard.
Orchestrates multi-expert perception across YOLO, GuideCane, RampNet, CitySurfaces,
and Sidewalk Validator through the Multi-Model Evidence Fusion Engine.
Supports FULL_HYBRID and RESOURCE_CONSTRAINED sequential memory profiles.
"""

import time
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
from app.cv.adapters.yolo_adapter import YOLOAdapter
from app.cv.adapters.guidecane_adapter import GuideCaneAdapter
from app.cv.adapters.rampnet_adapter import RampNetAdapter
from app.cv.adapters.citysurfaces_adapter import CitySurfacesAdapter
from app.cv.adapters.sidewalk_validator_adapter import SidewalkValidatorAdapter
from app.cv.fusion.evidence_fusion import EvidenceFusionEngine, FusedBarrierEvidence
from app.config import settings
from app.core.logging import logger


class HybridVisionAdapter(ModelAdapterV2):
    """
    Unified Hybrid Vision Adapter.
    Combines complementary models, reconciles contradictions, and emits calibrated evidence.
    """

    def __init__(
        self,
        mode: Optional[str] = None,
        enable_validator: Optional[bool] = None,
        device: Optional[str] = None
    ):
        super().__init__(name="hybrid_vision_adapter", version="2.0.0")
        self.mode = mode or settings.HYBRID_MODE
        self.enable_validator = enable_validator if enable_validator is not None else settings.ENABLE_VALIDATOR_FILTERING
        self.preferred_device = device or "cpu"

        # Instantiate sub-adapters
        self.yolo_adapter = YOLOAdapter()
        self.guidecane_adapter = GuideCaneAdapter()
        self.rampnet_adapter = RampNetAdapter(enable_seam_suppression=settings.ENABLE_SEAM_SUPPRESSION)
        self.citysurfaces_adapter = CitySurfacesAdapter()
        self.validator_adapter = SidewalkValidatorAdapter()
        self.fusion_engine = EvidenceFusionEngine(iou_threshold=0.40)

    def load(self, device: Optional[str] = None) -> bool:
        """Loads sub-adapters according to operating mode."""
        self._status = AdapterStatus.LOADING
        target_device = device or self.preferred_device
        self._device = target_device

        try:
            if self.mode == "FULL_HYBRID":
                # Load all models into memory concurrently
                self.yolo_adapter.load()
                self.guidecane_adapter.load(target_device)
                self.rampnet_adapter.load(target_device)
                self.citysurfaces_adapter.load(target_device)
                if self.enable_validator:
                    self.validator_adapter.load(target_device)
            else:
                # RESOURCE_CONSTRAINED: Staged loading, verified on demand
                logger.info("HybridAdapter operating in RESOURCE_CONSTRAINED mode (sequential offloading).")

            self._status = AdapterStatus.READY
            self.is_initialized = True
            return True
        except Exception as e:
            self._status = AdapterStatus.DEGRADED
            self._last_error = str(e)
            logger.warning(f"HybridAdapter loaded in degraded mode: {e}")
            return True

    def unload(self) -> bool:
        """Unloads all sub-adapters and purges caches."""
        try:
            self.yolo_adapter.unload() if hasattr(self.yolo_adapter, "unload") else None
            self.guidecane_adapter.unload()
            self.rampnet_adapter.unload()
            self.citysurfaces_adapter.unload()
            self.validator_adapter.unload()

            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

            self._status = AdapterStatus.NOT_LOADED
            self.is_initialized = False
            return True
        except Exception as e:
            logger.error(f"Error unloading HybridAdapter: {e}")
            return False

    def predict_v2(self, image: Union[np.ndarray, str], **kwargs) -> AdapterInferenceResult:
        """Executes hybrid multi-model inference and evidence fusion."""
        start_time = time.perf_counter()

        if self._status not in (AdapterStatus.READY, AdapterStatus.DEGRADED):
            if not self.load():
                return AdapterInferenceResult(
                    status=AdapterStatus.INFERENCE_FAILED,
                    adapter_name=self.name,
                    error_message=f"HybridAdapter not initialized ({self._last_error or 'load failed'})"
                )

        try:
            sub_results: List[AdapterInferenceResult] = []

            if self.mode == "RESOURCE_CONSTRAINED":
                # Phase 1: Primary Detectors (YOLO & GuideCane)
                self.guidecane_adapter.load(self._device)
                r_yolo = self.yolo_adapter.predict_v2(image)
                r_guide = self.guidecane_adapter.predict_v2(image)
                sub_results.extend([r_yolo, r_guide])
                self.guidecane_adapter.unload()

                # Phase 2: Curb Ramp Heatmaps (RampNet)
                self.rampnet_adapter.load(self._device)
                r_ramp = self.rampnet_adapter.predict_v2(image)
                sub_results.append(r_ramp)
                self.rampnet_adapter.unload()

                # Phase 3: Paving Roughness (CitySurfaces)
                self.citysurfaces_adapter.load(self._device)
                r_city = self.citysurfaces_adapter.predict_v2(image)
                sub_results.append(r_city)
                self.citysurfaces_adapter.unload()
            else:
                # Concurrent execution
                r_yolo = self.yolo_adapter.predict_v2(image)
                r_guide = self.guidecane_adapter.predict_v2(image)
                r_ramp = self.rampnet_adapter.predict_v2(image)
                r_city = self.citysurfaces_adapter.predict_v2(image)
                sub_results.extend([r_yolo, r_guide, r_ramp, r_city])

            # Phase 4: Candidate Verification (Sidewalk Validator)
            validator_verdict = None
            if self.enable_validator:
                if self.mode == "RESOURCE_CONSTRAINED":
                    self.validator_adapter.load(self._device)

                # Identify dominant candidate for verification
                candidate_crop = None
                candidate_type = "obstacle"
                for res in sub_results:
                    if res.keypoints:
                        candidate_crop = res.keypoints[0].to_detected_barrier().bbox
                        candidate_type = "curb_ramp_present"
                        break
                    elif res.detections:
                        candidate_crop = res.detections[0].bbox
                        candidate_type = res.detections[0].barrier_type
                        break

                if candidate_crop is not None:
                    v_res = self.validator_adapter.predict_v2(
                        image,
                        candidate_label=candidate_type,
                        crop_box=candidate_crop
                    )
                    validator_verdict = v_res.verification

                if self.mode == "RESOURCE_CONSTRAINED":
                    self.validator_adapter.unload()

            # Phase 5: Multi-Model Evidence Fusion
            fused = self.fusion_engine.fuse(sub_results, validator_verdict=validator_verdict)

            # Filter out rejected barriers
            valid_fused = [b for b in fused.fused_barriers if b.verification_state != "REJECTED"]
            detections = [b.to_detected_barrier() for b in valid_fused]

            latency = (time.perf_counter() - start_time) * 1000.0
            self._inference_count += 1
            self._last_latency_ms = latency

            return AdapterInferenceResult(
                status=self._status,
                adapter_name=self.name,
                detections=detections,
                keypoints=[kp for b in valid_fused for kp in b.keypoints],
                mask=fused.surface_mask,
                verification=validator_verdict,
                execution_time_ms=round(latency, 2),
                device_used=self._device,
                metadata={
                    "mode": self.mode,
                    "participating_adapters": list(fused.adapter_statuses.keys()),
                    "surface_roughness": fused.surface_roughness,
                    "fused_barrier_count": len(detections)
                }
            )

        except Exception as e:
            self._last_error = str(e)
            logger.error(f"HybridAdapter prediction failed: {e}")
            return AdapterInferenceResult(
                status=AdapterStatus.INFERENCE_FAILED,
                adapter_name=self.name,
                error_message=str(e),
                execution_time_ms=round((time.perf_counter() - start_time) * 1000.0, 2),
                device_used=self._device
            )

    def health(self) -> AdapterHealth:
        """Returns aggregated telemetry across sub-adapters."""
        return AdapterHealth(
            name=self.name,
            status=self._status,
            device=self._device,
            total_inferences=self._inference_count,
            last_latency_ms=round(self._last_latency_ms, 2),
            error_message=self._last_error
        )

    def metadata(self) -> AdapterMetadata:
        """Returns hybrid capabilities metadata."""
        return AdapterMetadata(
            name=self.name,
            version=self.version,
            model_family="Hybrid Ensemble (YOLO + GuideCane + RampNet + CitySurfaces + Validator)",
            modalities=[
                OutputModality.BOUNDING_BOX,
                OutputModality.POINT_KEYPOINT,
                OutputModality.SEMANTIC_MASK,
                OutputModality.VERIFICATION
            ],
            supported_classes=[
                "curb_ramp_present", "curb_ramp_missing", "stairs",
                "pole_or_bollard", "construction_obstacle", "surface_damage",
                "crosswalk_present", "tactile_paving_present"
            ],
            license="MIT (Open-Source Clean Integration)",
            vram_budget_mb=1820 if self.mode == "RESOURCE_CONSTRAINED" else 3500,
            default_device="cpu",
            citation="AccessGuard Hybrid Vision Intelligence Consortium, 2026",
            notes=f"Operating profile: {self.mode}. Dynamic memory offloading enabled."
        )
