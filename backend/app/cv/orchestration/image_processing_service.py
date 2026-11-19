"""
AccessGuard Canonical Unified Image-Processing Service (Mission 006.2).
Provides an immutable, auditable 15-stage processing pipeline for every real-world image
entering AccessGuard, regardless of ingress route (user uploads, autonomous ingestion, video frames, benchmarks).
"""

import os
import time
import uuid
import json
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from sqlalchemy.orm import Session

from app.core.config_wrapper import get_settings
from app.core.logging import logger
from app.core.sanitize import (
    validate_upload,
    extract_exif_gps,
    apply_privacy_redaction,
)
from app.cv.adapters.base_v2 import (
    AdapterStatus,
    ExecutionPolicy,
    OutputModality,
    PointKeypoint,
    SemanticMask,
    VerificationVerdict,
    AdapterInferenceResult,
)
from app.cv.adapters.hybrid_adapter import HybridVisionAdapter
from app.cv.adapters.rampnet_adapter import RampNetAdapter
from app.cv.adapters.yolo_adapter import YOLOAdapter
from app.cv.adapters.guidecane_adapter import GuideCaneAdapter
from app.cv.adapters.citysurfaces_adapter import CitySurfacesAdapter
from app.cv.adapters.sidewalk_validator_adapter import SidewalkValidatorAdapter
from app.cv.segmentation.base import SegmentationAdapter, SegmentedRegion
from app.cv.segmentation.yolo_seg_adapter import YOLOSegmentationAdapter
from app.cv.segmentation.heuristic_seg_adapter import HeuristicSegmentationAdapter
from app.cv.fusion.evidence_fusion import EvidenceFusionEngine, FusedBarrierEvidence
from app.cv.temporal_tracker import TemporalObstacleTracker, TrackedBarrier
from app.cv.base import DetectedBarrier
from app.db.models import (
    Observation,
    BarrierDetection,
    SegmentedObservation,
    ImageProvenance,
    PedestrianSegment,
)
from app.gis.spatial_index import snap_point_to_segments
from app.gis.scoring import calculate_segment_accessibility_score


@dataclass
class ImageProcessingRequest:
    """Canonical request payload for image processing."""
    image_bytes: bytes
    filename: str = "image.jpg"
    external_image_id: Optional[str] = None
    source: str = "user_upload"                       # "user_upload", "mapillary", "kartaview", "live_stream", "benchmark"
    source_metadata: Dict[str, Any] = field(default_factory=dict)
    requested_adapter: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    capture_timestamp: Optional[str] = None
    execution_policy: ExecutionPolicy = ExecutionPolicy.STRICT_GENUINE
    enable_privacy_redaction: bool = True
    contributor_id: Optional[str] = None
    preferred_device: Optional[str] = None
    save_media: bool = True


@dataclass
class ModelExecutionRecord:
    """Audit record capturing individual model adapter execution telemetry."""
    adapter_name: str
    model_family: str
    upstream_repo: str
    checkpoint_path: Optional[str]
    weight_sha256: Optional[str]
    model_architecture: str
    model_version: str
    class_mapping_version: str
    device_used: str
    execution_type: str                              # "model_inference", "heuristic_fallback", "unavailable"
    execution_time_ms: float
    input_dimensions: Tuple[int, int]
    output_count: int
    status: str                                     # "READY", "UNAVAILABLE", "DEGRADED", "INFERENCE_FAILED"
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class SegmentationResultRecord:
    """Record capturing pixel-level semantic segmentation outcomes."""
    status: str                                     # "READY", "UNAVAILABLE", "DEGRADED"
    model_identifier: str
    model_version: str
    execution_type: str                              # "model_inference", "heuristic_overlay", "none"
    weight_sha256: Optional[str] = None
    regions_count: int = 0
    regions: List[Dict[str, Any]] = field(default_factory=list)
    error_message: Optional[str] = None


@dataclass
class ImageProcessingResult:
    """Canonical, auditable output container for the unified inference pipeline."""
    run_id: str
    image_id: str
    source: str
    source_metadata: Dict[str, Any]
    original_sha256: str
    sanitized_sha256: str
    processing_timestamp: str
    sanitized_image_path: Optional[str]
    latitude: Optional[float]
    longitude: Optional[float]
    model_execution_records: List[ModelExecutionRecord]
    detections: List[DetectedBarrier]
    segmentation_results: SegmentationResultRecord
    fused_evidence: List[Dict[str, Any]]
    uncertainty_summary: Dict[str, Any]
    resource_telemetry: Dict[str, Any]
    warnings: List[str]
    errors: List[str]
    overall_status: str                              # "SUCCESS", "PARTIAL", "FAILED"
    observation_id: Optional[str] = None
    segment_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serializes result to JSON-compatible dictionary."""
        return {
            "run_id": self.run_id,
            "image_id": self.image_id,
            "source": self.source,
            "source_metadata": self.source_metadata,
            "original_sha256": self.original_sha256,
            "sanitized_sha256": self.sanitized_sha256,
            "processing_timestamp": self.processing_timestamp,
            "sanitized_image_path": self.sanitized_image_path,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "model_execution_records": [r.__dict__ for r in self.model_execution_records],
            "detections": [
                {
                    "barrier_type": d.barrier_type,
                    "confidence": d.confidence,
                    "uncertainty_score": d.uncertainty_score,
                    "bbox": d.bbox,
                    "evidence": d.evidence
                }
                for d in self.detections
            ],
            "segmentation_results": self.segmentation_results.__dict__,
            "fused_evidence": self.fused_evidence,
            "uncertainty_summary": self.uncertainty_summary,
            "resource_telemetry": self.resource_telemetry,
            "warnings": self.warnings,
            "errors": self.errors,
            "overall_status": self.overall_status,
            "observation_id": self.observation_id,
            "segment_id": self.segment_id
        }


@dataclass
class VideoProcessingResult:
    """Canonical, auditable output container for unified video processing and temporal tracking."""
    video_id: str
    video_path: str
    original_sha256: str
    duration_seconds: float
    total_frames: int
    sampled_frames: int
    frame_results: List[ImageProcessingResult]
    detected_barriers: List[DetectedBarrier]
    unique_obstacles_count: int
    execution_policy: str
    resource_telemetry: Dict[str, Any]
    warnings: List[str]
    errors: List[str]
    overall_status: str                              # "SUCCESS", "PARTIAL", "FAILED"
    observation_id: Optional[str] = None
    segment_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serializes result to JSON-compatible dictionary."""
        return {
            "video_id": self.video_id,
            "video_path": self.video_path,
            "original_sha256": self.original_sha256,
            "duration_seconds": self.duration_seconds,
            "total_frames": self.total_frames,
            "sampled_frames": self.sampled_frames,
            "frame_results_count": len(self.frame_results),
            "detected_barriers": [
                {
                    "barrier_type": d.barrier_type,
                    "confidence": d.confidence,
                    "uncertainty_score": d.uncertainty_score,
                    "bbox": d.bbox,
                    "evidence": d.evidence,
                    "is_confirmed": d.is_confirmed
                }
                for d in self.detected_barriers
            ],
            "unique_obstacles_count": self.unique_obstacles_count,
            "execution_policy": self.execution_policy,
            "resource_telemetry": self.resource_telemetry,
            "warnings": self.warnings,
            "errors": self.errors,
            "overall_status": self.overall_status,
            "observation_id": self.observation_id,
            "segment_id": self.segment_id
        }


class ImageProcessingService:
    """
    Central Image Processing Service.
    Enforces a single, auditable 15-stage lifecycle for all incoming streetscape imagery.
    """

    def __init__(
        self,
        hybrid_adapter: Optional[HybridVisionAdapter] = None,
        segmentation_adapter: Optional[SegmentationAdapter] = None,
        storage_dir: Optional[Path] = None,
        device: Optional[str] = None
    ):
        settings = get_settings()
        self.preferred_device = device or ("cuda" if self._has_cuda() else "cpu")
        self.storage_dir = storage_dir or settings.PROCESSED_DIR
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.hybrid_adapter = hybrid_adapter or HybridVisionAdapter(
            mode="RESOURCE_CONSTRAINED",
            device=self.preferred_device
        )
        self.seg_adapter = segmentation_adapter or YOLOSegmentationAdapter()

    @staticmethod
    def _has_cuda() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def process_image(
        self,
        request: ImageProcessingRequest,
        db: Optional[Session] = None
    ) -> ImageProcessingResult:
        """
        Executes the canonical 15-stage image processing lifecycle.
        """
        start_wall_time = time.perf_counter()
        processing_ts = datetime.now(timezone.utc).isoformat()
        run_id = f"proc_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        image_id = str(uuid.uuid4())

        warnings: List[str] = []
        errors: List[str] = []
        model_exec_records: List[ModelExecutionRecord] = []

        # 1 & 2: Validate image bytes, size, and magic numbers
        settings = get_settings()
        is_valid, mime_type, _ = validate_upload(
            request.image_bytes,
            request.filename,
            max_size_bytes=settings.MAX_IMAGE_SIZE_BYTES
        )
        if not is_valid:
            return self._build_failed_result(
                run_id=run_id,
                image_id=image_id,
                request=request,
                error_msg=f"Image validation failed: {mime_type}",
                processing_ts=processing_ts
            )

        # 3. Safe image decoding
        arr = np.frombuffer(request.image_bytes, np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return self._build_failed_result(
                run_id=run_id,
                image_id=image_id,
                request=request,
                error_msg="OpenCV failed to decode image bytes into valid pixel matrix.",
                processing_ts=processing_ts
            )

        h, w = img_bgr.shape[:2]
        if h < 32 or w < 32 or h > 8192 or w > 8192:
            return self._build_failed_result(
                run_id=run_id,
                image_id=image_id,
                request=request,
                error_msg=f"Image dimensions ({w}x{h}) out of allowable range [32, 8192].",
                processing_ts=processing_ts
            )

        # 4. Compute original image SHA-256
        orig_sha256 = hashlib.sha256(request.image_bytes).hexdigest()

        # 5. Extract GPS metadata without inventing false coordinates
        final_lat = request.latitude
        final_lon = request.longitude
        if final_lat is None or final_lon is None:
            exif_lat, exif_lon = extract_exif_gps(request.image_bytes)
            if exif_lat is not None and exif_lon is not None:
                final_lat = exif_lat
                final_lon = exif_lon
            else:
                warnings.append("No geographic coordinates provided or extracted from EXIF; image remains unanchored.")

        # 6. Apply privacy redaction (blur faces and license plates)
        if request.enable_privacy_redaction and settings.ENABLE_PRIVACY_BLURRING:
            redacted_bgr = apply_privacy_redaction(img_bgr)
        else:
            redacted_bgr = img_bgr.copy()

        # 7. Compute sanitized image SHA-256
        success_encode, encoded_buf = cv2.imencode(".jpg", redacted_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if success_encode:
            sanitized_bytes = encoded_buf.tobytes()
            sanitized_sha256 = hashlib.sha256(sanitized_bytes).hexdigest()
        else:
            sanitized_bytes = request.image_bytes
            sanitized_sha256 = orig_sha256
            warnings.append("Could not re-encode sanitized image for SHA calculation; using raw hash.")

        # Save sanitized file to storage if requested
        saved_media_path: Optional[str] = None
        if request.save_media:
            ext = request.filename.split(".")[-1].lower() if "." in request.filename else "jpg"
            saved_filename = f"obs_{run_id}.{ext}"
            saved_path_obj = self.storage_dir / saved_filename
            with open(saved_path_obj, "wb") as f_out:
                f_out.write(sanitized_bytes)
            saved_media_path = str(saved_path_obj)

        # 8 & 9. Execute configured genuine model adapters under strict policy
        allow_dev_fallback = (request.execution_policy == ExecutionPolicy.DEVELOPMENT_FALLBACK)
        hybrid_exec_start = time.perf_counter()

        req_adapter = (
            request.requested_adapter
            or request.source_metadata.get("requested_adapter")
            or "unified"
        ).lower().strip()

        # Track specialist latencies
        t_yolo = t_ramp = t_guide = t_city = t_val = 0.0
        valid_fused_barriers = []

        if req_adapter in ("yolo", "onnx"):
            # Execute only requested YOLO Specialist
            t0 = time.perf_counter()
            r_yolo = self.hybrid_adapter.yolo_adapter.predict_v2(
                redacted_bgr,
                allow_fallback=allow_dev_fallback
            )
            t_yolo = (time.perf_counter() - t0) * 1000.0
            model_exec_records.append(ModelExecutionRecord(
                adapter_name=self.hybrid_adapter.yolo_adapter.name,
                model_family="YOLOv8",
                upstream_repo="https://github.com/ultralytics/ultralytics",
                checkpoint_path=getattr(self.hybrid_adapter.yolo_adapter, "model_path", None),
                weight_sha256=getattr(self.hybrid_adapter.yolo_adapter, "_weight_sha256", None),
                model_architecture="YOLOv8n-Detection",
                model_version=getattr(self.hybrid_adapter.yolo_adapter, "version", "1.2.0"),
                class_mapping_version="v1.2",
                device_used=getattr(self.hybrid_adapter.yolo_adapter, "_device", "cpu"),
                execution_type=r_yolo.execution_type,
                execution_time_ms=round(t_yolo, 2),
                input_dimensions=(w, h),
                output_count=len(r_yolo.detections) if r_yolo.is_successful else 0,
                status=r_yolo.status.value,
                warnings=[] if r_yolo.is_successful else [f"YOLO returned non-ready status: {r_yolo.error_message or r_yolo.status.value}"],
                errors=[r_yolo.error_message] if (r_yolo.error_message and not r_yolo.is_successful) else []
            ))
            if r_yolo.error_message and not r_yolo.is_successful:
                errors.append(f"YOLO execution error: {r_yolo.error_message}")
            final_detections = r_yolo.detections if r_yolo.is_successful else []

        elif req_adapter in ("rampnet", "convnextv2"):
            # Execute only requested RampNet Specialist
            t0 = time.perf_counter()
            r_ramp = self.hybrid_adapter.rampnet_adapter.predict_v2(
                redacted_bgr,
                allow_fallback=allow_dev_fallback
            )
            t_ramp = (time.perf_counter() - t0) * 1000.0
            model_exec_records.append(ModelExecutionRecord(
                adapter_name=self.hybrid_adapter.rampnet_adapter.name,
                model_family="ConvNeXt-V2",
                upstream_repo="https://github.com/ProjectSidewalk/RampNet",
                checkpoint_path=getattr(self.hybrid_adapter.rampnet_adapter, "_weight_path", None),
                weight_sha256=getattr(self.hybrid_adapter.rampnet_adapter, "_weight_sha256", None),
                model_architecture="KeypointModel(convnextv2_base)",
                model_version=getattr(self.hybrid_adapter.rampnet_adapter, "version", "1.1.0"),
                class_mapping_version="v2.0",
                device_used=getattr(self.hybrid_adapter.rampnet_adapter, "_device", "cpu"),
                execution_type=r_ramp.execution_type,
                execution_time_ms=round(t_ramp, 2),
                input_dimensions=(w, h),
                output_count=len(r_ramp.keypoints) if r_ramp.is_successful else 0,
                status=r_ramp.status.value,
                warnings=[] if r_ramp.is_successful else ["RampNet not available or degraded."],
                errors=[r_ramp.error_message] if r_ramp.error_message else []
            ))
            if r_ramp.error_message and not r_ramp.is_successful:
                errors.append(f"RampNet execution error: {r_ramp.error_message}")
            final_detections = [kp.to_detected_barrier() for kp in r_ramp.keypoints] if r_ramp.is_successful else []

        elif req_adapter in ("heuristic", "fallback"):
            from app.cv.adapters.heuristic_adapter import HeuristicCVAdapter
            heur = HeuristicCVAdapter()
            t0 = time.perf_counter()
            detections = heur.predict(redacted_bgr)
            t_heur = (time.perf_counter() - t0) * 1000.0
            model_exec_records.append(ModelExecutionRecord(
                adapter_name=heur.name,
                model_family="OpenCV-Morphological",
                upstream_repo="internal",
                checkpoint_path=None,
                weight_sha256=None,
                model_architecture="HeuristicContourDetector",
                model_version=getattr(heur, "version", "1.0.0"),
                class_mapping_version="v1.0",
                device_used="cpu",
                execution_type="heuristic_fallback",
                execution_time_ms=round(t_heur, 2),
                input_dimensions=(w, h),
                output_count=len(detections),
                status="DEGRADED",
                warnings=["Operating in requested OpenCV heuristic fallback mode."],
                errors=[]
            ))
            final_detections = detections

        elif req_adapter in ("torchvision", "faster_rcnn"):
            from app.cv.adapters.torchvision_adapter import TorchvisionAdapter
            tv = TorchvisionAdapter(weights_path=getattr(self.hybrid_adapter, "torchvision_weights_path", None))
            t0 = time.perf_counter()
            r_tv = tv.predict_v2(redacted_bgr, allow_fallback=allow_dev_fallback)
            t_tv = (time.perf_counter() - t0) * 1000.0

            warnings_list = []
            if r_tv.status == AdapterStatus.DEGRADED:
                warnings_list.append("Torchvision running in fallback mode.")
            errors_list = [r_tv.error_message] if r_tv.error_message else []

            model_exec_records.append(ModelExecutionRecord(
                adapter_name=r_tv.adapter_name,
                model_family="MaskRCNN-ResNet50",
                upstream_repo=r_tv.upstream_repo or "torchvision",
                checkpoint_path=r_tv.weight_path,
                weight_sha256=r_tv.weight_sha256,
                model_architecture="MaskRCNN",
                model_version=r_tv.model_version or getattr(tv, "version", "1.0.0"),
                class_mapping_version="v1.0",
                device_used=r_tv.device_used,
                execution_type=r_tv.execution_type,
                execution_time_ms=round(t_tv, 2),
                input_dimensions=(w, h),
                output_count=len(r_tv.detections),
                status=r_tv.status.value,
                warnings=warnings_list,
                errors=errors_list
            ))
            final_detections = r_tv.detections

        elif req_adapter == "dummy":
            from app.cv.adapters.dummy_adapter import DummyAdapter
            dum = DummyAdapter()
            t0 = time.perf_counter()
            detections = dum.predict(redacted_bgr)
            t_dum = (time.perf_counter() - t0) * 1000.0
            model_exec_records.append(ModelExecutionRecord(
                adapter_name=dum.name,
                model_family="Deterministic-Benchmark",
                upstream_repo="internal",
                checkpoint_path=None,
                weight_sha256=None,
                model_architecture="DummyBenchmarkAdapter",
                model_version=getattr(dum, "version", "1.0.0"),
                class_mapping_version="v1.0",
                device_used="cpu",
                execution_type="heuristic_fallback",
                execution_time_ms=round(t_dum, 2),
                input_dimensions=(w, h),
                output_count=len(detections),
                status="READY",
                warnings=["Deterministic dummy benchmark adapter executed."],
                errors=[]
            ))
            final_detections = detections

        else:
            # Full Unified Hybrid Vision Pipeline with Multi-Model Evidence Fusion
            # Execute YOLO Specialist
            t0 = time.perf_counter()
            r_yolo = self.hybrid_adapter.yolo_adapter.predict_v2(
                redacted_bgr,
                allow_fallback=allow_dev_fallback
            )
            t_yolo = (time.perf_counter() - t0) * 1000.0
            model_exec_records.append(ModelExecutionRecord(
                adapter_name=self.hybrid_adapter.yolo_adapter.name,
                model_family="YOLOv8",
                upstream_repo="https://github.com/ultralytics/ultralytics",
                checkpoint_path=getattr(self.hybrid_adapter.yolo_adapter, "model_path", None),
                weight_sha256=getattr(self.hybrid_adapter.yolo_adapter, "_weight_sha256", None),
                model_architecture="YOLOv8n-Detection",
                model_version=getattr(self.hybrid_adapter.yolo_adapter, "version", "1.2.0"),
                class_mapping_version="v1.2",
                device_used=getattr(self.hybrid_adapter.yolo_adapter, "_device", "cpu"),
                execution_type=r_yolo.execution_type,
                execution_time_ms=round(t_yolo, 2),
                input_dimensions=(w, h),
                output_count=len(r_yolo.detections) if r_yolo.is_successful else 0,
                status=r_yolo.status.value,
                warnings=[] if r_yolo.is_successful else [f"YOLO returned non-ready status: {r_yolo.error_message or r_yolo.status.value}"],
                errors=[r_yolo.error_message] if (r_yolo.error_message and not r_yolo.is_successful) else []
            ))

            # Execute RampNet Specialist (ConvNeXt-V2)
            t0 = time.perf_counter()
            r_ramp = self.hybrid_adapter.rampnet_adapter.predict_v2(
                redacted_bgr,
                allow_fallback=allow_dev_fallback
            )
            t_ramp = (time.perf_counter() - t0) * 1000.0
            model_exec_records.append(ModelExecutionRecord(
                adapter_name=self.hybrid_adapter.rampnet_adapter.name,
                model_family="ConvNeXt-V2",
                upstream_repo="https://github.com/ProjectSidewalk/RampNet",
                checkpoint_path=getattr(self.hybrid_adapter.rampnet_adapter, "_weight_path", None),
                weight_sha256=getattr(self.hybrid_adapter.rampnet_adapter, "_weight_sha256", None),
                model_architecture="KeypointModel(convnextv2_base)",
                model_version=getattr(self.hybrid_adapter.rampnet_adapter, "version", "1.1.0"),
                class_mapping_version="v2.0",
                device_used=getattr(self.hybrid_adapter.rampnet_adapter, "_device", "cpu"),
                execution_type=r_ramp.execution_type,
                execution_time_ms=round(t_ramp, 2),
                input_dimensions=(w, h),
                output_count=len(r_ramp.keypoints),
                status=r_ramp.status.value,
                warnings=[] if r_ramp.is_successful else ["RampNet not available or degraded."],
                errors=[r_ramp.error_message] if r_ramp.error_message else []
            ))

            # Execute GuideCane Specialist
            t0 = time.perf_counter()
            r_guide = self.hybrid_adapter.guidecane_adapter.predict_v2(
                redacted_bgr,
                allow_fallback=allow_dev_fallback
            )
            t_guide = (time.perf_counter() - t0) * 1000.0
            model_exec_records.append(ModelExecutionRecord(
                adapter_name=self.hybrid_adapter.guidecane_adapter.name,
                model_family="YOLOv8-GuideCane",
                upstream_repo="https://github.com/yolo0731/GuideCaneOD",
                checkpoint_path=getattr(self.hybrid_adapter.guidecane_adapter, "model_path", None),
                weight_sha256=getattr(self.hybrid_adapter.guidecane_adapter, "_weight_sha256", None),
                model_architecture="GuideCane-17Class-Obstacle",
                model_version=getattr(self.hybrid_adapter.guidecane_adapter, "version", "1.0.0"),
                class_mapping_version="v1.0",
                device_used=getattr(self.hybrid_adapter.guidecane_adapter, "_device", "cpu"),
                execution_type=r_guide.execution_type,
                execution_time_ms=round(t_guide, 2),
                input_dimensions=(w, h),
                output_count=len(r_guide.detections),
                status=r_guide.status.value,
                warnings=[] if r_guide.is_successful else ["GuideCane weights not found; specialist omitted from fusion."],
                errors=[r_guide.error_message] if r_guide.error_message else []
            ))

            # Execute CitySurfaces Specialist
            t0 = time.perf_counter()
            r_city = self.hybrid_adapter.citysurfaces_adapter.predict_v2(
                redacted_bgr,
                allow_fallback=allow_dev_fallback
            )
            t_city = (time.perf_counter() - t0) * 1000.0
            model_exec_records.append(ModelExecutionRecord(
                adapter_name=self.hybrid_adapter.citysurfaces_adapter.name,
                model_family="CitySurfaces-HRNet",
                upstream_repo="https://github.com/VIDA-NYU/city-surfaces",
                checkpoint_path=getattr(self.hybrid_adapter.citysurfaces_adapter, "model_path", None),
                weight_sha256=getattr(self.hybrid_adapter.citysurfaces_adapter, "_weight_sha256", None),
                model_architecture="CitySurfaces-10Class-Roughness",
                model_version=getattr(self.hybrid_adapter.citysurfaces_adapter, "version", "1.0.0"),
                class_mapping_version="v1.0",
                device_used=getattr(self.hybrid_adapter.citysurfaces_adapter, "_device", "cpu"),
                execution_type=r_city.execution_type,
                execution_time_ms=round(t_city, 2),
                input_dimensions=(w, h),
                output_count=1 if r_city.mask else 0,
                status=r_city.status.value,
                warnings=[] if r_city.is_successful else ["CitySurfaces weights missing; surface roughness not measured."],
                errors=[r_city.error_message] if r_city.error_message else []
            ))

            # Execute Sidewalk Validator on Candidate Crop
            validator_verdict: Optional[VerificationVerdict] = None
            t_val = 0.0
            if self.hybrid_adapter.enable_validator:
                t0 = time.perf_counter()
                candidate_crop = None
                candidate_type = "curb_ramp_present"

                if r_ramp.keypoints:
                    candidate_crop = r_ramp.keypoints[0].to_detected_barrier().bbox
                    candidate_type = "curb_ramp_present"
                elif r_yolo.detections:
                    candidate_crop = r_yolo.detections[0].bbox
                    candidate_type = r_yolo.detections[0].barrier_type

                if candidate_crop is not None:
                    r_val = self.hybrid_adapter.validator_adapter.predict_v2(
                        redacted_bgr,
                        candidate_label=candidate_type,
                        crop_box=candidate_crop,
                        allow_fallback=allow_dev_fallback
                    )
                    validator_verdict = r_val.verification
                    t_val = (time.perf_counter() - t0) * 1000.0
                    model_exec_records.append(ModelExecutionRecord(
                        adapter_name=self.hybrid_adapter.validator_adapter.name,
                        model_family="DINOv2-Classifier",
                        upstream_repo="https://github.com/ProjectSidewalk/sidewalk-validator-ai",
                        checkpoint_path=getattr(self.hybrid_adapter.validator_adapter, "model_path", None),
                        weight_sha256=getattr(self.hybrid_adapter.validator_adapter, "_weight_sha256", None),
                        model_architecture="DINOv2-SidewalkValidator",
                        model_version=getattr(self.hybrid_adapter.validator_adapter, "version", "1.0.0"),
                        class_mapping_version="v1.0",
                        device_used=getattr(self.hybrid_adapter.validator_adapter, "_device", "cpu"),
                        execution_type=r_val.execution_type,
                        execution_time_ms=round(t_val, 2),
                        input_dimensions=(w, h),
                        output_count=1 if validator_verdict else 0,
                        status=r_val.status.value,
                        warnings=[] if r_val.is_successful else ["Validator unavailable; candidates marked tentative."],
                        errors=[r_val.error_message] if r_val.error_message else []
                    ))

            # 11 & 12. Multi-Model Evidence Fusion & Class-Aware Association
            sub_results = [r_yolo, r_guide, r_ramp, r_city]
            valid_sub_results = [r for r in sub_results if r.is_successful]

            fused_result = self.hybrid_adapter.fusion_engine.fuse(
                valid_sub_results,
                validator_verdict=validator_verdict
            )

            valid_fused_barriers = [b for b in fused_result.fused_barriers if b.verification_state != "REJECTED"]
            final_detections = [b.to_detected_barrier() for b in valid_fused_barriers]

        # 10. Execute Genuine Segmentation Adapter
        t0 = time.perf_counter()
        seg_record = self._execute_segmentation(redacted_bgr, allow_dev_fallback)
        t_seg = (time.perf_counter() - t0) * 1000.0

        # Uncertainty Summary
        uncertainties = [d.uncertainty_score for d in final_detections]
        confidences = [d.confidence for d in final_detections]
        uncertainty_summary = {
            "mean_uncertainty": round(float(np.mean(uncertainties)), 4) if uncertainties else 0.0,
            "max_uncertainty": round(float(np.max(uncertainties)), 4) if uncertainties else 0.0,
            "mean_confidence": round(float(np.mean(confidences)), 4) if confidences else 0.0,
            "total_fused_findings": len(final_detections),
            "consensus_count": sum(1 for b in valid_fused_barriers if b.consensus_status == "CONSENSUS"),
            "disagreement_count": sum(1 for b in valid_fused_barriers if b.consensus_status == "DISAGREEMENT")
        }

        # 13. Persist to Database if Session provided
        obs_id: Optional[str] = None
        seg_id: Optional[str] = None

        if db is not None:
            obs_id, seg_id = self._persist_records(
                db=db,
                request=request,
                run_id=run_id,
                image_id=image_id,
                saved_media_path=saved_media_path,
                latitude=final_lat,
                longitude=final_lon,
                orig_sha256=orig_sha256,
                sanitized_sha256=sanitized_sha256,
                final_detections=final_detections,
                seg_record=seg_record,
                model_exec_records=model_exec_records,
                uncertainty_summary=uncertainty_summary
            )

        # 14. Resource Telemetry
        total_wall_ms = (time.perf_counter() - start_wall_time) * 1000.0
        vram_mb, ram_mb = self._measure_resource_usage()

        resource_telemetry = {
            "total_wall_time_ms": round(total_wall_ms, 2),
            "yolo_time_ms": round(t_yolo, 2),
            "rampnet_time_ms": round(t_ramp, 2),
            "guidecane_time_ms": round(t_guide, 2),
            "citysurfaces_time_ms": round(t_city, 2),
            "validator_time_ms": round(t_val, 2),
            "segmentation_time_ms": round(t_seg, 2),
            "device": self.preferred_device,
            "peak_vram_mb": vram_mb,
            "ram_used_mb": ram_mb
        }

        # Determine overall status
        ready_count = sum(1 for r in model_exec_records if r.status in ("READY", "DEGRADED"))
        has_critical_failure = any(r.status in ("INFERENCE_FAILED", "RESOURCE_EXHAUSTED", "FAILED") for r in model_exec_records)

        if req_adapter not in ("unified", "hybrid"):
            overall_status = "SUCCESS" if (ready_count >= 1 and not has_critical_failure) else "FAILED"
        else:
            overall_status = "SUCCESS" if ready_count >= 1 else "FAILED"

        if valid_fused_barriers:
            fused_evidence_list = [
                {
                    "barrier_type": b.canonical_type,
                    "confidence": b.fused_confidence,
                    "uncertainty": b.uncertainty_score,
                    "consensus": b.consensus_status,
                    "verification_state": b.verification_state,
                    "adapters": b.contributing_adapters,
                    "bbox": b.bbox
                }
                for b in valid_fused_barriers
            ]
        else:
            fused_evidence_list = [
                {
                    "barrier_type": d.barrier_type,
                    "confidence": d.confidence,
                    "uncertainty": d.uncertainty_score,
                    "consensus": "SINGLE_MODEL",
                    "verification_state": "VERIFIED" if d.confidence >= 0.85 else "TENTATIVE",
                    "adapters": [req_adapter] if req_adapter not in ("unified", "hybrid") else ["hybrid_vision"],
                    "bbox": d.bbox
                }
                for d in final_detections
            ]

        return ImageProcessingResult(
            run_id=run_id,
            image_id=image_id,
            source=request.source,
            source_metadata=request.source_metadata,
            original_sha256=orig_sha256,
            sanitized_sha256=sanitized_sha256,
            processing_timestamp=processing_ts,
            sanitized_image_path=saved_media_path,
            latitude=final_lat,
            longitude=final_lon,
            model_execution_records=model_exec_records,
            detections=final_detections,
            segmentation_results=seg_record,
            fused_evidence=fused_evidence_list,
            uncertainty_summary=uncertainty_summary,
            resource_telemetry=resource_telemetry,
            warnings=warnings,
            errors=errors,
            overall_status=overall_status,
            observation_id=obs_id,
            segment_id=seg_id
        )

    def _execute_segmentation(
        self,
        image_bgr: np.ndarray,
        allow_fallback: bool
    ) -> SegmentationResultRecord:
        """
        Executes genuine segmentation model or explicitly reports UNAVAILABLE.
        Never masquerades heuristic contours as neural model inference.
        Records initialization vs inference failures separately.
        """
        h, w = image_bgr.shape[:2]

        init_error: Optional[str] = None
        if hasattr(self.seg_adapter, "load") and not getattr(self.seg_adapter, "is_loaded", False):
            try:
                load_success = self.seg_adapter.load()
                if not load_success:
                    init_error = getattr(self.seg_adapter, "_last_error", None) or "Segmentation model load returned False (weights not found or invalid)."
            except Exception as e:
                init_error = f"Segmentation adapter initialization exception: {type(e).__name__}: {e}"
                logger.error(init_error)

        if hasattr(self.seg_adapter, "model") and self.seg_adapter.model is not None:
            # Genuine neural segmentation model loaded
            try:
                regions = self.seg_adapter.predict_masks(image_bgr)
                regions_dicts = [
                    {
                        "class_id": r.class_id,
                        "class_name": r.class_name,
                        "confidence": r.confidence,
                        "polygon": r.polygon,
                        "area_ratio": r.area_ratio,
                        "area_pixels": r.area_pixels
                    }
                    for r in regions
                ]
                return SegmentationResultRecord(
                    status="READY",
                    model_identifier=self.seg_adapter.name,
                    model_version=getattr(self.seg_adapter, "version", "1.0.0"),
                    execution_type="model_inference",
                    weight_sha256=getattr(self.seg_adapter, "_weight_sha256", None),
                    regions_count=len(regions_dicts),
                    regions=regions_dicts
                )
            except Exception as e:
                msg = str(e).lower()
                name = type(e).__name__.lower()
                is_oom = any(tok in msg or tok in name for tok in (
                    "out of memory", "cuda oom", "cudadrivererror", "resourceexhausted", "memoryerror"
                ))
                fail_status = "RESOURCE_EXHAUSTED" if is_oom else "INFERENCE_FAILED"
                err_msg = f"Segmentation inference exception: {type(e).__name__}: {e}"
                logger.error(err_msg)
                return SegmentationResultRecord(
                    status=fail_status,
                    model_identifier=self.seg_adapter.name,
                    model_version=getattr(self.seg_adapter, "version", "1.0.0"),
                    execution_type="failed",
                    weight_sha256=getattr(self.seg_adapter, "_weight_sha256", None),
                    regions_count=0,
                    regions=[],
                    error_message=err_msg
                )

        if allow_fallback:
            # Explicit development fallback mode requested
            fallback_seg = HeuristicSegmentationAdapter()
            regions = fallback_seg.predict_masks(image_bgr)
            regions_dicts = [
                {
                    "class_id": r.class_id,
                    "class_name": r.class_name,
                    "confidence": r.confidence,
                    "polygon": r.polygon,
                    "area_ratio": r.area_ratio,
                    "area_pixels": r.area_pixels
                }
                for r in regions
            ]
            return SegmentationResultRecord(
                status="DEGRADED",
                model_identifier="heuristic_morphological_fallback",
                model_version="1.0.0",
                execution_type="heuristic_overlay",
                weight_sha256=None,
                regions_count=len(regions_dicts),
                regions=regions_dicts,
                error_message="GENUINE_WEIGHTS_MISSING: operating in development heuristic overlay mode."
            )

        # STRICT_GENUINE mode: missing weights or initialization failure reports UNAVAILABLE without fake masks
        err_msg = (
            f"INITIALIZATION_ERROR: {init_error}"
            if init_error
            else "MODEL_WEIGHTS_NOT_FOUND: No compatible neural segmentation weights loaded. Segmentation skipped."
        )
        return SegmentationResultRecord(
            status="UNAVAILABLE",
            model_identifier=self.seg_adapter.name,
            model_version=getattr(self.seg_adapter, "version", "1.0.0"),
            execution_type="none",
            weight_sha256=None,
            regions_count=0,
            regions=[],
            error_message=err_msg
        )

    def _persist_records(
        self,
        db: Session,
        request: ImageProcessingRequest,
        run_id: str,
        image_id: str,
        saved_media_path: Optional[str],
        latitude: Optional[float],
        longitude: Optional[float],
        orig_sha256: str,
        sanitized_sha256: str,
        final_detections: List[DetectedBarrier],
        seg_record: SegmentationResultRecord,
        model_exec_records: List[ModelExecutionRecord],
        uncertainty_summary: Dict[str, Any]
    ) -> Tuple[str, Optional[str]]:
        """
        Atomically persists Observation, BarrierDetections, SegmentedObservations,
        ImageProvenance, and updates pedestrian segment accessibility scores.
        """
        settings = get_settings()

        # 1. Geographic association (only if coordinates present)
        segment_id: Optional[str] = None
        if latitude is not None and longitude is not None:
            all_segments = db.query(PedestrianSegment).all()
            snap_res = snap_point_to_segments(
                latitude, longitude, all_segments,
                max_distance_meters=settings.SNAP_TOLERANCE_METERS
            )
            if snap_res:
                matched_seg, _ = snap_res
                segment_id = matched_seg.id

        # 2. Create Observation
        obs = Observation(
            id=image_id,
            contributor_id=request.contributor_id,
            segment_id=segment_id,
            media_path=saved_media_path or f"mem://{run_id}.jpg",
            media_type="image",
            latitude=latitude,
            longitude=longitude,
            status="tentative",
            verification_score=0,
            captured_at=datetime.fromisoformat(request.capture_timestamp.replace("Z", "+00:00"))
            if request.capture_timestamp and "T" in request.capture_timestamp
            else datetime.now(timezone.utc)
        )
        db.add(obs)
        db.flush()

        # 3. Create Barrier Detections
        for det in final_detections:
            ymin, xmin, ymax, xmax = det.bbox
            b_det = BarrierDetection(
                id=str(uuid.uuid4()),
                observation_id=obs.id,
                barrier_type=det.barrier_type,
                confidence=det.confidence,
                uncertainty_score=det.uncertainty_score,
                is_confirmed=False,
                bbox_ymin=ymin,
                bbox_xmin=xmin,
                bbox_ymax=ymax,
                bbox_xmax=xmax,
                polygon_json=json.dumps(det.polygon) if det.polygon else None,
                evidence_json=json.dumps(det.evidence) if det.evidence else "{}"
            )
            db.add(b_det)

        # 4. Create Segmented Observations (only if genuine neural or allowed overlay)
        if seg_record.status in ("READY", "DEGRADED") and seg_record.regions:
            for reg in seg_record.regions:
                seg_obs = SegmentedObservation(
                    id=str(uuid.uuid4()),
                    observation_id=obs.id,
                    class_id=reg["class_id"],
                    class_name=reg["class_name"],
                    confidence=reg["confidence"],
                    polygon_json=json.dumps(reg["polygon"]),
                    rle_json=None,
                    area_ratio=reg.get("area_ratio", 0.0),
                    model_identifier=seg_record.model_identifier,
                    model_version=seg_record.model_version
                )
                db.add(seg_obs)

        # 5. Create Image Provenance Record
        raw_meta = {
            "run_id": run_id,
            "original_sha256": orig_sha256,
            "sanitized_sha256": sanitized_sha256,
            "execution_policy": request.execution_policy.value,
            "source_metadata": request.source_metadata,
            "models_executed": [r.__dict__ for r in model_exec_records],
            "segmentation_status": seg_record.status,
            "uncertainty_summary": uncertainty_summary
        }

        resolved_external_id = (
            request.external_image_id
            or request.source_metadata.get("item_id")
            or request.source_metadata.get("external_image_id")
            or request.filename
        )

        provenance = ImageProvenance(
            observation_id=obs.id,
            provider=request.source,
            external_image_id=str(resolved_external_id),
            sequence_id=request.source_metadata.get("sequence_id"),
            capture_timestamp=request.capture_timestamp or datetime.now(timezone.utc).isoformat(),
            ingestion_timestamp=str(int(time.time())),
            latitude=latitude,
            longitude=longitude,
            heading=request.source_metadata.get("heading"),
            source_attribution=request.source_metadata.get("attribution", "AccessGuard Unified Ingress"),
            license_metadata=request.source_metadata.get("license", "Standard Platform License"),
            model_version="hybrid_v2.0.0",
            raw_metadata_json=json.dumps(raw_meta)
        )
        db.add(provenance)

        # 6. Recalculate pedestrian segment score if snapped
        if segment_id is not None:
            seg_obs_list = db.query(Observation).filter(Observation.segment_id == segment_id).all()
            net_score, grade, breakdown = calculate_segment_accessibility_score(seg_obs_list)
            target_seg = db.query(PedestrianSegment).filter(PedestrianSegment.id == segment_id).first()
            if target_seg:
                target_seg.accessibility_score = net_score
                target_seg.score_grade = grade
                target_seg.score_criteria_breakdown = json.dumps(breakdown.model_dump())
                target_seg.barrier_count = sum(len(o.detections) for o in seg_obs_list)

        db.commit()
        return obs.id, segment_id

    @staticmethod
    def _measure_resource_usage() -> Tuple[float, float]:
        """Measures real peak VRAM and host RAM usage."""
        vram_mb = 0.0
        try:
            import torch
            if torch.cuda.is_available():
                vram_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
        except Exception:
            pass

        ram_mb = 0.0
        try:
            import psutil
            process = psutil.Process()
            ram_mb = round(process.memory_info().rss / (1024 * 1024), 2)
        except Exception:
            pass

        return vram_mb, ram_mb

    def _build_failed_result(
        self,
        run_id: str,
        image_id: str,
        request: ImageProcessingRequest,
        error_msg: str,
        processing_ts: str
    ) -> ImageProcessingResult:
        """Returns standard failure container without throwing unhandled exceptions."""
        logger.error(f"Image processing pipeline failure on {request.filename}: {error_msg}")
        return ImageProcessingResult(
            run_id=run_id,
            image_id=image_id,
            source=request.source,
            source_metadata=request.source_metadata,
            original_sha256="",
            sanitized_sha256="",
            processing_timestamp=processing_ts,
            sanitized_image_path=None,
            latitude=request.latitude,
            longitude=request.longitude,
            model_execution_records=[],
            detections=[],
            segmentation_results=SegmentationResultRecord(
                status="UNAVAILABLE",
                model_identifier="none",
                model_version="0.0.0",
                execution_type="none",
                error_message=error_msg
            ),
            fused_evidence=[],
            uncertainty_summary={},
            resource_telemetry={},
            warnings=[],
            errors=[error_msg],
            overall_status="FAILED"
        )

    def process_video(
        self,
        video_path: Union[str, Path],
        source: str = "user_upload_video",
        filename: Optional[str] = None,
        contributor_id: Optional[str] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        sample_fps: float = 1.0,
        max_duration_seconds: int = 60,
        execution_policy: Optional[ExecutionPolicy] = None,
        requested_adapter: Optional[str] = None,
        db: Optional[Session] = None,
        save_media: bool = True
    ) -> VideoProcessingResult:
        """
        Canonical unified video processing engine.
        Extracts keyframes at sample_fps, routes every individual frame through
        the canonical ImageProcessingService.process_image() contract, feeds detections
        into the temporal obstacle tracking layer to suppress duplicates and confirm
        persistent physical obstacles, and records full cryptographic provenance.
        """
        v_path = Path(video_path)
        vid_id = f"vid_{uuid.uuid4().hex[:12]}"
        policy = execution_policy or ExecutionPolicy.STRICT_GENUINE
        warnings: List[str] = []
        errors: List[str] = []

        if not v_path.exists():
            error_msg = f"Video file not found at {v_path}"
            logger.error(error_msg)
            return VideoProcessingResult(
                video_id=vid_id,
                video_path=str(v_path),
                original_sha256="",
                duration_seconds=0.0,
                total_frames=0,
                sampled_frames=0,
                frame_results=[],
                detected_barriers=[],
                unique_obstacles_count=0,
                execution_policy=policy.value,
                resource_telemetry={},
                warnings=[],
                errors=[error_msg],
                overall_status="FAILED"
            )

        # 1. Compute original video SHA-256
        hasher = hashlib.sha256()
        try:
            with open(v_path, "rb") as f:
                while chunk := f.read(65536):
                    hasher.update(chunk)
            video_sha256 = hasher.hexdigest()
        except Exception as e:
            video_sha256 = ""
            warnings.append(f"Failed to compute video SHA-256: {e}")

        # 2. Open video capture
        cap = cv2.VideoCapture(str(v_path))
        if not cap.isOpened():
            error_msg = f"Unable to decode video container at {v_path}"
            logger.error(error_msg)
            return VideoProcessingResult(
                video_id=vid_id,
                video_path=str(v_path),
                original_sha256=video_sha256,
                duration_seconds=0.0,
                total_frames=0,
                sampled_frames=0,
                frame_results=[],
                detected_barriers=[],
                unique_obstacles_count=0,
                execution_policy=policy.value,
                resource_telemetry={},
                warnings=warnings,
                errors=[error_msg],
                overall_status="FAILED"
            )

        native_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / native_fps if native_fps > 0 else 0.0

        if duration > max_duration_seconds:
            cap.release()
            raise ValueError(f"Video duration ({duration:.1f}s) exceeds maximum allowed {max_duration_seconds}s.")

        step = max(1, int(round(native_fps / sample_fps)))
        frame_idx = 0
        sampled_idx = 0
        frame_results: List[ImageProcessingResult] = []

        # Initialize temporal tracker
        tracker = TemporalObstacleTracker(
            iou_match_threshold=0.35,
            max_track_staleness_seconds=3.0,
            min_frames_for_confirmation=3
        )
        all_tracks_map: Dict[str, TrackedBarrier] = {}

        logger.info(f"Unified Video Ingestion: analyzing {v_path.name} ({total_frames} frames @ {native_fps:.1f} fps, step {step})")

        while cap.isOpened() and frame_idx < total_frames:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if frame_idx % step == 0:
                sampled_idx += 1
                current_ts_sec = frame_idx / native_fps if native_fps > 0 else 0.0

                success, encoded_jpg = cv2.imencode(".jpg", frame)
                if not success:
                    warnings.append(f"Failed to encode frame {frame_idx}")
                    frame_idx += 1
                    continue

                frame_req = ImageProcessingRequest(
                    image_bytes=encoded_jpg.tobytes(),
                    filename=f"frame_{frame_idx:06d}.jpg",
                    source=f"{source}_frame",
                    source_metadata={
                        "video_id": vid_id,
                        "video_sha256": video_sha256,
                        "frame_index": frame_idx,
                        "timestamp_seconds": round(current_ts_sec, 3)
                    },
                    latitude=latitude,
                    longitude=longitude,
                    requested_adapter=requested_adapter,
                    execution_policy=policy,
                    enable_privacy_redaction=True,
                    save_media=False
                )

                frame_res = self.process_image(frame_req, db=None)
                frame_results.append(frame_res)
                if frame_res.errors:
                    errors.extend(frame_res.errors)
                if frame_res.warnings:
                    warnings.extend(frame_res.warnings)

                active_tracks, newly_confirmed = tracker.update(
                    frame_res.detections,
                    current_timestamp=current_ts_sec
                )
                for trk in active_tracks:
                    all_tracks_map[trk.track_id] = trk

            frame_idx += 1

        cap.release()

        # 3. Consolidate temporal obstacle tracks into unified DetectedBarriers
        consolidated_barriers: List[DetectedBarrier] = []
        for tid, trk in all_tracks_map.items():
            if trk.frame_count >= tracker.min_frames_for_confirmation or trk.is_confirmed:
                boosted_conf = min(0.96, round(trk.max_confidence + 0.08, 2))
                reduced_unc = max(0.08, round(trk.min_uncertainty - 0.10, 2))
                is_confirmed = True
                expl = f"Verified persistent obstacle across {trk.frame_count} sampled video frames ({trk.duration_seconds:.1f}s)."
            elif trk.frame_count == 1:
                boosted_conf = max(0.35, round(trk.max_confidence - 0.10, 2))
                reduced_unc = min(0.70, round(trk.min_uncertainty + 0.15, 2))
                is_confirmed = False
                expl = f"Single fleeting video frame detection ({trk.duration_seconds:.1f}s)."
            else:
                boosted_conf = trk.max_confidence
                reduced_unc = trk.min_uncertainty
                is_confirmed = False
                expl = f"Observed across {trk.frame_count} sampled video frames ({trk.duration_seconds:.1f}s)."

            evidence = {
                "video_persistence_frames": trk.frame_count,
                "first_seen_sec": round(trk.first_seen_timestamp, 2),
                "last_seen_sec": round(trk.last_seen_timestamp, 2),
                "duration_sec": round(trk.duration_seconds, 2),
                "track_id": trk.track_id,
                "is_confirmed": is_confirmed,
                "explanation": expl
            }

            consolidated_barriers.append(
                DetectedBarrier(
                    barrier_type=trk.barrier_type,
                    confidence=boosted_conf,
                    uncertainty_score=reduced_unc,
                    bbox=trk.bbox,
                    polygon=None,
                    evidence=evidence
                )
            )

        # Telemetry
        vram_mb, ram_mb = self._measure_resource_usage()
        telemetry = {
            "vram_allocated_mb": vram_mb,
            "ram_used_mb": ram_mb,
            "total_frames": total_frames,
            "sampled_frames": sampled_idx,
            "native_fps": native_fps,
            "duration_seconds": round(duration, 2)
        }

        # Determine overall status
        if errors and not consolidated_barriers:
            overall_status = "FAILED"
        elif errors or warnings:
            overall_status = "PARTIAL"
        else:
            overall_status = "SUCCESS"

        # 4. Database Persistence
        obs_id: Optional[str] = None
        segment_id: Optional[str] = None

        if db is not None and save_media:
            obs_id = vid_id
            if latitude is not None and longitude is not None:
                try:
                    all_segments = db.query(PedestrianSegment).all()
                    snap_res = snap_point_to_segments(
                        latitude, longitude, all_segments, max_distance_meters=50.0
                    )
                    if snap_res:
                        matched_seg, _ = snap_res
                        segment_id = matched_seg.id
                except Exception as e:
                    warnings.append(f"Spatial snapping failed: {e}")

            obs = Observation(
                id=obs_id,
                contributor_id=contributor_id,
                media_path=str(v_path),
                media_type="video",
                latitude=latitude,
                longitude=longitude,
                segment_id=segment_id,
                status="tentative",
                verification_score=0,
                captured_at=datetime.now(timezone.utc)
            )
            db.add(obs)
            db.flush()

            for det in consolidated_barriers:
                ymin, xmin, ymax, xmax = det.bbox
                b_det = BarrierDetection(
                    id=str(uuid.uuid4()),
                    observation_id=obs.id,
                    barrier_type=det.barrier_type,
                    confidence=det.confidence,
                    uncertainty_score=det.uncertainty_score,
                    is_confirmed=det.is_confirmed,
                    bbox_ymin=ymin,
                    bbox_xmin=xmin,
                    bbox_ymax=ymax,
                    bbox_xmax=xmax,
                    polygon_json=json.dumps(det.polygon) if det.polygon else None,
                    evidence_json=json.dumps(det.evidence) if det.evidence else "{}"
                )
                db.add(b_det)

            raw_meta = {
                "video_id": vid_id,
                "original_sha256": video_sha256,
                "execution_policy": policy.value,
                "total_frames": total_frames,
                "sampled_frames": sampled_idx,
                "duration_seconds": round(duration, 2),
                "unique_obstacles_count": len(consolidated_barriers),
                "frames_summary": [
                    {
                        "run_id": fr.run_id,
                        "detections_count": len(fr.detections),
                        "overall_status": fr.overall_status
                    }
                    for fr in frame_results
                ]
            }

            provenance = ImageProvenance(
                observation_id=obs.id,
                provider=source,
                external_image_id=filename or v_path.name,
                capture_timestamp=datetime.now(timezone.utc).isoformat(),
                ingestion_timestamp=str(int(time.time())),
                latitude=latitude,
                longitude=longitude,
                source_attribution="AccessGuard Unified Video Ingress",
                license_metadata="Standard Platform License",
                model_version="unified_video_v2.0.0",
                raw_metadata_json=json.dumps(raw_meta)
            )
            db.add(provenance)

            if segment_id is not None:
                seg_obs_list = db.query(Observation).filter(Observation.segment_id == segment_id).all()
                net_score, grade, breakdown = calculate_segment_accessibility_score(seg_obs_list)
                target_seg = db.query(PedestrianSegment).filter(PedestrianSegment.id == segment_id).first()
                if target_seg:
                    target_seg.accessibility_score = net_score
                    target_seg.score_grade = grade
                    target_seg.score_criteria_breakdown = json.dumps(breakdown.model_dump())
                    target_seg.barrier_count = sum(len(o.detections) for o in seg_obs_list)

            db.commit()

        return VideoProcessingResult(
            video_id=vid_id,
            video_path=str(v_path),
            original_sha256=video_sha256,
            duration_seconds=round(duration, 2),
            total_frames=total_frames,
            sampled_frames=sampled_idx,
            frame_results=frame_results,
            detected_barriers=consolidated_barriers,
            unique_obstacles_count=len(consolidated_barriers),
            execution_policy=policy.value,
            resource_telemetry=telemetry,
            warnings=warnings,
            errors=errors,
            overall_status=overall_status,
            observation_id=obs_id,
            segment_id=segment_id
        )
