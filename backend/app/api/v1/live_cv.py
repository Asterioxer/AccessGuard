"""
Live Camera & Video Streaming CV Inference Endpoints.

Provides real-time inference for browser webcam feeds, video keyframes,
temporal deduplication, latency metrics, and adapter switching.
"""

import time
import base64
import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Depends
from pydantic import BaseModel, Field

from app.cv.pipeline import CVPipeline, preprocess_image, blur_sensitive_regions
from app.cv.temporal_tracker import TemporalObstacleTracker
from app.cv.base import DetectedBarrier
from app.cv.adapters.heuristic_adapter import HeuristicCVAdapter
from app.cv.adapters.torchvision_adapter import TorchvisionAdapter
from app.cv.adapters.yolo_adapter import YOLOAdapter
from app.cv.adapters.onnx_adapter import ONNXAdapter
from app.core.logging import logger

router = APIRouter(prefix="/cv", tags=["Computer Vision Live Inference"])

# Global session tracker cache keyed by stream_id
_stream_trackers: Dict[str, TemporalObstacleTracker] = {}

# Active adapter singleton
_available_adapters = {
    "heuristic": HeuristicCVAdapter(),
    "torchvision": TorchvisionAdapter(),
    "yolo": YOLOAdapter(),
    "onnx": ONNXAdapter("data/models/yolov8n.onnx")
}
_active_adapter_name = "heuristic"

class LiveDetectionDTO(BaseModel):
    barrier_type: str
    confidence: float
    uncertainty_score: float
    bbox: Tuple[float, float, float, float] # ymin, xmin, ymax, xmax
    is_uncertain: bool
    is_confirmed: bool
    track_id: Optional[str] = None
    evidence: Dict[str, Any] = Field(default_factory=dict)

class LiveFrameResponse(BaseModel):
    detections: List[LiveDetectionDTO]
    latency_ms: float
    fps_estimate: float
    adapter_used: str
    image_quality: Dict[str, Any]
    privacy_redacted: bool
    active_tracks_count: int

class SetAdapterRequest(BaseModel):
    adapter_name: str

@router.get("/adapters")
def list_adapters():
    """Lists available model adapters and the currently active one."""
    return {
        "available_adapters": list(_available_adapters.keys()),
        "active_adapter": _active_adapter_name,
        "onnx_available": _available_adapters["onnx"].is_initialized or _available_adapters["onnx"].load()
    }

@router.post("/active-adapter")
def set_active_adapter(req: SetAdapterRequest):
    """Sets the active adapter for live and batch inference."""
    global _active_adapter_name
    name = req.adapter_name.lower().strip()
    if name not in _available_adapters:
        raise HTTPException(status_code=400, detail=f"Adapter '{name}' not found. Choose from {list(_available_adapters.keys())}")
    
    _active_adapter_name = name
    adp = _available_adapters[name]
    if not adp.is_initialized:
        adp.load()
    logger.info(f"Switched active CV adapter to: {name}")
    return {"status": "success", "active_adapter": _active_adapter_name}

@router.post("/live-frame", response_model=LiveFrameResponse)
async def process_live_frame(
    file: Optional[UploadFile] = File(None),
    base64_image: Optional[str] = Form(None),
    stream_id: Optional[str] = Form(None),
    timestamp: Optional[float] = Form(None)
):
    """
    Processes a live camera frame.
    Applies image preprocessing (blur detection, CLAHE, privacy redaction),
    runs active CV model adapter, matches temporal tracks to prevent duplication,
    and returns bounding boxes with inference latency metrics.
    """
    t_start = time.perf_counter()

    # Read image bytes from multipart upload or base64 data URL
    if file:
        content = await file.read()
    elif base64_image:
        try:
            # Strip data:image/...;base64, prefix if present
            if "," in base64_image:
                base64_image = base64_image.split(",", 1)[1]
            content = base64.b64decode(base64_image)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {e}")
    else:
        raise HTTPException(status_code=400, detail="Either 'file' or 'base64_image' must be provided.")

    # Decode OpenCV BGR image
    nparr = np.frombuffer(content, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode image from provided payload.")

    # 1. Preprocessing: Blur detection and adaptive CLAHE contrast
    proc_img, is_blurry, blur_var, clahe_applied = preprocess_image(img_bgr)

    # 2. Privacy redaction: Gaussian blur on faces/plates
    redacted_img = blur_sensitive_regions(proc_img)

    # 3. Model inference using active adapter
    adapter = _available_adapters.get(_active_adapter_name, _available_adapters["heuristic"])
    if not adapter.is_initialized:
        adapter.load()

    raw_detections = adapter.predict(redacted_img)

    # If blur penalty applies
    if is_blurry:
        for d in raw_detections:
            d.uncertainty_score = min(1.0, round(d.uncertainty_score + 0.35, 4))
            d.confidence = max(0.1, round(d.confidence - 0.25, 4))
            d.evidence["blur_variance"] = round(blur_var, 2)
            d.evidence["quality_warning"] = "High motion blur detected; confidence penalized"

    # 4. Temporal tracking and deduplication
    now_ts = timestamp if timestamp is not None else time.time()
    sid = stream_id or "default_stream"
    if sid not in _stream_trackers:
        _stream_trackers[sid] = TemporalObstacleTracker()

    tracker = _stream_trackers[sid]
    all_tracks, _ = tracker.update(raw_detections, current_timestamp=now_ts)

    # Format response DTOs
    out_detections: List[LiveDetectionDTO] = []
    for d in raw_detections:
        # Match track_id if available
        matched_tid = None
        for trk in all_tracks:
            if trk.barrier_type == d.barrier_type:
                matched_tid = trk.track_id
                break

        out_detections.append(
            LiveDetectionDTO(
                barrier_type=d.barrier_type,
                confidence=d.confidence,
                uncertainty_score=d.uncertainty_score,
                bbox=d.bbox,
                is_uncertain=d.is_uncertain,
                is_confirmed=d.is_confirmed,
                track_id=matched_tid,
                evidence=d.evidence
            )
        )

    t_end = time.perf_counter()
    latency_ms = round((t_end - t_start) * 1000.0, 2)
    fps_est = round(1000.0 / latency_ms, 1) if latency_ms > 0 else 30.0

    return LiveFrameResponse(
        detections=out_detections,
        latency_ms=latency_ms,
        fps_estimate=fps_est,
        adapter_used=adapter.name,
        image_quality={
            "is_blurry": is_blurry,
            "blur_variance": round(blur_var, 2),
            "clahe_applied": clahe_applied,
            "resolution": f"{img_bgr.shape[1]}x{img_bgr.shape[0]}"
        },
        privacy_redacted=True,
        active_tracks_count=len(all_tracks)
    )
