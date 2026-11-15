import os
import uuid
import json
from datetime import datetime, timezone
from typing import Optional, List
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query, status
from sqlalchemy.orm import Session
import cv2

from app.db.session import get_db
from app.db.models import Observation, BarrierDetection, PedestrianSegment, User
from app.schemas.observation import (
    ObservationResponse, BarrierDetectionSchema, BoundingBox, BarrierEvidence
)
from app.core.config_wrapper import get_settings
from app.core.security import get_optional_user_payload
from app.core.sanitize import (
    validate_upload, extract_exif_gps, strip_exif_and_reencode, apply_privacy_redaction
)
from app.cv.pipeline import CVPipeline
from app.gis.spatial_index import snap_point_to_segments
from app.gis.scoring import calculate_segment_accessibility_score
from app.core.logging import logger

router = APIRouter(prefix="/observations", tags=["Observations"])

def format_detection_schema(det: BarrierDetection) -> BarrierDetectionSchema:
    evidence_dict = {}
    if det.evidence_json:
        try:
            evidence_dict = json.loads(det.evidence_json)
        except Exception:
            pass

    evidence = None
    if evidence_dict:
        evidence = BarrierEvidence(
            feature_type=evidence_dict.get("feature_type", "visual_cue"),
            measured_metric=float(evidence_dict.get("measured_metric", 0.0)),
            threshold_applied=float(evidence_dict.get("threshold_applied", 0.0)),
            explanation=evidence_dict.get("explanation", ""),
            metadata=evidence_dict
        )

    polygon = None
    if det.polygon_json:
        try:
            polygon = json.loads(det.polygon_json)
        except Exception:
            pass

    is_uncertain = (det.uncertainty_score > 0.35) or (0.45 <= det.confidence < 0.70)

    return BarrierDetectionSchema(
        id=det.id,
        barrier_type=det.barrier_type,
        confidence=det.confidence,
        uncertainty_score=det.uncertainty_score,
        is_confirmed=det.is_confirmed,
        is_uncertain=is_uncertain,
        bbox=BoundingBox(
            ymin=det.bbox_ymin,
            xmin=det.bbox_xmin,
            ymax=det.bbox_ymax,
            xmax=det.bbox_xmax
        ),
        polygon=polygon,
        evidence=evidence,
        created_at=det.created_at
    )

def format_observation_response(obs: Observation) -> ObservationResponse:
    detections = [format_detection_schema(d) for d in obs.detections]
    return ObservationResponse(
        id=obs.id,
        contributor_id=obs.contributor_id,
        segment_id=obs.segment_id,
        media_url=f"/api/v1/observations/{obs.id}/media",
        media_type=obs.media_type,
        latitude=obs.latitude,
        longitude=obs.longitude,
        status=obs.status,
        verification_score=obs.verification_score,
        captured_at=obs.captured_at,
        created_at=obs.created_at,
        detections=detections
    )

@router.post("/upload", response_model=ObservationResponse, status_code=status.HTTP_201_CREATED)
async def upload_observation(
    file: UploadFile = File(...),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
    adapter: Optional[str] = Form(None),
    user_payload: Optional[dict] = Depends(get_optional_user_payload),
    db: Session = Depends(get_db)
):
    """
    Accepts image or short video upload:
    1. Validates magic bytes and file limits.
    2. Extracts EXIF GPS if available, then strips metadata for privacy.
    3. Runs privacy redaction (blurring faces/plates).
    4. Executes interchangeable CV model adapter.
    5. Associates detection with nearest pedestrian segment and calculates accessibility scores.
    """
    settings = get_settings()
    file_bytes = await file.read()
    
    # 1. Validation
    max_size = settings.MAX_VIDEO_SIZE_BYTES if file.filename.lower().endswith(('.mp4', '.mov', '.m4v')) else settings.MAX_IMAGE_SIZE_BYTES
    is_valid, mime_type, media_category = validate_upload(file_bytes, file.filename, max_size)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Upload validation failed: {mime_type}"
        )

    # 2. Coordinate resolution
    final_lat = latitude
    final_lon = longitude
    
    if media_category == "image":
        exif_lat, exif_lon = extract_exif_gps(file_bytes)
        if final_lat is None and exif_lat is not None:
            final_lat = exif_lat
            final_lon = exif_lon

    # Default to downtown demo coordinates if no GPS provided
    if final_lat is None or final_lon is None:
        final_lat = 37.7765
        final_lon = -122.4194

    obs_id = str(uuid.uuid4())
    ext = file.filename.split(".")[-1].lower() if "." in file.filename else "jpg"
    filename = f"{obs_id}.{ext}"
    saved_path = settings.PROCESSED_DIR / filename

    pipeline = CVPipeline(adapter_name=adapter or settings.CV_ADAPTER)
    detected_barriers = []

    if media_category == "image":
        # Privacy redaction: strip EXIF and re-encode
        clean_bytes = strip_exif_and_reencode(file_bytes, output_format=ext)
        
        # Run CV Pipeline
        detections, img_bgr = pipeline.process_image(clean_bytes)
        detected_barriers = detections
        
        # Blur faces/plates if privacy enabled
        if settings.ENABLE_PRIVACY_BLURRING and img_bgr is not None:
            img_bgr = apply_privacy_redaction(img_bgr)
            cv2.imwrite(str(saved_path), img_bgr)
        else:
            with open(saved_path, "wb") as f:
                f.write(clean_bytes)
    else:
        # Video pipeline
        temp_video_path = settings.UPLOAD_DIR / f"temp_{filename}"
        with open(temp_video_path, "wb") as f:
            f.write(file_bytes)
            
        try:
            detected_barriers = pipeline.process_video(
                str(temp_video_path),
                max_duration_seconds=settings.MAX_VIDEO_DURATION_SECONDS
            )
            # Rename temp to processed
            if os.path.exists(saved_path):
                os.remove(saved_path)
            os.rename(temp_video_path, saved_path)
        except Exception as e:
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
            raise HTTPException(status_code=400, detail=f"Video processing failed: {str(e)}")

    # 3. Create Observation record
    contributor_id = user_payload.get("sub") if user_payload else None
    observation = Observation(
        id=obs_id,
        contributor_id=contributor_id,
        media_path=str(saved_path),
        media_type=media_category,
        latitude=final_lat,
        longitude=final_lon,
        status="tentative",
        verification_score=0,
        captured_at=datetime.now(timezone.utc)
    )
    db.add(observation)

    # 4. Attach Barrier Detections
    for det in detected_barriers:
        poly_str = json.dumps(det.polygon) if det.polygon else None
        evidence_str = json.dumps(det.evidence) if det.evidence else "{}"
        
        db_det = BarrierDetection(
            id=str(uuid.uuid4()),
            observation_id=obs_id,
            barrier_type=det.barrier_type,
            confidence=det.confidence,
            uncertainty_score=det.uncertainty_score,
            is_confirmed=det.is_confirmed,
            bbox_ymin=det.bbox[0],
            bbox_xmin=det.bbox[1],
            bbox_ymax=det.bbox[2],
            bbox_xmax=det.bbox[3],
            polygon_json=poly_str,
            evidence_json=evidence_str
        )
        db.add(db_det)

    db.commit()

    # 5. Snap to nearest pedestrian segment & update score
    all_segments = db.query(PedestrianSegment).all()
    snap_result = snap_point_to_segments(final_lat, final_lon, all_segments, max_distance_meters=settings.SNAP_TOLERANCE_METERS)
    
    if snap_result:
        matched_seg, _ = snap_result
        observation.segment_id = matched_seg.id
        db.commit()
        
        # Recalculate segment score
        seg_obs = db.query(Observation).filter(Observation.segment_id == matched_seg.id).all()
        net_score, grade, breakdown = calculate_segment_accessibility_score(seg_obs)
        matched_seg.accessibility_score = net_score
        matched_seg.score_grade = grade
        matched_seg.score_criteria_breakdown = json.dumps(breakdown.model_dump())
        matched_seg.barrier_count = sum(len(o.detections) for o in seg_obs)
        matched_seg.last_evaluated_at = datetime.now(timezone.utc)
        db.commit()

    db.refresh(observation)
    return format_observation_response(observation)

@router.get("", response_model=List[ObservationResponse])
def list_observations(
    status: Optional[str] = Query(None),
    barrier_type: Optional[str] = Query(None),
    segment_id: Optional[str] = Query(None),
    min_confidence: Optional[float] = Query(None),
    limit: int = Query(50, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db)
):
    """Lists observations with optional geospatial and barrier filters."""
    query = db.query(Observation)
    if status:
        query = query.filter(Observation.status == status)
    if segment_id:
        query = query.filter(Observation.segment_id == segment_id)
        
    observations = query.order_by(Observation.created_at.desc()).offset(offset).limit(limit).all()
    
    # Filter by barrier_type or min_confidence in memory if specified
    results = []
    for obs in observations:
        if barrier_type or min_confidence:
            matched_dets = [
                d for d in obs.detections
                if (not barrier_type or d.barrier_type == barrier_type) and
                   (min_confidence is None or d.confidence >= min_confidence)
            ]
            if not matched_dets:
                continue
        results.append(format_observation_response(obs))
        
    return results

@router.get("/{observation_id}", response_model=ObservationResponse)
def get_observation_detail(observation_id: str, db: Session = Depends(get_db)):
    """Retrieves full details of an observation, including detections and evidence."""
    obs = db.query(Observation).filter(Observation.id == observation_id).first()
    if not obs:
        raise HTTPException(status_code=404, detail="Observation not found")
    return format_observation_response(obs)

@router.get("/{observation_id}/media")
def get_observation_media(observation_id: str, db: Session = Depends(get_db)):
    """Streams stored image or video media."""
    from fastapi.responses import FileResponse
    obs = db.query(Observation).filter(Observation.id == observation_id).first()
    if not obs or not os.path.exists(obs.media_path):
        raise HTTPException(status_code=404, detail="Media file not found")
        
    media_type = "image/jpeg" if obs.media_type == "image" else "video/mp4"
    return FileResponse(obs.media_path, media_type=media_type)
