"""
Canonical video upload and background processing API.

Every uploaded video is processed by ImageProcessingService.process_video,
which sends sampled frames through the same genuine inference path used by
single-image uploads and persists the video observation plus provenance.
"""
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logging import logger
from app.core.security import get_optional_user_payload
from app.db.session import SessionLocal
from app.cv.adapters.base_v2 import ExecutionPolicy
from app.cv.orchestration.image_processing_service import ImageProcessingService
from app.ingestion.connectors.base import MAX_VIDEO_BYTES

router = APIRouter(prefix="/video", tags=["Video"])
UPLOAD_DIR = Path("data/uploads/videos")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# In-process job registry. The durable observation/provenance record is stored in the database.
_video_jobs: Dict[str, Dict[str, Any]] = {}


def _summarize_model_execution(frame_results) -> list:
    aggregate: Dict[str, Dict[str, Any]] = {}
    for frame in frame_results:
        for record in frame.model_execution_records:
            item = aggregate.setdefault(record.adapter_name, {
                "adapter_name": record.adapter_name,
                "model_family": record.model_family,
                "status_counts": {},
                "execution_types": {},
                "frames_reported": 0,
                "detections_reported": 0,
                "errors": []
            })
            item["status_counts"][record.status] = item["status_counts"].get(record.status, 0) + 1
            item["execution_types"][record.execution_type] = item["execution_types"].get(record.execution_type, 0) + 1
            item["frames_reported"] += 1
            item["detections_reported"] += record.output_count
            item["errors"].extend(record.errors or [])
    return list(aggregate.values())


def run_video_job(
    job_id: str,
    file_path: str,
    original_filename: str,
    contributor_id: Optional[str],
    latitude: Optional[float],
    longitude: Optional[float],
    requested_adapter: Optional[str]
):
    """Run canonical video inference in a background worker and persist its observation."""
    job = _video_jobs[job_id]
    job["status"] = "PROCESSING"
    db: Session = SessionLocal()
    try:
        policy = ExecutionPolicy.DEVELOPMENT_FALLBACK if requested_adapter in ("heuristic", "fallback") else ExecutionPolicy.STRICT_GENUINE
        result = ImageProcessingService().process_video(
            video_path=file_path,
            source="user_upload_video",
            filename=original_filename,
            contributor_id=contributor_id,
            latitude=latitude,
            longitude=longitude,
            sample_fps=1.0,
            max_duration_seconds=settings.MAX_VIDEO_DURATION_SECONDS,
            execution_policy=policy,
            requested_adapter=requested_adapter,
            db=db,
            save_media=True
        )

        tracks = []
        for detection in result.detected_barriers:
            evidence = detection.evidence or {}
            tracks.append({
                "track_id": evidence.get("track_id", f"video-detection-{len(tracks) + 1}"),
                "barrier_type": detection.barrier_type,
                "first_seen_sec": evidence.get("first_seen_sec", 0.0),
                "last_seen_sec": evidence.get("last_seen_sec", 0.0),
                "duration_sec": evidence.get("duration_sec", 0.0),
                "frame_count": evidence.get("video_persistence_frames", 1),
                "max_confidence": detection.confidence,
                "bbox": detection.bbox,
                "is_physical_obstacle": bool(evidence.get("is_confirmed", False))
            })

        job.update({
            "status": "COMPLETED" if result.overall_status in ("SUCCESS", "PARTIAL") else "FAILED",
            "progress_percent": 100.0,
            "total_frames": result.total_frames,
            "sampled_frames": result.sampled_frames,
            "processed_frames": result.sampled_frames,
            "duration_seconds": result.duration_seconds,
            "native_fps": result.resource_telemetry.get("native_fps", 0.0),
            "unique_obstacles_count": result.unique_obstacles_count,
            "confirmed_tracks": tracks,
            "observation_id": result.observation_id,
            "overall_status": result.overall_status,
            "model_execution_summary": _summarize_model_execution(result.frame_results),
            "warnings": result.warnings,
            "errors": result.errors,
            "error_message": "; ".join(result.errors) if result.overall_status == "FAILED" else None
        })
    except Exception as exc:
        logger.exception("Canonical video job %s failed", job_id)
        job.update({
            "status": "FAILED",
            "progress_percent": 100.0,
            "error_message": str(exc),
            "errors": [str(exc)]
        })
        db.rollback()
    finally:
        db.close()


@router.post("/process", status_code=status.HTTP_202_ACCEPTED)
async def upload_and_process_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
    adapter: Optional[str] = Form("unified"),
    user_payload: Optional[dict] = Depends(get_optional_user_payload)
):
    """Queue a real video for canonical frame inference, temporal tracking, and persistence."""
    filename = file.filename or "video.mp4"
    ext = Path(filename).suffix.lower()
    if ext not in (".mp4", ".webm", ".avi", ".mov", ".m4v"):
        raise HTTPException(status_code=400, detail="Invalid video format. Supported: mp4, webm, avi, mov, m4v")
    if adapter == "dummy":
        raise HTTPException(status_code=400, detail="The deterministic dummy adapter cannot process user videos.")

    job_id = f"vid_{uuid.uuid4().hex[:12]}"
    save_path = UPLOAD_DIR / f"{job_id}{ext}"
    size = 0
    try:
        with open(save_path, "wb") as output:
            while True:
                chunk = await file.read(65536)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_VIDEO_BYTES:
                    raise HTTPException(status_code=413, detail=f"Video exceeds maximum size limit of {MAX_VIDEO_BYTES} bytes.")
                output.write(chunk)
    except Exception:
        save_path.unlink(missing_ok=True)
        raise

    _video_jobs[job_id] = {
        "job_id": job_id,
        "status": "PENDING",
        "progress_percent": 0.0,
        "total_frames": 0,
        "sampled_frames": 0,
        "processed_frames": 0,
        "duration_seconds": 0.0,
        "native_fps": 0.0,
        "unique_obstacles_count": 0,
        "confirmed_tracks": [],
        "observation_id": None,
        "overall_status": None,
        "model_execution_summary": [],
        "warnings": [],
        "errors": [],
        "error_message": None
    }
    contributor_id = user_payload.get("sub") if user_payload else None
    background_tasks.add_task(
        run_video_job,
        job_id,
        str(save_path),
        filename,
        contributor_id,
        latitude,
        longitude,
        adapter
    )
    return {
        "job_id": job_id,
        "status": "PENDING",
        "message": "Video queued for canonical genuine image-frame inference and temporal tracking.",
        "file_size_bytes": size
    }


@router.get("/jobs/{job_id}")
def get_video_job_status(job_id: str):
    """Return real processing status, persisted observation ID, and model execution telemetry."""
    job = _video_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Video processing job not found.")
    return job
