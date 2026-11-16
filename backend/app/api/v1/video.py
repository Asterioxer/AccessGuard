"""
Video Intelligence and Asynchronous Deduplication API.
"""

import os
import uuid
import shutil
from pathlib import Path
from typing import Any, Dict
from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks, status

from app.cv.video import VideoProcessor, VideoProcessingProgress
from app.ingestion.connectors.base import MAX_VIDEO_BYTES
from app.core.logging import logger

router = APIRouter(prefix="/video", tags=["Video"])
processor = VideoProcessor()
UPLOAD_DIR = Path("data/uploads/videos")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def run_video_job(job_id: str, file_path: str):
    """Background task running video frame extraction and temporal obstacle deduplication."""
    try:
        processor.process_video_file(file_path, job_id=job_id)
    except Exception as e:
        logger.error(f"Error processing background video job {job_id}: {e}")


@router.post("/process", status_code=status.HTTP_202_ACCEPTED)
async def upload_and_process_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Uploads a pedestrian walkway video for asynchronous frame extraction,
    temporal tracking, and physical obstacle deduplication.
    """
    ext = Path(file.filename or "video.mp4").suffix.lower()
    if ext not in [".mp4", ".webm", ".avi", ".mov"]:
        raise HTTPException(status_code=400, detail="Invalid video format. Supported: mp4, webm, avi, mov")

    job_id = f"vid_{uuid.uuid4().hex[:8]}"
    save_path = UPLOAD_DIR / f"{job_id}{ext}"

    # Stream file to disk while enforcing max size limit
    size = 0
    with open(save_path, "wb") as f_out:
        while True:
            chunk = await file.read(65536)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_VIDEO_BYTES:
                save_path.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail=f"Video exceeds maximum size limit of {MAX_VIDEO_BYTES} bytes.")
            f_out.write(chunk)

    # Initialize job entry
    progress = VideoProcessingProgress(
        job_id=job_id,
        status="PENDING",
        progress_percent=0.0
    )
    processor._jobs[job_id] = progress

    # Dispatch to background task
    background_tasks.add_task(run_video_job, job_id, str(save_path))

    return {
        "job_id": job_id,
        "status": "PENDING",
        "message": "Video queued for temporal deduplication and obstacle tracking.",
        "file_size_bytes": size
    }


@router.get("/jobs/{job_id}")
def get_video_job_status(job_id: str):
    """Returns video processing telemetry, progress, and confirmed unique obstacles."""
    prog = processor.get_job_progress(job_id)
    if not prog:
        raise HTTPException(status_code=404, detail="Video processing job not found.")

    return {
        "job_id": prog.job_id,
        "status": prog.status,
        "progress_percent": prog.progress_percent,
        "total_frames": prog.total_frames,
        "sampled_frames": prog.sampled_frames,
        "duration_seconds": prog.duration_seconds,
        "native_fps": prog.fps,
        "unique_obstacles_count": prog.unique_obstacles_count,
        "confirmed_tracks": prog.confirmed_tracks,
        "error_message": prog.error_message
    }
