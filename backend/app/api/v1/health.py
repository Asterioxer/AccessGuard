import time
from typing import Dict, Any
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.db.session import get_db
from app.config import settings

router = APIRouter(tags=["Health & Monitoring"])

START_TIME = time.time()

@router.get("/health")
def health_check(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Liveness and readiness health check."""
    db_ok = True
    try:
        db.execute(text("SELECT 1"))
    except Exception:
        db_ok = False

    return {
        "status": "healthy" if db_ok else "degraded",
        "database_connected": db_ok,
        "version": settings.VERSION,
        "active_cv_adapter": settings.CV_ADAPTER,
        "uptime_seconds": round(time.time() - START_TIME, 1)
    }

@router.get("/metrics")
def system_metrics(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Operational monitoring metrics."""
    return {
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "privacy_blurring_enabled": settings.ENABLE_PRIVACY_BLURRING,
        "exif_stripping_enabled": settings.STRIP_EXIF_AFTER_EXTRACTION,
        "max_image_size_mb": settings.MAX_IMAGE_SIZE_BYTES / (1024 * 1024),
        "max_video_size_mb": settings.MAX_VIDEO_SIZE_BYTES / (1024 * 1024)
    }
