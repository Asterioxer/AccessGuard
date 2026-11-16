import os
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "AccessGuard"
    VERSION: str = "1.3.0"
    API_V1_STR: str = "/api/v1"
    
    # Environment
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    
    # Security
    SECRET_KEY: str = "accessguard-development-secret-key-replace-in-production-32bytesmin"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours
    
    # CORS
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://localhost:8080",
    ]
    
    # Database
    # Default to sqlite:///./accessguard.db for instant zero-dependency testing, or postgresql:// for production PostGIS
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./accessguard.db")
    
    # Media Storage
    UPLOAD_DIR: Path = Path("./data/uploads")
    PROCESSED_DIR: Path = Path("./data/processed")
    MAX_IMAGE_SIZE_BYTES: int = 20 * 1024 * 1024   # 20 MB
    MAX_VIDEO_SIZE_BYTES: int = 100 * 1024 * 1024  # 100 MB
    MAX_VIDEO_DURATION_SECONDS: int = 60
    
    # Privacy & Anonymization
    ENABLE_PRIVACY_BLURRING: bool = True
    STRIP_EXIF_AFTER_EXTRACTION: bool = True
    MINIMIZE_MEDIA_RETENTION: bool = True
    MEDIA_RETENTION_HOURS: int = 72
    
    # CV Pipeline Settings
    CV_ADAPTER: str = "heuristic"  # "heuristic", "torchvision", "yolo", "dummy", "hybrid"
    CONFIDENCE_TENTATIVE_THRESHOLD: float = 0.50
    CONFIDENCE_CONFIRMED_THRESHOLD: float = 0.85
    UNCERTAINTY_FLAG_THRESHOLD: float = 0.35
    
    # Hybrid Vision Intelligence (Mission 006)
    HYBRID_MODE: str = os.getenv("HYBRID_MODE", "RESOURCE_CONSTRAINED") # "RESOURCE_CONSTRAINED" vs "FULL_HYBRID"
    ENABLE_VALIDATOR_FILTERING: bool = True
    ENABLE_SEAM_SUPPRESSION: bool = True
    GPU_MEMORY_LIMIT_MB: int = 3500
    
    # Verification & Consensus
    MIN_VERIFICATIONS_TO_CONFIRM: int = 3
    CONSENSUS_AGREEMENT_RATIO: float = 0.67
    
    # GIS & Scoring
    PEDESTRIAN_BUFFER_METERS: float = 15.0
    SNAP_TOLERANCE_METERS: float = 25.0
    
    model_config = {
        "env_file": ".env",
        "case_sensitive": True,
        "extra": "ignore"
    }

settings = Settings()

if settings.ENVIRONMENT.lower() == "production" and ("development" in settings.SECRET_KEY.lower() or len(settings.SECRET_KEY) < 32):
    raise RuntimeError("CRITICAL SECURITY ERROR: Production deployment requires an explicitly configured SECRET_KEY of at least 32 characters.")

# Ensure directories exist
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
