from contextlib import asynccontextmanager
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, Counter, Histogram

from app.config import settings
from app.core.limiter import limiter
from app.core.logging import setup_logging, logger
from app.db.session import engine, Base, SessionLocal
from app.db.models import User
from app.core.security import get_password_hash

# Routers
from app.api.v1.auth import router as auth_router
from app.api.v1.observations import router as obs_router
from app.api.v1.segments import router as seg_router
from app.api.v1.routing import router as routing_router
from app.api.v1.verifications import router as verify_router
from app.api.v1.analytics import router as analytics_router
from app.api.v1.health import router as health_router
from app.api.v1.live_cv import router as live_cv_router
from app.api.v1.ingestion import router as ingestion_router
from app.api.v1.segmentation import router as segmentation_router
from app.api.v1.video import router as video_router
from app.api.v1.reality import router as reality_router

setup_logging()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup initializes schema and only creates accounts when explicit bootstrap secrets are configured.
    # No fictional streets, sample media, or pre-labelled observations are inserted here.
    logger.info("Initializing AccessGuard database schema...")
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        admin = db.query(User).filter(User.email == "admin@accessguard.org").first()
        if not admin and settings.ADMIN_INITIAL_PASSWORD:
            db.add(User(
                email="admin@accessguard.org",
                hashed_password=get_password_hash(settings.ADMIN_INITIAL_PASSWORD),
                role="admin",
                reputation_score=100
            ))
            db.commit()
            logger.info("Administrator account created from ADMIN_INITIAL_PASSWORD.")

        verifier = db.query(User).filter(User.email == "verifier@accessguard.org").first()
        if not verifier and settings.VERIFIER_INITIAL_PASSWORD:
            db.add(User(
                email="verifier@accessguard.org",
                hashed_password=get_password_hash(settings.VERIFIER_INITIAL_PASSWORD),
                role="verifier",
                reputation_score=50
            ))
            db.commit()
            logger.info("Verifier account created from VERIFIER_INITIAL_PASSWORD.")

        if not admin and not settings.ADMIN_INITIAL_PASSWORD:
            logger.warning("No administrator account exists. Configure ADMIN_INITIAL_PASSWORD to bootstrap one.")
    finally:
        db.close()
        
    yield
    logger.info("Shutting down AccessGuard API service...")

app = FastAPI(
    title="AccessGuard Intelligence Platform API",
    version=settings.VERSION,
    description="Public-benefit accessibility intelligence platform analyzing street-level imagery for wheelchair accessibility barriers.",
    lifespan=lifespan
)

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS if settings.BACKEND_CORS_ORIGINS else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API v1 Routers
app.include_router(health_router)
app.include_router(auth_router, prefix=settings.API_V1_STR)
app.include_router(obs_router, prefix=settings.API_V1_STR)
app.include_router(seg_router, prefix=settings.API_V1_STR)
app.include_router(routing_router, prefix=settings.API_V1_STR)
app.include_router(verify_router, prefix=settings.API_V1_STR)
app.include_router(analytics_router, prefix=settings.API_V1_STR)
app.include_router(live_cv_router, prefix=settings.API_V1_STR)
app.include_router(ingestion_router, prefix=settings.API_V1_STR)
app.include_router(segmentation_router, prefix=settings.API_V1_STR)
app.include_router(video_router, prefix=settings.API_V1_STR)
app.include_router(reality_router, prefix=settings.API_V1_STR)

@app.get("/")
def root_info():
    return {
        "platform": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "docs_url": "/docs",
        "health_url": "/health",
        "metrics_url": "/metrics",
        "api_v1": settings.API_V1_STR
    }

@app.get("/metrics", tags=["Observability"])
def prometheus_metrics():
    """Exposes real-time Prometheus metrics for telemetry and monitoring."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
