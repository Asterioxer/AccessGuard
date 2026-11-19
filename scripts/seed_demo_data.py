"""
Seed local development accounts without inserting fabricated streets, media, or detections.

Real-world pedestrian network data must come from an explicitly configured OSM import
or another provenance-bearing source. Real observations must come from uploaded or
authorized source imagery processed by the application.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.db.session import engine, SessionLocal, Base
from app.db.models import User
from app.core.security import get_password_hash


def seed():
    print("Creating database tables if not present...")
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    try:
        admin = db.query(User).filter(User.email == "admin@accessguard.org").first()
        if not admin:
            db.add(User(
                email="admin@accessguard.org",
                hashed_password=get_password_hash("AccessGuard2026!"),
                role="admin",
                reputation_score=100
            ))

        verifier = db.query(User).filter(User.email == "verifier@accessguard.org").first()
        if not verifier:
            db.add(User(
                email="verifier@accessguard.org",
                hashed_password=get_password_hash("Verifier2026!"),
                role="verifier",
                reputation_score=50
            ))

        db.commit()
        print("Development accounts are ready. No demo streets, images, observations, or detections were created.")
    finally:
        db.close()


if __name__ == "__main__":
    seed()
