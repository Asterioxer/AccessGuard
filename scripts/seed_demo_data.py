"""
Initialize the database schema and optional local development accounts.

No synthetic streets, images, observations, or detections are generated.
Set ADMIN_INITIAL_PASSWORD and/or VERIFIER_INITIAL_PASSWORD in the environment
before running this script if bootstrap accounts are required.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.db.session import engine, SessionLocal, Base
from app.db.models import User
from app.core.security import get_password_hash
from app.config import settings


def seed():
    print("Creating database tables if not present...")
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

        verifier = db.query(User).filter(User.email == "verifier@accessguard.org").first()
        if not verifier and settings.VERIFIER_INITIAL_PASSWORD:
            db.add(User(
                email="verifier@accessguard.org",
                hashed_password=get_password_hash(settings.VERIFIER_INITIAL_PASSWORD),
                role="verifier",
                reputation_score=50
            ))

        db.commit()
        print("Schema initialized. No synthetic streets, media, observations, or detections were created.")
        if not settings.ADMIN_INITIAL_PASSWORD:
            print("ADMIN_INITIAL_PASSWORD is unset; no default administrator was created.")
    finally:
        db.close()


if __name__ == "__main__":
    seed()
