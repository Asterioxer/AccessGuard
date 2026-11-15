import json
import cv2
import numpy as np
from pathlib import Path
from sqlalchemy.orm import Session
from datetime import datetime, timezone

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.db.session import engine, SessionLocal, Base
from app.db.models import User, PedestrianSegment, Observation, BarrierDetection, VerificationReview
from app.core.security import get_password_hash
from app.gis.osm_importer import create_pedestrian_segment_models
from app.gis.scoring import calculate_segment_accessibility_score
from app.config import settings

def seed():
    print("Creating tables if not present...")
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    try:
        # 1. Seed Users
        admin = db.query(User).filter(User.email == "admin@accessguard.org").first()
        if not admin:
            admin = User(
                email="admin@accessguard.org",
                hashed_password=get_password_hash("AccessGuard2026!"),
                role="admin",
                reputation_score=100
            )
            db.add(admin)

        verifier = db.query(User).filter(User.email == "verifier@accessguard.org").first()
        if not verifier:
            verifier = User(
                email="verifier@accessguard.org",
                hashed_password=get_password_hash("Verifier2026!"),
                role="verifier",
                reputation_score=50
            )
            db.add(verifier)

        db.commit()

        # 2. Seed Segments
        if db.query(PedestrianSegment).count() == 0:
            print("Seeding pedestrian network segments...")
            segs = create_pedestrian_segment_models()
            for s in segs:
                db.add(s)
            db.commit()

        # 3. Create Sample Media Images
        settings.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

        # Image 1: Stairs at Central Plaza
        w, h = 640, 480
        img_stairs = np.full((h, w, 3), 175, dtype=np.uint8)
        for y in [240, 280, 320, 360, 400]:
            cv2.line(img_stairs, (80, y), (560, y), (30, 30, 30), 4)
        for y in [260, 300, 340, 380]:
            cv2.rectangle(img_stairs, (80, y-10), (560, y+10), (135, 135, 135), -1)
        path_stairs = settings.PROCESSED_DIR / "demo_stairs.jpg"
        cv2.imwrite(str(path_stairs), img_stairs)

        # Image 2: Accessible Ramp at Plaza Bypass
        img_ramp = np.full((h, w, 3), 180, dtype=np.uint8)
        cv2.rectangle(img_ramp, (0, 360), (w, h), (85, 85, 85), -1)
        cv2.line(img_ramp, (120, 360), (240, 300), (45, 45, 45), 3)
        cv2.line(img_ramp, (520, 360), (400, 300), (45, 45, 45), 3)
        path_ramp = settings.PROCESSED_DIR / "demo_ramp.jpg"
        cv2.imwrite(str(path_ramp), img_ramp)

        # Image 3: Broken Pavement Surface Damage
        img_dmg = np.full((h, w, 3), 155, dtype=np.uint8)
        np.random.seed(42)
        for _ in range(45):
            pt1 = (np.random.randint(120, 520), np.random.randint(280, 450))
            pt2 = (pt1[0] + np.random.randint(-30, 30), pt1[1] + np.random.randint(-30, 30))
            cv2.line(img_dmg, pt1, pt2, (15, 15, 15), 2)
        path_dmg = settings.PROCESSED_DIR / "demo_damage.jpg"
        cv2.imwrite(str(path_dmg), img_dmg)

        # 4. Attach Observations
        if db.query(Observation).count() == 0:
            print("Creating demo observations and detections...")

            # Obs 1: Stairs on seg_central_plaza_stairs
            obs1 = Observation(
                id="obs-stairs-01",
                segment_id="seg_central_plaza_stairs",
                media_path=str(path_stairs),
                media_type="image",
                latitude=37.7771,
                longitude=-122.4194,
                status="verified",
                verification_score=3,
                captured_at=datetime.now(timezone.utc)
            )
            db.add(obs1)
            det1 = BarrierDetection(
                id="det-stairs-01",
                observation_id=obs1.id,
                barrier_type="stairs",
                confidence=0.94,
                uncertainty_score=0.10,
                is_confirmed=True,
                bbox_ymin=0.45,
                bbox_xmin=0.12,
                bbox_ymax=0.88,
                bbox_xmax=0.88,
                polygon_json=json.dumps([[0.12, 0.45], [0.88, 0.45], [0.88, 0.88], [0.12, 0.88]]),
                evidence_json=json.dumps({
                    "feature_type": "periodic_horizontal_edges",
                    "measured_metric": 5.0,
                    "threshold_applied": 3.0,
                    "explanation": "5 concrete step rungs detected. Impassable for manual wheelchair without lift or ramp."
                })
            )
            db.add(det1)

            # Obs 2: Ramp present on seg_plaza_bypass_ramp_north
            obs2 = Observation(
                id="obs-ramp-02",
                segment_id="seg_plaza_bypass_ramp_north",
                media_path=str(path_ramp),
                media_type="image",
                latitude=37.7775,
                longitude=-122.4186,
                status="verified",
                verification_score=2,
                captured_at=datetime.now(timezone.utc)
            )
            db.add(obs2)
            det2 = BarrierDetection(
                id="det-ramp-02",
                observation_id=obs2.id,
                barrier_type="curb_ramp_present",
                confidence=0.88,
                uncertainty_score=0.12,
                is_confirmed=True,
                bbox_ymin=0.62,
                bbox_xmin=0.18,
                bbox_ymax=0.95,
                bbox_xmax=0.82,
                polygon_json=json.dumps([[0.18, 0.62], [0.82, 0.62], [0.82, 0.95], [0.18, 0.95]]),
                evidence_json=json.dumps({
                    "feature_type": "sloped_curb_transition",
                    "measured_metric": 1.0,
                    "threshold_applied": 0.5,
                    "explanation": "Standard ADA slope transition detected with flared side slopes. Accessible street crossing."
                })
            )
            db.add(det2)

            # Obs 3: Surface damage on seg_civic_walkway_west (Tentative - In Verification Queue)
            obs3 = Observation(
                id="obs-damage-03",
                segment_id="seg_civic_walkway_west",
                media_path=str(path_dmg),
                media_type="image",
                latitude=37.7749,
                longitude=-122.4205,
                status="tentative",
                verification_score=0,
                captured_at=datetime.now(timezone.utc)
            )
            db.add(obs3)
            det3 = BarrierDetection(
                id="det-damage-03",
                observation_id=obs3.id,
                barrier_type="surface_damage",
                confidence=0.76,
                uncertainty_score=0.28,
                is_confirmed=False,
                bbox_ymin=0.55,
                bbox_xmin=0.15,
                bbox_ymax=0.95,
                bbox_xmax=0.85,
                polygon_json=json.dumps([[0.15, 0.55], [0.85, 0.55], [0.85, 0.95], [0.15, 0.95]]),
                evidence_json=json.dumps({
                    "feature_type": "surface_roughness_and_crack_density",
                    "measured_metric": 0.082,
                    "threshold_applied": 0.07,
                    "explanation": "High-frequency crack pattern on pedestrian slab. Potential tipping risk for small wheelchair casters."
                })
            )
            db.add(det3)

            db.commit()

            # 5. Recalculate segment scores
            for seg in db.query(PedestrianSegment).all():
                seg_obs = db.query(Observation).filter(Observation.segment_id == seg.id).all()
                score, grade, breakdown = calculate_segment_accessibility_score(seg_obs)
                seg.accessibility_score = score
                seg.score_grade = grade
                seg.score_criteria_breakdown = json.dumps(breakdown.model_dump())
                seg.barrier_count = sum(len(o.detections) for o in seg_obs)
                db.commit()

        print("Successfully seeded demo data!")

    finally:
        db.close()

if __name__ == "__main__":
    seed()
