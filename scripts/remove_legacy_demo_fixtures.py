"""
Safely remove only the exact legacy demo fixtures created by the old seed script.

Default behavior is dry-run. Pass --apply only after backing up the database.
No user-uploaded observations or arbitrary media files are targeted.
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.config import settings
from app.db.session import SessionLocal
from app.db.models import Observation, PedestrianSegment


DEMO_OBSERVATION_IDS = {
    "obs-stairs-01",
    "obs-ramp-02",
    "obs-damage-03",
}

DEMO_SEGMENT_IDS = {
    "seg_main_st_north",
    "seg_central_plaza_stairs",
    "seg_plaza_bypass_ramp_east",
    "seg_plaza_bypass_ramp_north",
    "seg_market_crosswalk",
    "seg_civic_walkway_west",
    "seg_transit_access_north",
    "seg_transit_to_market",
}

DEMO_MEDIA_NAMES = {
    "demo_stairs.jpg",
    "demo_ramp.jpg",
    "demo_damage.jpg",
}


def main():
    parser = argparse.ArgumentParser(description="Remove only known legacy AccessGuard demo fixtures.")
    parser.add_argument("--apply", action="store_true", help="Apply deletions. Without this flag the script is a dry-run.")
    args = parser.parse_args()

    db = SessionLocal()
    try:
        observations = db.query(Observation).filter(Observation.id.in_(DEMO_OBSERVATION_IDS)).all()
        segments = db.query(PedestrianSegment).filter(PedestrianSegment.id.in_(DEMO_SEGMENT_IDS)).all()
        media_paths = [settings.PROCESSED_DIR / name for name in DEMO_MEDIA_NAMES]
        existing_media = [path for path in media_paths if path.is_file()]

        print("Legacy demo fixture cleanup plan")
        print(f"  Known demo observations: {len(observations)}")
        print(f"  Known demo segments found: {len(segments)}")
        print(f"  Known demo media files found: {len(existing_media)}")
        for path in existing_media:
            print(f"  Media: {path}")

        if not args.apply:
            print("DRY RUN ONLY. Back up the database, then rerun with --apply to delete these exact fixtures.")
            return

        for observation in observations:
            db.delete(observation)
        db.flush()

        deleted_segments = 0
        for segment in segments:
            remaining = db.query(Observation).filter(Observation.segment_id == segment.id).count()
            if remaining == 0:
                db.delete(segment)
                deleted_segments += 1
            else:
                print(f"Preserving segment {segment.id}: {remaining} non-demo observation(s) still reference it.")

        db.commit()

        for path in existing_media:
            # Only fixed basenames under the configured processed-media directory are eligible.
            if path.parent.resolve() == settings.PROCESSED_DIR.resolve() and path.name in DEMO_MEDIA_NAMES:
                path.unlink(missing_ok=True)

        print(f"Removed {len(observations)} known demo observations, {deleted_segments} unreferenced demo segments, and {len(existing_media)} known demo media files.")
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    main()
