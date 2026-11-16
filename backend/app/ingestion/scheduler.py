"""
Asynchronous Ingestion Scheduler and Job Dispatcher.

Periodically queries pending/scheduled ingestion jobs, checks priority,
and dispatches execution to the AcquisitionEngine.
"""

import asyncio
from datetime import datetime, timezone
from typing import Optional
from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.db.models import IngestionJob
from app.ingestion.engine import AcquisitionEngine
from app.core.logging import logger


class IngestionScheduler:
    """
    Background scheduler for polling external imagery sources.
    """

    def __init__(self, engine: Optional[AcquisitionEngine] = None):
        self.engine = engine or AcquisitionEngine()
        self._is_running = False

    def trigger_pending_jobs(self, db: Session, max_jobs: int = 5) -> int:
        """
        Executes pending or due scheduled jobs.
        Priority order: high -> normal -> low.
        """
        now = datetime.now(timezone.utc)
        jobs = db.query(IngestionJob).filter(
            (IngestionJob.status == "pending") |
            ((IngestionJob.status == "completed") & (IngestionJob.next_run_at <= now))
        ).all()

        if not jobs:
            return 0

        # Sort by priority
        prio_order = {"high": 1, "normal": 2, "low": 3}
        sorted_jobs = sorted(jobs, key=lambda j: prio_order.get(j.priority, 2))

        executed = 0
        for job in sorted_jobs[:max_jobs]:
            try:
                logger.info(f"Scheduler dispatching job {job.id} for provider {job.provider} (priority: {job.priority})")
                self.engine.execute_job(job.id, db)
                executed += 1
            except Exception as e:
                logger.error(f"Scheduler error on job {job.id}: {e}")

        return executed

    async def run_loop(self, check_interval_sec: int = 60):
        """Asynchronous execution loop for background runners."""
        self._is_running = True
        logger.info(f"Ingestion scheduler loop started (check interval: {check_interval_sec}s).")
        while self._is_running:
            db = SessionLocal()
            try:
                self.trigger_pending_jobs(db)
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
            finally:
                db.close()
            await asyncio.sleep(check_interval_sec)

    def stop(self):
        self._is_running = False
        logger.info("Ingestion scheduler loop stopped.")
