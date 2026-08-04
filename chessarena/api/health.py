"""Health endpoint (section 16.1).

Reports database, worker heartbeat and cutechess availability.  The worker
heartbeat comes from the single-row ``worker_state`` table; if the row is
missing or stale the worker is considered offline.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from ..config import Settings, get_settings
from ..db import get_db
from ..models import WorkerState, coerce_utc
from ..schemas import HealthOut

router = APIRouter(tags=["health"])


def _worker_status(session: Session, settings: Settings) -> str:
    row = session.query(WorkerState).filter(WorkerState.id == 1).first()
    if row is None:
        return "offline"
    heartbeat = coerce_utc(row.heartbeat_at)
    age = (datetime.now(timezone.utc) - heartbeat).total_seconds()
    if age > settings.worker_stale_seconds:
        return f"stale ({int(age)}s)"
    return "ok"


def _cutechess_status(settings: Settings) -> str:
    return "ok" if settings.cutechess.exists() else "missing"


@router.get("/health", response_model=HealthOut)
def health(
    session: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    db_ok = True
    try:
        session.execute(text("SELECT 1"))
    except Exception:
        db_ok = False

    worker = _worker_status(session, settings)
    cutechess = _cutechess_status(settings)

    active = (
        session.query(WorkerState.tournament_id)
        .filter(WorkerState.id == 1, WorkerState.tournament_id.isnot(None))
        .scalar()
    )

    status = "ok"
    if not db_ok or worker != "ok" or cutechess != "ok":
        status = "degraded"

    return HealthOut(
        status=status,
        database="ok" if db_ok else "error",
        worker_heartbeat=worker,
        cutechess=cutechess,
        active_tournament_id=active,
    )
