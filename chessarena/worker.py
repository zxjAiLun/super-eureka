"""Worker entry point (section 19).

The worker owns the scheduler.  It:
1. runs recovery once at boot,
2. loops: scheduler tick + worker heartbeat,
3. on SIGTERM/SIGINT stops scheduling, terminates the cutechess process group
   (15s grace then SIGKILL), and marks the current attempt INTERRUPTED.

Run as:  python -m chessarena.worker
"""

from __future__ import annotations

import logging
import signal
import sys
import threading
import time
from datetime import datetime, timezone

from .config import Settings, get_settings
from .db import make_engine, make_session_factory
from .models import WorkerState
from .services import artifacts, recovery
from .services.scheduler import Scheduler

logger = logging.getLogger("chessarena.worker")

_stop_event = threading.Event()


def _handle_signal(signum, frame):
    logger.info("received signal %s, shutting down", signum)
    _stop_event.set()


def _heartbeat(session_factory, scheduler: Scheduler) -> None:
    with session_factory() as session:
        state = session.get(WorkerState, 1)
        if state is None:
            state = WorkerState(id=1)
            session.add(state)
        now = datetime.now(timezone.utc)
        state.status = "running" if scheduler.active_proc else "idle"
        state.heartbeat_at = now
        state.pid = scheduler.active_proc.pid if scheduler.active_proc else None
        state.tournament_id = scheduler.active_tournament_id
        state.pair_job_id = scheduler.active_pair_job_id
        session.commit()


def run_worker(settings: Settings, session_factory) -> int:
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    logger.info("worker starting; running recovery")
    recovery.run_recovery(settings, session_factory)

    scheduler = Scheduler(settings, session_factory)
    logger.info("worker recovery complete; entering scheduler loop")

    last_beat = 0.0
    while not _stop_event.is_set():
        started = time.monotonic()
        try:
            action = scheduler.tick()
            if action != "idle":
                logger.info("tick: %s", action)
        except Exception:
            logger.exception("scheduler tick failed")
        if time.monotonic() - last_beat >= settings.worker_heartbeat_seconds:
            try:
                _heartbeat(session_factory, scheduler)
                last_beat = time.monotonic()
            except Exception:
                logger.exception("heartbeat failed")

        elapsed = time.monotonic() - started
        sleep_for = max(0.0, settings.worker_poll_seconds - elapsed)
        if not _stop_event.wait(sleep_for):
            continue
        break

    logger.info("worker stopping; terminating active process group if any")
    scheduler.shutdown()
    _mark_stopped(session_factory)
    logger.info("worker stopped cleanly")
    return 0


def _mark_stopped(session_factory) -> None:
    try:
        with session_factory() as session:
            state = session.get(WorkerState, 1)
            if state is None:
                state = WorkerState(id=1)
                session.add(state)
            state.status = "stopped"
            state.heartbeat_at = datetime.now(timezone.utc)
            session.commit()
    except Exception:
        logger.exception("failed to mark worker stopped")


def main() -> int:
    logging.basicConfig(
        level=logging.getLevelName(get_settings().log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    settings = get_settings()
    artifacts.configure_artifact_service(settings)
    engine = make_engine(settings.db_url)
    session_factory = make_session_factory(engine)
    return run_worker(settings, session_factory)


if __name__ == "__main__":
    sys.exit(main())
