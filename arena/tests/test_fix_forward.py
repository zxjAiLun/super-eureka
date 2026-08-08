"""Fix-forward acceptance tests (remote review P1/P2).

Covers the deployment-boundary issues that the Windows in-process fake tests
could not see:

- P1.2: worker restart mid-pair re-runs the interrupted pair; a tournament
  completes only under strict conditions (all pairs COMPLETED + exact game
  count), never on a leftover INTERRUPTED pair.
- P1.5: a non-zero cutechess exit code fails the pair and tournament even
  when the artifacts look complete.
- P1.6: the artifact manifest enumerates every pair artifact by
  tournament-relative path (no basename collisions).
- P2.1: cancel takes priority over pause when finalizing PAUSING.
- P2.2: cancel requested during the last pair wins over auto-completion.
- P2.4: admin forms need a CSRF token; state-changing API requests reject
  cross-site origins.
- P1.3: force-cancel works across real API + worker processes (the worker is
  spawned as a separate OS process and picks the flag up from the database).
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from chessarena.models import (
    CANCELLED,
    COMPLETED,
    FAILED,
    INTERRUPTED,
    PAUSED,
    PAUSING,
    PENDING,
    RUNNING,
    Game,
    PairJob,
    Tournament,
    WorkerState,
    utcnow,
)
from chessarena.services import recovery, artifacts
from chessarena.services.scheduler import Scheduler

ARENA_ROOT = Path(__file__).resolve().parents[1]


def _tick_until(engine_factory, tournament_id, predicate, scheduler, max_ticks=400):
    for _ in range(max_ticks):
        with engine_factory() as session:
            tournament = session.get(Tournament, tournament_id)
            if predicate(tournament):
                return tournament
        scheduler.tick()
        time.sleep(0.02)
    raise AssertionError("condition not reached within tick budget")


def _wait_db(engine_factory, check, timeout=15.0, interval=0.1):
    deadline = time.time() + timeout
    while time.time() < deadline:
        with engine_factory() as session:
            if check(session):
                return
        time.sleep(interval)
    raise AssertionError("database condition not reached in time")


# ---------------------------------------------------------------------------
# P1.2 restart scenarios
# ---------------------------------------------------------------------------
def _simulate_restart(scheduler, engine_factory, tournament_id, settings,
                      pairs_to_run):
    """Run scheduler_a until the N-th pair is RUNNING, then stop the worker.

    Returns a fresh Scheduler (simulating the restarted worker process).
    """
    # drive to the requested pair index
    for _ in range(200):
        with engine_factory() as session:
            running = (
                session.query(PairJob)
                .filter(
                    PairJob.tournament_id == tournament_id,
                    PairJob.status == RUNNING,
                )
                .first()
            )
            if running is not None and running.pair_index == pairs_to_run:
                break
        scheduler.tick()
        time.sleep(0.02)
    else:
        raise AssertionError("requested pair never became RUNNING")

    scheduler.shutdown()  # graceful worker stop mid-pair

    # worker boot: recovery + a brand-new scheduler
    recovery.run_recovery(settings, engine_factory)
    return Scheduler(settings, engine_factory)


@pytest.mark.parametrize("interrupt_at_pair, total_pairs", [
    (0, 1),    # single pair
    (0, 3),    # first pair
    (1, 3),    # middle pair
])
def test_restart_reruns_interrupted_pair(settings, engine_factory,
                                         tournament_factory, interrupt_at_pair,
                                         total_pairs):
    os.environ["FAKE_CUTECHESS_SLEEP_MS"] = "1200"
    try:
        tournament_id = tournament_factory(status="QUEUED", pairs=total_pairs)
        scheduler = Scheduler(settings, engine_factory)
        scheduler = _simulate_restart(
            scheduler, engine_factory, tournament_id, settings, interrupt_at_pair
        )

        with engine_factory() as session:
            interrupted = (
                session.query(PairJob)
                .filter(
                    PairJob.tournament_id == tournament_id,
                    PairJob.pair_index == interrupt_at_pair,
                )
                .first()
            )
            # The interrupted pair was reset to PENDING with a bumped attempt.
            assert interrupted.status == PENDING
            assert interrupted.attempt == 2

        # The restarted worker drives the whole tournament to completion.
        tournament = _tick_until(
            engine_factory, tournament_id,
            lambda t: t.status in (COMPLETED, FAILED),
            scheduler,
        )
        assert tournament.status == COMPLETED
        assert tournament.completed_pairs == total_pairs
        with engine_factory() as session:
            games = (
                session.query(Game)
                .filter(Game.tournament_id == tournament_id)
                .count()
            )
            assert games == total_pairs * 2
            pairs = (
                session.query(PairJob)
                .filter(PairJob.tournament_id == tournament_id)
                .all()
            )
            assert all(p.status == COMPLETED for p in pairs)
    finally:
        os.environ.pop("FAKE_CUTECHESS_SLEEP_MS", None)


def test_restart_last_pair_does_not_prematurely_complete(settings, engine_factory,
                                                         tournament_factory):
    """P1.2: a shutdown during the only pair must not yield COMPLETED."""
    os.environ["FAKE_CUTECHESS_SLEEP_MS"] = "1200"
    try:
        tournament_id = tournament_factory(status="QUEUED", pairs=1)
        scheduler = Scheduler(settings, engine_factory)
        scheduler = _simulate_restart(
            scheduler, engine_factory, tournament_id, settings, 0
        )
        # Immediately after recovery the tournament must NOT be COMPLETED.
        with engine_factory() as session:
            tournament = session.get(Tournament, tournament_id)
            assert tournament.status != COMPLETED
            assert tournament.completed_pairs == 0
        # It completes only after the retried pair actually runs.
        tournament = _tick_until(
            engine_factory, tournament_id,
            lambda t: t.status in (COMPLETED, FAILED),
            scheduler,
        )
        assert tournament.status == COMPLETED
        assert tournament.completed_pairs == 1
        with engine_factory() as session:
            assert session.query(Game).filter(
                Game.tournament_id == tournament_id
            ).count() == 2
    finally:
        os.environ.pop("FAKE_CUTECHESS_SLEEP_MS", None)


def test_double_recovery_does_not_double_score(settings, engine_factory,
                                               tournament_factory):
    os.environ["FAKE_CUTECHESS_SLEEP_MS"] = "1200"
    try:
        tournament_id = tournament_factory(status="QUEUED", pairs=1)
        scheduler = Scheduler(settings, engine_factory)
        _simulate_restart(scheduler, engine_factory, tournament_id, settings, 0)

        recovery.run_recovery(settings, engine_factory)
        recovery.run_recovery(settings, engine_factory)

        with engine_factory() as session:
            tournament = session.get(Tournament, tournament_id)
            assert tournament.completed_pairs == 0
            assert session.query(Game).filter(
                Game.tournament_id == tournament_id
            ).count() == 0
            pair = session.query(PairJob).filter(
                PairJob.tournament_id == tournament_id
            ).first()
            assert pair.attempt == 2
            assert pair.status == PENDING
    finally:
        os.environ.pop("FAKE_CUTECHESS_SLEEP_MS", None)


def test_interrupted_pending_mix_never_false_completed(settings, engine_factory,
                                                       tournament_factory):
    """An INTERRUPTED pair left in a QUEUED tournament must be re-run."""
    tournament_id = tournament_factory(status="QUEUED", pairs=2)
    with engine_factory() as session:
        pair = (
            session.query(PairJob)
            .filter(
                PairJob.tournament_id == tournament_id,
                PairJob.pair_index == 0,
            )
            .first()
        )
        pair.status = INTERRUPTED
        pair.attempt = 1
        session.commit()

    scheduler = Scheduler(settings, engine_factory)
    scheduler.tick()  # reschedules and immediately relaunches the pair
    with engine_factory() as session:
        pair = (
            session.query(PairJob)
            .filter(
                PairJob.tournament_id == tournament_id,
                PairJob.pair_index == 0,
            )
            .first()
        )
        assert pair.attempt == 2
        assert pair.status == RUNNING


# ---------------------------------------------------------------------------
# P1.5 nonzero exit code
# ---------------------------------------------------------------------------
def test_nonzero_exit_never_scored(settings, engine_factory, tournament_factory):
    """Complete artifacts + non-zero manager exit code = FAILED, no score."""
    os.environ["FAKE_CUTECHESS_EXIT_CODE"] = "1"
    try:
        tournament_id = tournament_factory(status="QUEUED", pairs=1)
        scheduler = Scheduler(settings, engine_factory)
        tournament = _tick_until(
            engine_factory, tournament_id,
            lambda t: t.status in (COMPLETED, FAILED),
            scheduler,
        )
        assert tournament.status == FAILED
        with engine_factory() as session:
            pair = session.query(PairJob).filter(
                PairJob.tournament_id == tournament_id
            ).first()
            assert pair.status == FAILED
            assert pair.return_code == 1
            assert pair.failure_reason and "code 1" in pair.failure_reason
            assert pair.verification["return_code"] == 1
            assert pair.verification["verified"] is True  # diagnostic only
            assert session.query(Game).filter(
                Game.tournament_id == tournament_id
            ).count() == 0
            assert session.get(Tournament, tournament_id).completed_pairs == 0
    finally:
        os.environ.pop("FAKE_CUTECHESS_EXIT_CODE", None)


# ---------------------------------------------------------------------------
# P1.6 manifest completeness
# ---------------------------------------------------------------------------
def test_artifact_manifest_relative_paths(settings, engine_factory,
                                          tournament_factory):
    tournament_id = tournament_factory(status="QUEUED", pairs=2)
    scheduler = Scheduler(settings, engine_factory)
    _tick_until(
        engine_factory, tournament_id,
        lambda t: t.status in (COMPLETED, FAILED),
        scheduler,
    )

    manifest_path = artifacts.tournament_run_dir(tournament_id) / "artifact-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest["files"]

    # No basename collisions: keys are tournament-relative paths.
    assert "match.pgn" not in files
    assert "combined.pgn" in files
    assert "summary.json" in files
    for pair in range(2):
        prefix = f"pairs/{pair:06d}/attempt-01/"
        for name in ("match.pgn", "stdout.log", "stderr.log", "command.json",
                     "command.txt", "opening.epd", "verification.json"):
            key = prefix + name
            assert key in files, f"missing {key}"
            assert len(files[key]["sha256"]) == 64


# ---------------------------------------------------------------------------
# P2.1 cancel priority over pause
# ---------------------------------------------------------------------------
def test_pausing_with_cancel_becomes_cancelled(scheduler, engine_factory,
                                               tournament_factory, settings):
    tournament_id = tournament_factory(status=PAUSING, pairs=2)
    with engine_factory() as session:
        tournament = session.get(Tournament, tournament_id)
        tournament.pause_requested = True
        tournament.cancel_requested = True
        session.commit()
    scheduler.tick()
    with engine_factory() as session:
        assert session.get(Tournament, tournament_id).status == CANCELLED


# ---------------------------------------------------------------------------
# P2.2 cancel during last pair
# ---------------------------------------------------------------------------
def test_cancel_during_last_pair_wins_over_completion(scheduler, engine_factory,
                                                      tournament_factory, settings):
    os.environ["FAKE_CUTECHESS_SLEEP_MS"] = "400"
    try:
        tournament_id = tournament_factory(status="QUEUED", pairs=2)
        # Let pair 0 finish and pair 1 start.
        for _ in range(200):
            with engine_factory() as session:
                running = (
                    session.query(PairJob)
                    .filter(
                        PairJob.tournament_id == tournament_id,
                        PairJob.status == RUNNING,
                    )
                    .first()
                )
                if running is not None and running.pair_index == 1:
                    break
            scheduler.tick()
            time.sleep(0.02)
        with engine_factory() as session:
            tournament = session.get(Tournament, tournament_id)
            tournament.cancel_requested = True
            session.commit()
        # Drive to a terminal state; it must be CANCELLED, not COMPLETED.
        tournament = _tick_until(
            engine_factory, tournament_id,
            lambda t: t.status in (COMPLETED, CANCELLED, FAILED),
            scheduler,
        )
        assert tournament.status == CANCELLED
        with engine_factory() as session:
            assert session.get(Tournament, tournament_id).completed_pairs == 2
    finally:
        os.environ.pop("FAKE_CUTECHESS_SLEEP_MS", None)


# ---------------------------------------------------------------------------
# P2.4 CSRF + same-origin
# ---------------------------------------------------------------------------
def test_admin_dashboard_renders_active_tournament(
    app_client, engine_factory, tournament_factory
):
    """P1 regression: /admin/ must render while a match is RUNNING.  The
    dashboard includes _tournament_status.html which needs tournament/pairs;
    before the fix it only passed 'active' and the page 500'd."""
    tid = tournament_factory(name="Active Smoke", pairs=2, status=RUNNING)
    with engine_factory() as session:
        t = session.query(Tournament).filter(Tournament.id == tid).one()
        t.started_at = utcnow()
        pair = t.pair_jobs[0]
        pair.status = RUNNING
        worker = session.get(WorkerState, 1)
        if worker is None:
            worker = WorkerState(id=1)
            session.add(worker)
        worker.tournament_id = tid
        worker.pair_job_id = pair.id
        worker.status = "running"
        worker.heartbeat_at = utcnow()
        session.commit()

    r = app_client.get("/chessarena/admin/")
    assert r.status_code == 200
    assert "Running match" in r.text
    assert "Active Smoke" in r.text
    assert "Current pair:" in r.text


def test_admin_dashboard_idle_has_no_poll_target(app_client):
    """Idle dashboard must not emit /admin/tournaments//status (double slash)
    as an HTMX polling target."""
    r = app_client.get("/chessarena/admin/")
    assert r.status_code == 200
    assert "/tournaments//status" not in r.text
    assert "No match currently running" in r.text
    assert "Running match" not in r.text


def test_admin_form_without_csrf_token_is_403(app_client):
    opening = app_client.get("/chessarena/api/v1/opening-sets").json()[0]
    payload = {
        "name": "csrf",
        "engine_a_preset": "chessengine-production",
        "engine_b_preset": "chessengine-legacy-current",
        "opening_set_id": opening["opening_set_id"],
        "time_control": "blitz_3_2",
        "pairs": 2,
    }
    response = app_client.post(
        "/chessarena/admin/tournaments", data=payload, follow_redirects=False
    )
    assert response.status_code == 403


def test_admin_form_with_csrf_token_works(app_client):
    # P2.4: the token is bound to this browser's session cookie.
    page = app_client.get("/chessarena/admin/tournaments/new")
    token = app_client.cookies.get("arena_csrf")
    assert token, "CSRF cookie was not set"
    assert token == page.text.split('name="_csrf_token" value="')[1].split('"')[0]

    opening = app_client.get("/chessarena/api/v1/opening-sets").json()[0]
    payload = {
        "name": "csrf-ok",
        "engine_a_preset": "chessengine-production",
        "engine_b_preset": "chessengine-legacy-current",
        "opening_set_id": opening["opening_set_id"],
        "time_control": "blitz_3_2",
        "pairs": 2,
        "_csrf_token": token,
    }
    response = app_client.post(
        "/chessarena/admin/tournaments", data=payload, follow_redirects=False
    )
    assert response.status_code == 303


def test_csrf_token_is_session_isolated(app_client):
    """A token from one browser must not be accepted with another's session."""
    from fastapi.testclient import TestClient

    other = TestClient(app_client.app)
    other.get("/chessarena/admin/tournaments/new")  # give it its own cookie
    other_token = other.cookies.get("arena_csrf")
    my_token = app_client.cookies.get("arena_csrf")
    if my_token is None:
        app_client.get("/chessarena/admin/tournaments/new")
        my_token = app_client.cookies.get("arena_csrf")
    assert other_token != my_token

    build = app_client.get("/chessarena/api/v1/builds").json()[0]
    opening = app_client.get("/chessarena/api/v1/opening-sets").json()[0]
    payload = {
        "name": "csrf-mix",
        "engine_a_build": build["build_id"],
        "engine_a_profile": "current-final",
        "engine_b_build": build["build_id"],
        "engine_b_profile": "current",
        "opening_set_id": opening["opening_set_id"],
        "time_control": "blitz_3_2",
        "pairs": 2,
        "_csrf_token": my_token,  # browser A's token
    }
    # Submitted with browser B's cookie -> rejected.
    response = other.post(
        "/chessarena/admin/tournaments", data=payload, follow_redirects=False
    )
    assert response.status_code == 403


def test_api_post_with_cross_origin_is_403(app_client):
    data = _create_payload_for(app_client)
    response = app_client.post(
        "/chessarena/api/v1/tournaments",
        json=data,
        headers={"Origin": "https://evil.example"},
    )
    assert response.status_code == 403


def _create_payload_for(app_client):
    build = app_client.get("/chessarena/api/v1/builds").json()[0]
    opening = app_client.get("/chessarena/api/v1/opening-sets").json()[0]
    return {
        "name": "origin",
        "engine_a": {"build_id": build["build_id"], "profile": "current-final"},
        "engine_b": {"build_id": build["build_id"], "profile": "current"},
        "opening_set_id": opening["opening_set_id"],
        "time_control": "blitz_3_2",
        "pairs": 2,
    }


# ---------------------------------------------------------------------------
# P1.3 cross-process force-cancel
# ---------------------------------------------------------------------------
def test_force_cancel_cleanup_failure_retains_state(settings, engine_factory,
                                                    tournament_factory,
                                                    scheduler, monkeypatch):
    """P1: a failed cleanup retains active state, identity and the force-cancel
    flag, the next tick RETRIES the kill, and only a confirmed-clean group is
    allowed to cancel the tournament / clear the active state."""
    from chessarena.services import cutechess as cc
    from chessarena.models import WorkerState

    os.environ["FAKE_CUTECHESS_SLEEP_MS"] = "5000"
    original_terminate = cc.terminate_process_group
    try:
        tournament_id = tournament_factory(status="QUEUED", pairs=2)
        scheduler.tick()  # pair 0 running; worker_state identity recorded
        with engine_factory() as session:
            tournament = session.get(Tournament, tournament_id)
            tournament.force_cancel_requested = True
            session.commit()
            pair_id = session.query(PairJob).filter(
                PairJob.tournament_id == tournament_id,
                PairJob.pair_index == 0,
            ).first().id

        results = iter([False, False, True])  # fail, retry-fail, succeed

        def fake_terminate(proc, grace):
            return next(results)

        monkeypatch.setattr(
            "chessarena.services.cutechess.terminate_process_group", fake_terminate
        )

        # Tick 1: cleanup fails -> everything retained.
        assert "pending" in scheduler.tick()
        assert scheduler.active_proc is not None
        assert scheduler.active_pair_job_id == pair_id
        with engine_factory() as session:
            tournament = session.get(Tournament, tournament_id)
            assert tournament.force_cancel_requested is True
            assert tournament.status == "RUNNING"
            pair1 = session.query(PairJob).filter(
                PairJob.tournament_id == tournament_id,
                PairJob.pair_index == 1,
            ).first()
            assert pair1.status == "PENDING"
            state = session.get(WorkerState, 1)
            assert state is not None
            assert state.pid == scheduler.active_proc.pid
            assert state.pair_job_id == pair_id
            assert state.tournament_id == tournament_id
            # Identity evidence is retained (non-None on Linux, where it exists).
            if sys.platform.startswith("linux"):
                assert state.pid_start_marker is not None
                assert state.pid_cmdline is not None

        # Tick 2: cleanup fails AGAIN -> the retry really happened, state kept.
        assert "pending" in scheduler.tick()
        assert scheduler.active_proc is not None
        with engine_factory() as session:
            tournament = session.get(Tournament, tournament_id)
            assert tournament.force_cancel_requested is True
            assert tournament.status == "RUNNING"
            state = session.get(WorkerState, 1)
            assert state.pid == scheduler.active_proc.pid
            assert state.pair_job_id == pair_id

        # Tick 3: cleanup succeeds -> only now cancel/interrupt/clear.
        assert "force-cancelled" in scheduler.tick()
        assert scheduler.active_proc is None
        with engine_factory() as session:
            tournament = session.get(Tournament, tournament_id)
            assert tournament.status == "CANCELLED"
            assert tournament.force_cancel_requested is False
            pair0 = session.query(PairJob).filter(
                PairJob.tournament_id == tournament_id,
                PairJob.pair_index == 0,
            ).first()
            assert pair0.status == "INTERRUPTED"
            pair1 = session.query(PairJob).filter(
                PairJob.tournament_id == tournament_id,
                PairJob.pair_index == 1,
            ).first()
            assert pair1.status == "PENDING"  # still never started
    finally:
        os.environ.pop("FAKE_CUTECHESS_SLEEP_MS", None)
        monkeypatch.setattr(
            "chessarena.services.cutechess.terminate_process_group",
            original_terminate,
        )
        if scheduler.active_proc is not None:
            scheduler.shutdown()


def test_force_cancel_across_real_processes(settings, engine_factory,
                                            tournament_factory, app_client):
    """The worker runs as a separate OS process; force-cancel still lands."""
    os.environ["FAKE_CUTECHESS_SLEEP_MS"] = "30000"
    os.environ["ARENA_WORKER_POLL_SECONDS"] = "0.2"
    os.environ["ARENA_WORKER_HEARTBEAT_SECONDS"] = "0.2"
    proc: subprocess.Popen | None = None
    try:
        tournament_id = tournament_factory(status="QUEUED", pairs=1)

        # The worker refuses to start while any enabled build lacks a
        # capability schema (deployment gate); backfill it first.
        from chessarena.models import EngineBuild

        with engine_factory() as session:
            for build in session.query(EngineBuild):
                build.uci_options_schema = {
                    "Hash": {"type": "spin", "min": 1, "max": 1024}
                }
            session.commit()

        # Real worker process (API and worker share only the database).
        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        proc = subprocess.Popen(
            [sys.executable, "-B", "-m", "chessarena.worker"],
            cwd=str(ARENA_ROOT),
            env=env,
        )

        # Wait until the pair is running.
        _wait_db(
            engine_factory,
            lambda session: (
                session.query(PairJob)
                .filter(
                    PairJob.tournament_id == tournament_id,
                    PairJob.status == RUNNING,
                )
                .count()
                > 0
            ),
            timeout=30,
        )

        # Force-cancel through the API (parent process -> database flag).
        response = app_client.post(
            f"/chessarena/api/v1/tournaments/{tournament_id}/force-cancel",
            params={"confirm": "true"},
        )
        assert response.status_code == 200
        assert response.json()["force_cancel_requested"] is True

        # The worker applies it: tournament CANCELLED within a few seconds.
        _wait_db(
            engine_factory,
            lambda session: (
                session.get(Tournament, tournament_id).status == CANCELLED
            ),
            timeout=15,
        )
        with engine_factory() as session:
            tournament = session.get(Tournament, tournament_id)
            assert tournament.force_cancel_requested is False
            assert tournament.completed_pairs == 0
            pair = session.query(PairJob).filter(
                PairJob.tournament_id == tournament_id
            ).first()
            assert pair.status == INTERRUPTED
            assert pair.failure_reason == "force-cancelled"
    finally:
        os.environ.pop("FAKE_CUTECHESS_SLEEP_MS", None)
        os.environ.pop("ARENA_WORKER_POLL_SECONDS", None)
        os.environ.pop("ARENA_WORKER_HEARTBEAT_SECONDS", None)
        if proc is not None:
            try:
                proc.send_signal(signal.SIGTERM)
                proc.wait(timeout=15)
            except (OSError, subprocess.TimeoutExpired):
                proc.kill()
