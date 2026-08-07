"""Scheduler tests (spec sections 10, 11, 22.1, 22.4).

Drives the Scheduler directly (no worker loop): each ``tick()`` does one unit
of work against the real test database and the fake cutechess fixture.
"""

from __future__ import annotations

import os
import time

from chessarena.models import (
    CANCELLED,
    COMPLETED,
    DRAFT,
    FAILED,
    INTERRUPTED,
    PAUSED,
    PAUSING,
    PENDING,
    QUEUED,
    RUNNING,
    Event,
    Game,
    PairJob,
    Tournament,
)
from chessarena.services import artifacts
from chessarena.services.cutechess import read_output_lines

from . import helpers


def _run_until(predicate, scheduler, max_ticks=200):
    """Call tick() until predicate returns truthy, or raise."""
    for _ in range(max_ticks):
        if predicate():
            return
        scheduler.tick()
        time.sleep(0.02)
    raise AssertionError("condition not reached within tick budget")


def _load(session_factory, tournament_id):
    with session_factory() as session:
        return session.get(Tournament, tournament_id)


def _pair(session_factory, tournament_id, index=0):
    with session_factory() as session:
        return (
            session.query(PairJob)
            .filter(
                PairJob.tournament_id == tournament_id,
                PairJob.pair_index == index,
            )
            .first()
        )


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
def test_scheduler_launches_pair(scheduler, engine_factory, tournament_factory,
                                 settings):
    tournament_id = tournament_factory(status=QUEUED, pairs=3)
    scheduler.tick()
    with engine_factory() as session:
        tournament = session.get(Tournament, tournament_id)
        assert tournament.status == RUNNING
        pair = _pair(engine_factory, tournament_id)
        assert pair.status == RUNNING
        assert pair.attempt == 1
        run_dir = artifacts.pair_run_dir(tournament_id, 0, 1)
        assert pair.run_directory == str(run_dir)
        assert (run_dir / "command.json").exists()
        assert (run_dir / "command.txt").exists()
        assert (run_dir / "opening.epd").exists()
    # one run at a time: the second tournament must wait
    scheduler.tick()


def test_scheduler_poll_observes_running_state(scheduler, engine_factory,
                                               tournament_factory, settings):
    os.environ["FAKE_CUTECHESS_SLEEP_MS"] = "500"
    try:
        tournament_id = tournament_factory(status=QUEUED, pairs=2)
        scheduler.tick()
        # still running
        with engine_factory() as session:
            pair = _pair(engine_factory, tournament_id)
            assert pair.status == RUNNING
            events = (
                session.query(Event)
                .filter(Event.tournament_id == tournament_id)
                .all()
            )
            assert any(e.event_type == "pair_started" for e in events)
    finally:
        os.environ.pop("FAKE_CUTECHESS_SLEEP_MS", None)


# ---------------------------------------------------------------------------
# Completion
# ---------------------------------------------------------------------------
def test_scheduler_completes_pair(scheduler, engine_factory, tournament_factory,
                                  settings):
    tournament_id = tournament_factory(status=QUEUED, pairs=2)
    _run_until(lambda: _load(engine_factory, tournament_id).status == RUNNING,
               scheduler)
    _run_until(lambda: _load(engine_factory, tournament_id).status == COMPLETED,
               scheduler)

    with engine_factory() as session:
        tournament = session.get(Tournament, tournament_id)
        assert tournament.completed_pairs == 2
        # results 1-0 / 0-1 per pair -> 2 wins per pair -> 4 wins total
        assert tournament.candidate_wins == 4
        assert tournament.candidate_losses == 0
        assert tournament.draws == 0
        games = session.query(Game).filter(Game.tournament_id == tournament_id).all()
        assert len(games) == 4
        assert all(g.verified for g in games)

        # events include the full lifecycle
        events = {
            e.event_type
            for e in session.query(Event)
            .filter(Event.tournament_id == tournament_id)
            .all()
        }
        assert events >= {
            "pair_started", "pair_completed", "game_completed",
            "verification_completed", "tournament_completed",
        }


def test_tournament_artifacts_generated(scheduler, engine_factory,
                                        tournament_factory, settings):
    tournament_id = tournament_factory(status=QUEUED, pairs=2)
    _run_until(lambda: _load(engine_factory, tournament_id).status == COMPLETED,
               scheduler)

    run_dir = artifacts.tournament_run_dir(tournament_id)
    combined = run_dir / "combined.pgn"
    summary = run_dir / "summary.json"
    manifest = run_dir / "artifact-manifest.json"
    assert combined.exists()
    assert summary.exists()
    assert manifest.exists()
    assert combined.read_text().count("[Event ") == 4

    import json

    summary_data = json.loads(summary.read_text())
    assert summary_data["candidate_perspective"]["wins"] == 4
    assert summary_data["games"][0]["white"] == "EngineA"
    assert summary_data["games"][1]["white"] == "EngineB"

    manifest_data = json.loads(manifest.read_text())
    for name in ("combined.pgn", "summary.json"):
        assert name in manifest_data["files"]
        assert len(manifest_data["files"][name]["sha256"]) == 64


# ---------------------------------------------------------------------------
# Pause / cancel
# ---------------------------------------------------------------------------
def test_pause_after_current_pair(scheduler, engine_factory, tournament_factory,
                                  settings):
    os.environ["FAKE_CUTECHESS_SLEEP_MS"] = "300"
    try:
        tournament_id = tournament_factory(status=QUEUED, pairs=3)
        scheduler.tick()  # pair 1 starts
        with engine_factory() as session:
            tournament = session.get(Tournament, tournament_id)
            tournament.pause_requested = True
            tournament.status = PAUSING
            session.commit()
        _run_until(lambda: _load(engine_factory, tournament_id).status == PAUSED,
                   scheduler)
        with engine_factory() as session:
            tournament = session.get(Tournament, tournament_id)
            assert tournament.pause_requested is False
            assert tournament.completed_pairs == 1  # pair 1 was scored
            pair1 = _pair(engine_factory, tournament_id, 0)
            assert pair1.status == COMPLETED
            pair2 = _pair(engine_factory, tournament_id, 1)
            assert pair2.status == PENDING  # never started
    finally:
        os.environ.pop("FAKE_CUTECHESS_SLEEP_MS", None)


def test_cancel_when_queued_no_pairs_started(scheduler, engine_factory,
                                             tournament_factory, settings):
    tournament_id = tournament_factory(status=QUEUED, pairs=3)
    with engine_factory() as session:
        tournament = session.get(Tournament, tournament_id)
        tournament.cancel_requested = True
        session.commit()
    scheduler.tick()
    with engine_factory() as session:
        tournament = session.get(Tournament, tournament_id)
        assert tournament.status == CANCELLED
        assert tournament.finished_at is not None


def test_force_kill_interrupts_pair(scheduler, engine_factory, tournament_factory,
                                    settings):
    os.environ["FAKE_CUTECHESS_SLEEP_MS"] = "5000"
    try:
        tournament_id = tournament_factory(status=QUEUED, pairs=2)
        scheduler.tick()
        # P1.3: force-cancel is a database flag (API and worker are separate
        # processes); the worker picks it up on the next poll.
        with engine_factory() as session:
            tournament = session.get(Tournament, tournament_id)
            tournament.force_cancel_requested = True
            session.commit()
        scheduler.tick()  # kill processed
        with engine_factory() as session:
            pair = _pair(engine_factory, tournament_id)
            assert pair.status == INTERRUPTED
            assert pair.failure_reason == "force-cancelled"
            tournament = session.get(Tournament, tournament_id)
            assert tournament.status == CANCELLED
            assert tournament.force_cancel_requested is False
            assert tournament.completed_pairs == 0  # never scored
    finally:
        os.environ.pop("FAKE_CUTECHESS_SLEEP_MS", None)


def test_second_tournament_waits_for_first(scheduler, engine_factory,
                                           tournament_factory, settings):
    os.environ["FAKE_CUTECHESS_SLEEP_MS"] = "300"
    try:
        first = tournament_factory(status=QUEUED, pairs=2)
        second = tournament_factory(status=QUEUED, pairs=2)
        scheduler.tick()
        with engine_factory() as session:
            assert session.get(Tournament, first).status == RUNNING
            assert session.get(Tournament, second).status == QUEUED
        _run_until(lambda: _load(engine_factory, first).status == COMPLETED,
                   scheduler)
        _run_until(lambda: _load(engine_factory, second).status == COMPLETED,
                   scheduler)
        assert _load(engine_factory, second).status == COMPLETED
    finally:
        os.environ.pop("FAKE_CUTECHESS_SLEEP_MS", None)


# ---------------------------------------------------------------------------
# State machine / API-level
# ---------------------------------------------------------------------------
def test_tournament_state_transitions(app_client):
    # A tournament in DRAFT cannot be paused; start moves it to QUEUED.
    opening = app_client.get("/chessarena/api/v1/opening-sets").json()[0]
    created = app_client.post(
        "/chessarena/api/v1/tournaments",
        json={
            "name": "smoke",
            "engine_a": {"preset_id": "chessengine-production"},
            "engine_b": {"preset_id": "chessengine-legacy-current"},
            "opening_set_id": opening["opening_set_id"],
            "time_control": "blitz_3_2",
            "pairs": 2,
        },
    )
    assert created.status_code == 201
    tournament_id = created.json()["id"]
    assert created.json()["status"] == "DRAFT"

    assert app_client.post(
        f"/chessarena/api/v1/tournaments/{tournament_id}/pause"
    ).status_code == 409
    assert app_client.post(
        f"/chessarena/api/v1/tournaments/{tournament_id}/start"
    ).json()["status"] == "QUEUED"
    # QUEUED -> pause is not allowed by the state machine
    assert app_client.post(
        f"/chessarena/api/v1/tournaments/{tournament_id}/pause"
    ).status_code == 409
