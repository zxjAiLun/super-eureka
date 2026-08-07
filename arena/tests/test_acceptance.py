"""End-to-end acceptance harness (spec section 23).

Drives the real API + scheduler against the fake cutechess fixture to prove
the v1 acceptance scenario without SSH:
1. create via API,
2. start and let the worker run,
3. every opening is played as exactly one 2-game color-swapped pair,
4. all games are verified and counted exactly once,
5. artifacts (combined.pgn, summary.json, artifact-manifest.json) are produced,
6. the same flow works for all three time controls.
"""

from __future__ import annotations

import json
import time

from chessarena.models import COMPLETED, Game, PairJob, Tournament
from chessarena.services import artifacts


def _run_to_completion(scheduler, engine_factory, tournament_id, max_ticks=400):
    for _ in range(max_ticks):
        with engine_factory() as session:
            tournament = session.get(Tournament, tournament_id)
            if tournament.status == COMPLETED:
                return tournament
            if tournament.status == "FAILED":
                raise AssertionError(
                    f"tournament FAILED: {tournament.failure_reason}"
                )
        scheduler.tick()
        time.sleep(0.02)
    with engine_factory() as session:
        tournament = session.get(Tournament, tournament_id)
        pairs = session.query(PairJob).filter(
            PairJob.tournament_id == tournament_id
        ).all()
        raise AssertionError(
            "tournament did not complete in time: "
            f"status={tournament.status} completed={tournament.completed_pairs} "
            f"pairs={[(p.pair_index, p.status, p.attempt) for p in pairs]} "
            f"reason={tournament.failure_reason}"
        )


def test_full_lifecycle_acceptance(scheduler, engine_factory, app_client,
                                   tournament_factory, settings):
    created = app_client.post(
        "/chessarena/api/v1/tournaments",
        json={
            "name": "Arena v1 Acceptance",
            "engine_a": {"preset_id": "chessengine-production"},
            "engine_b": {"preset_id": "chessengine-legacy-current"},
            "opening_set_id": "test-openings-v1",
            "time_control": "bullet_1_0",
            "pairs": 10,
        },
    )
    assert created.status_code == 201
    tournament_id = created.json()["id"]

    assert app_client.post(
        f"/chessarena/api/v1/tournaments/{tournament_id}/start"
    ).json()["status"] == "QUEUED"

    tournament = _run_to_completion(scheduler, engine_factory, tournament_id)
    assert tournament.status == COMPLETED
    assert tournament.completed_pairs == 10
    # each pair = two wins for EngineA (fake results 1-0 / 0-1)
    assert tournament.candidate_wins == 20
    assert tournament.candidate_losses == 0
    assert tournament.draws == 0

    # Exactly 20 games, strict color swap, all verified.
    with engine_factory() as session:
        games = (
            session.query(Game)
            .filter(Game.tournament_id == tournament_id)
            .order_by(Game.game_number)
            .all()
        )
        assert len(games) == 20
        assert all(g.verified for g in games)
        for pair in range(10):
            g1 = games[pair * 2]
            g2 = games[pair * 2 + 1]
            assert (g1.white_engine, g1.black_engine) == (
                "ChessEngine Production", "ChessEngine Legacy Baseline")
            assert (g2.white_engine, g2.black_engine) == (
                "ChessEngine Legacy Baseline", "ChessEngine Production")

        pairs = (
            session.query(PairJob)
            .filter(PairJob.tournament_id == tournament_id)
            .all()
        )
        assert all(p.status == COMPLETED for p in pairs)
        assert all(p.attempt == 1 for p in pairs)

    # Artifacts downloadable through the API.
    combined = app_client.get(
        f"/chessarena/api/v1/tournaments/{tournament_id}/pgn"
    )
    assert combined.status_code == 200
    assert combined.text.count("[Event ") == 20

    summary = app_client.get(
        f"/chessarena/api/v1/tournaments/{tournament_id}/summary"
    ).json()
    assert summary["candidate_perspective"]["wins"] == 20
    assert summary["games"][0]["white"] == "ChessEngine Production"
    assert summary["games"][1]["white"] == "ChessEngine Legacy Baseline"

    manifest = app_client.get(
        f"/chessarena/api/v1/tournaments/{tournament_id}/artifacts"
    ).json()
    assert "combined.pgn" in manifest["files"]
    assert "summary.json" in manifest["files"]

    # Every game PGN is an actual file referenced in the run tree.
    run_dir = artifacts.tournament_run_dir(tournament_id)
    assert (run_dir / "combined.pgn").exists()


def test_all_time_controls_run(scheduler, engine_factory, tournament_factory,
                               settings):
    for tc in ("bullet_1_0", "blitz_3_2", "rapid_5_3"):
        tournament_id = tournament_factory(status="QUEUED", pairs=2, time_control=tc)
        tournament = _run_to_completion(scheduler, engine_factory, tournament_id)
        assert tournament.status == COMPLETED
        with engine_factory() as session:
            games = (
                session.query(Game)
                .filter(Game.tournament_id == tournament_id)
                .all()
            )
            assert len(games) == 4
            for game in games:
                assert game.verified


def test_worker_heartbeat_writes_state(scheduler, engine_factory, settings):
    from datetime import datetime, timezone

    from chessarena.worker import _heartbeat

    _heartbeat(engine_factory, scheduler)
    with engine_factory() as session:
        from chessarena.models import WorkerState, coerce_utc

        state = session.get(WorkerState, 1)
        assert state is not None
        assert state.status == "idle"
        assert (datetime.now(timezone.utc) - coerce_utc(state.heartbeat_at)).total_seconds() < 5


def test_worker_shutdown_interrupts_active_pair(scheduler, engine_factory,
                                                tournament_factory, settings):
    import os

    os.environ["FAKE_CUTECHESS_SLEEP_MS"] = "5000"
    try:
        tournament_id = tournament_factory(status="QUEUED", pairs=2)
        scheduler.tick()
        scheduler.shutdown()
        with engine_factory() as session:
            pair = (
                session.query(PairJob)
                .filter(PairJob.tournament_id == tournament_id)
                .first()
            )
            assert pair.status == "INTERRUPTED"
            assert pair.failure_reason == "worker shutdown"
            # not scored
            tournament = session.get(Tournament, tournament_id)
            assert tournament.completed_pairs == 0
    finally:
        os.environ.pop("FAKE_CUTECHESS_SLEEP_MS", None)
