"""Recovery tests (spec sections 15, 22.3).

Covers:
- RUNNING pair with no PGN -> interrupted, attempt+1, PENDING (re-run whole pair)
- RUNNING pair with a one-game PGN -> same
- RUNNING pair with a verified 2-game PGN -> COMPLETED and scored
- RUNNING pair with a 2-game PGN that fails verification -> pair FAILED,
  tournament FAILED
- recovery is idempotent (re-running does not double-score or duplicate games)
- PAUSED tournaments are not auto-resumed
- CANCELLED tournaments are not re-scheduled
"""

from __future__ import annotations

import json

import pytest

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
    Game,
    PairJob,
    Tournament,
)
from chessarena.services import recovery
from chessarena.services import verifier

from . import helpers


def _make_running_pair(engine_factory, tournament_factory, status=RUNNING):
    """A tournament whose first pair is RUNNING with the given status."""
    tournament_id = tournament_factory(status=status, pairs=3)
    with engine_factory() as session:
        pair = (
            session.query(PairJob)
            .filter(
                PairJob.tournament_id == tournament_id,
                PairJob.pair_index == 0,
            )
            .first()
        )
        pair.status = RUNNING
        session.commit()
        return tournament_id, pair.id


def _recover(settings, engine_factory, tournament_id=None):
    recovery.run_recovery(settings, engine_factory)
    with engine_factory() as session:
        return {
            "tournament": session.get(Tournament, tournament_id) if tournament_id else None,
            "pairs": (
                session.query(PairJob)
                .filter(PairJob.tournament_id == tournament_id)
                .order_by(PairJob.pair_index)
                .all()
                if tournament_id
                else []
            ),
            "games": (
                session.query(Game).filter(Game.tournament_id == tournament_id).all()
                if tournament_id
                else []
            ),
        }


def test_running_pair_no_pgn_retries(settings, engine_factory, tournament_factory):
    tournament_id, pair_id = _make_running_pair(engine_factory, tournament_factory)
    result = _recover(settings, engine_factory, tournament_id)
    pair = result["pairs"][0]
    assert pair.status == PENDING
    assert pair.attempt == 2
    assert pair.run_directory is None
    assert result["tournament"].status == QUEUED
    assert result["games"] == []


def test_running_pair_one_game_pgn_retries(settings, engine_factory, tournament_factory):
    tournament_id, pair_id = _make_running_pair(engine_factory, tournament_factory)
    from chessarena.services import artifacts

    run_dir = artifacts.pair_run_dir(tournament_id, 0, 1)
    run_dir.mkdir(parents=True)
    (run_dir / "match.pgn").write_text(
        '[Event "1"]\n[White "EngineA"]\n[Black "EngineB"]\n'
        '[Result "1-0"]\n\n1. e4 1-0\n\n',
        encoding="utf-8",
    )
    with engine_factory() as session:
        pair = session.get(PairJob, pair_id)
        pair.run_directory = str(run_dir)
        session.commit()
    result = _recover(settings, engine_factory, tournament_id)
    assert result["pairs"][0].status == PENDING
    assert result["pairs"][0].attempt == 2
    assert result["tournament"].status == QUEUED


def test_running_pair_verified_pgn_completes(settings, engine_factory,
                                             tournament_factory):
    tournament_id, pair_id = _make_running_pair(engine_factory, tournament_factory)
    # Build a valid 2-game PGN directory for the first pair.
    from chessarena.models import EngineBuild, OpeningSet

    with engine_factory() as session:
        tournament = session.get(Tournament, tournament_id)
        engine_a = (
            session.query(EngineBuild)
            .filter(EngineBuild.build_id == tournament.engine_a_build_id)
            .first()
        )
        engine_b = (
            session.query(EngineBuild)
            .filter(EngineBuild.build_id == tournament.engine_b_build_id)
            .first()
        )
        opening = (
            session.query(OpeningSet)
            .filter(OpeningSet.opening_set_id == tournament.opening_set_id)
            .first()
        )
        pair = session.get(PairJob, pair_id)

    run_dir = helpers.run_fake_pair(
        settings,
        tournament=tournament,
        pair_job=pair,
        engine_a_build=engine_a,
        engine_b_build=engine_b,
        opening_set=opening,
    )
    with engine_factory() as session:
        pair = session.get(PairJob, pair_id)
        pair.run_directory = str(run_dir)
        session.commit()

    result = _recover(settings, engine_factory, tournament_id)
    assert result["pairs"][0].status == COMPLETED
    assert result["tournament"].status == QUEUED  # 1 of 3 pairs done
    assert result["tournament"].completed_pairs == 1
    assert result["tournament"].candidate_wins == 2
    assert len(result["games"]) == 2
    assert all(g.verified for g in result["games"])


def test_running_pair_unverifiable_pgn_fails_tournament(settings, engine_factory,
                                                        tournament_factory):
    tournament_id, pair_id = _make_running_pair(engine_factory, tournament_factory)
    from chessarena.services import artifacts

    run_dir = artifacts.pair_run_dir(tournament_id, 0, 1)
    helpers.write_pair_artifacts(
        run_dir,
        games=[
            {"white": "EngineA", "black": "EngineA", "result": "1-0", "fen": "startpos"},
            {"white": "EngineB", "black": "EngineA", "result": "1-0", "fen": "startpos"},
        ],
        stdout_lines=["Score of EngineA vs EngineB: 1 - 0 - 0  [1.000] 2"],
        command_argv=[str(settings.cutechess), "-engine", "name=EngineA"],
    )
    # Fix the opening position to match the registered opening set.
    from chessarena.models import OpeningSet

    with engine_factory() as session:
        opening = session.query(OpeningSet).first()
        fen = helpers.opening_fen_for_index(opening, 0)
        pgn = (run_dir / "match.pgn").read_text()
        pgn = pgn.replace('"startpos"', f'"{fen}"')
        (run_dir / "match.pgn").write_text(pgn)
        pair = session.get(PairJob, pair_id)
        pair.run_directory = str(run_dir)
        session.commit()

    result = _recover(settings, engine_factory, tournament_id)
    assert result["pairs"][0].status == FAILED
    assert result["tournament"].status == FAILED
    assert "color assignment" in result["pairs"][0].failure_reason


def test_recovery_idempotent(settings, engine_factory, tournament_factory):
    """Re-running recovery must not double-count or duplicate games."""
    tournament_id, pair_id = _make_running_pair(engine_factory, tournament_factory)
    recovery.run_recovery(settings, engine_factory)
    recovery.run_recovery(settings, engine_factory)
    with engine_factory() as session:
        tournament = session.get(Tournament, tournament_id)
        assert tournament.completed_pairs == 0
        games = session.query(Game).filter(Game.tournament_id == tournament_id).all()
        assert games == []
        pairs = (
            session.query(PairJob)
            .filter(PairJob.tournament_id == tournament_id)
            .all()
        )
        assert all(p.status == PENDING for p in pairs)


def test_recovery_paused_not_resumed(settings, engine_factory, tournament_factory):
    tournament_id = tournament_factory(status=PAUSED, pairs=3)
    _recover(settings, engine_factory, tournament_id)
    with engine_factory() as session:
        assert session.get(Tournament, tournament_id).status == PAUSED


def test_recovery_cancelled_not_rescheduled(settings, engine_factory,
                                            tournament_factory):
    tournament_id = tournament_factory(status=CANCELLED, pairs=3)
    _recover(settings, engine_factory, tournament_id)
    with engine_factory() as session:
        assert session.get(Tournament, tournament_id).status == CANCELLED


def test_recovery_pausing_becomes_paused(settings, engine_factory,
                                         tournament_factory):
    tournament_id = tournament_factory(status=PAUSING, pairs=3)
    with engine_factory() as session:
        tournament = session.get(Tournament, tournament_id)
        tournament.pause_requested = True
        session.commit()
    _recover(settings, engine_factory, tournament_id)
    with engine_factory() as session:
        tournament = session.get(Tournament, tournament_id)
        assert tournament.status == PAUSED
        assert tournament.pause_requested is False


def test_recovery_completes_full_tournament(settings, engine_factory,
                                            tournament_factory):
    """RUNNING tournament whose pairs are all done becomes COMPLETED."""
    tournament_id = tournament_factory(status=RUNNING, pairs=1)
    with engine_factory() as session:
        pair = (
            session.query(PairJob)
            .filter(PairJob.tournament_id == tournament_id)
            .first()
        )
        pair.status = COMPLETED
        tournament = session.get(Tournament, tournament_id)
        tournament.completed_pairs = 1
        tournament.candidate_wins = 2
        session.commit()
    _recover(settings, engine_factory, tournament_id)
    with engine_factory() as session:
        tournament = session.get(Tournament, tournament_id)
        assert tournament.status == COMPLETED
