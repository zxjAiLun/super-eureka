"""Startup recovery (section 15).

The worker runs recovery once at boot, before scheduling.  It normalizes any
state left behind by a crash or an unclean restart:

- RUNNING pair with a fully written, verifiable 2-game PGN -> completed and
  scored (no re-run).
- RUNNING pair without a usable PGN -> attempt interrupted (old directory is
  preserved), attempt number incremented, pair reset to PENDING to re-run the
  whole pair from scratch.  Missing single games are never patched in.
- RUNNING tournament -> back to QUEUED (or terminal if already finished).
- PAUSING tournament -> PAUSED, unless its current pair can be verified as
  complete, in which case the pair is scored first.

Idempotency: recovery only touches RUNNING/PAUSING state and only transitions
each row once; re-running it changes nothing.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..models import (
    CANCELLED,
    COMPLETED,
    FAILED,
    INTERRUPTED,
    PAUSED,
    PENDING,
    RUNNING,
    Event,
    Game,
    PairJob,
    Tournament,
    utcnow,
)
from . import artifacts
from . import verifier

logger = logging.getLogger("chessarena.recovery")


def _record_event(session, tournament_id, event_type, pair_job_id=None,
                  **payload) -> None:
    session.add(
        Event(
            tournament_id=tournament_id,
            pair_job_id=pair_job_id,
            event_type=event_type,
            payload=dict(payload),
        )
    )


def run_recovery(settings, session_factory) -> None:
    """Normalize tournament state after a worker (re)start."""
    with session_factory() as session:
        tournaments = (
            session.query(Tournament)
            .filter(Tournament.status.in_([RUNNING, "PAUSING"]))
            .all()
        )
        for tournament in tournaments:
            _recover_tournament(settings, session, tournament)
        session.commit()
    logger.info("recovery complete")


def _recover_tournament(settings, session, tournament) -> None:
    logger.info("recovering tournament %s (status=%s)", tournament.id, tournament.status)

    running_pairs = (
        session.query(PairJob)
        .filter(
            PairJob.tournament_id == tournament.id,
            PairJob.status == RUNNING,
        )
        .all()
    )

    for pair in running_pairs:
        recovered = _recover_running_pair(settings, session, tournament, pair)
        if not recovered:
            return  # pair/tournament reached a terminal state

    if tournament.status in (COMPLETED, FAILED, CANCELLED):
        return

    _record_event(session, tournament.id, "worker_recovered")

    if tournament.cancel_requested:
        tournament.status = CANCELLED
        tournament.finished_at = utcnow()
        return

    pending = (
        session.query(PairJob)
        .filter(
            PairJob.tournament_id == tournament.id,
            PairJob.status == PENDING,
        )
        .count()
    )
    if pending:
        if tournament.status == "PAUSING":
            tournament.status = PAUSED
            tournament.pause_requested = False
        else:
            tournament.status = "QUEUED"
        return

    # No pending pairs: all terminal.
    failed = (
        session.query(PairJob)
        .filter(
            PairJob.tournament_id == tournament.id,
            PairJob.status == FAILED,
        )
        .count()
    )
    if failed:
        tournament.status = FAILED
        tournament.finished_at = utcnow()
        tournament.failure_reason = "one or more pairs failed"
    else:
        tournament.status = COMPLETED
        tournament.finished_at = utcnow()
        _record_event(session, tournament.id, "tournament_completed")
        _generate_artifacts(session, tournament)


def _recover_running_pair(settings, session, tournament, pair) -> bool:
    """Recover one RUNNING pair.

    Returns False if the pair/tournament reached a terminal state (nothing
    else to schedule), True otherwise.
    """
    run_dir = Path(pair.run_directory) if pair.run_directory else None
    match_pgn = run_dir / "match.pgn" if run_dir is not None else None

    pgn_ok = (
        match_pgn is not None
        and match_pgn.exists()
        and len(verifier.parse_pgn(match_pgn)) == 2
    )

    if pgn_ok and run_dir is not None:
        try:
            verification = verifier.verify_pair(
                settings,
                tournament=tournament,
                pair_job=pair,
                run_dir=run_dir,
                engine_a_build=_engine_a(session, tournament),
                engine_b_build=_engine_b(session, tournament),
                opening_set=_opening_set(session, tournament),
            )
        except verifier.VerificationFailure as exc:
            logger.warning(
                "pair %s verification failed during recovery: %s", pair.id, exc
            )
            pair.status = FAILED
            pair.finished_at = utcnow()
            pair.failure_reason = f"verification failed during recovery: {exc}"
            pair.verification = {"verified": False, "reason": str(exc)}
            tournament.status = FAILED
            tournament.finished_at = utcnow()
            tournament.failure_reason = str(exc)
            _record_event(session, tournament.id, "pair_failed",
                          pair_job_id=pair.id, reason=str(exc))
            _record_event(session, tournament.id, "tournament_failed",
                          pair_job_id=pair.id, reason=str(exc))
            return False
        except Exception as exc:
            logger.exception("recovery verification error for pair %s", pair.id)
            pair.status = FAILED
            pair.finished_at = utcnow()
            pair.failure_reason = f"verification error during recovery: {exc}"
            pair.verification = {"verified": False, "reason": f"error: {exc}"}
            tournament.status = FAILED
            tournament.finished_at = utcnow()
            tournament.failure_reason = str(exc)
            _record_event(session, tournament.id, "pair_failed",
                          pair_job_id=pair.id, reason=str(exc))
            _record_event(session, tournament.id, "tournament_failed",
                          pair_job_id=pair.id, reason=str(exc))
            return False

        _complete_pair(session, tournament, pair, run_dir, verification)
        logger.info(
            "recovered RUNNING pair %s as COMPLETED (2-game PGN verified)", pair.id
        )
        return True

    # No usable PGN: interrupt the attempt and re-run the whole pair.
    logger.warning(
        "interrupting pair %s attempt %d (no complete 2-game PGN); will retry",
        pair.id,
        pair.attempt,
    )
    _record_event(
        session,
        tournament.id,
        "pair_interrupted",
        pair_job_id=pair.id,
        attempt=pair.attempt,
        run_directory=str(run_dir) if run_dir else None,
        reason="worker recovery: no complete pair PGN",
    )
    pair.status = INTERRUPTED
    pair.finished_at = utcnow()
    pair.failure_reason = "interrupted by recovery"
    pair.attempt += 1
    pair.status = PENDING
    pair.run_directory = None
    return True


def _engine_a(session, tournament):
    from ..models import EngineBuild

    return session.query(EngineBuild).filter(
        EngineBuild.build_id == tournament.engine_a_build_id
    ).first()


def _engine_b(session, tournament):
    from ..models import EngineBuild

    return session.query(EngineBuild).filter(
        EngineBuild.build_id == tournament.engine_b_build_id
    ).first()


def _opening_set(session, tournament):
    from ..models import OpeningSet

    return session.query(OpeningSet).filter(
        OpeningSet.opening_set_id == tournament.opening_set_id
    ).first()


def _complete_pair(session, tournament, pair, run_dir, verification) -> None:
    from ..config import ENGINE_A_NAME, ENGINE_B_NAME

    pgn_path = run_dir / "match.pgn"
    colors = [
        {"white": ENGINE_A_NAME, "black": ENGINE_B_NAME},
        {"white": ENGINE_B_NAME, "black": ENGINE_A_NAME},
    ]
    results = verification["results"]
    terminations = verification["terminations"]
    now = utcnow()

    # Idempotency guard: never create duplicate game records.
    existing = (
        session.query(Game)
        .filter(Game.pair_job_id == pair.id)
        .count()
    )
    if existing == 0:
        for idx in range(2):
            game = Game(
                tournament_id=tournament.id,
                pair_job_id=pair.id,
                game_number=pair.pair_index * 2 + idx + 1,
                white_engine=colors[idx]["white"],
                black_engine=colors[idx]["black"],
                opening_index=pair.opening_index,
                result=results[idx],
                termination=terminations[idx],
                pgn_path=str(pgn_path),
                started_at=pair.started_at,
                finished_at=now,
                verified=True,
            )
            session.add(game)
            session.flush()
            _record_event(
                session,
                tournament.id,
                "game_completed",
                pair_job_id=pair.id,
                game_id=game.id,
                game_number=game.game_number,
                result=results[idx],
                termination=terminations[idx],
            )

    pair.status = COMPLETED
    pair.finished_at = now
    pair.verification = verification
    pair.engine_a_white_game_id = (
        session.query(Game).filter(
            Game.pair_job_id == pair.id, Game.game_number == pair.pair_index * 2 + 1
        ).first().id
    )
    pair.engine_a_black_game_id = (
        session.query(Game).filter(
            Game.pair_job_id == pair.id, Game.game_number == pair.pair_index * 2 + 2
        ).first().id
    )

    computed = verification["candidate_perspective"]
    tournament.completed_pairs += 1
    tournament.candidate_wins += computed["wins"]
    tournament.candidate_losses += computed["losses"]
    tournament.draws += computed["draws"]

    _record_event(session, tournament.id, "pair_completed", pair_job_id=pair.id,
                  result=computed)
    _record_event(session, tournament.id, "verification_completed",
                  pair_job_id=pair.id)

    if tournament.completed_pairs >= tournament.requested_pairs:
        tournament.status = COMPLETED
        tournament.finished_at = now
        tournament.cancel_requested = False
        tournament.pause_requested = False
        _record_event(session, tournament.id, "tournament_completed")
        artifacts.generate_tournament_artifacts(tournament)


def _generate_artifacts(session, tournament) -> None:
    artifacts.generate_tournament_artifacts(tournament)
