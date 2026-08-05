"""Startup recovery (section 15, P1.2/P1.3/P1.4/P2.1/P2.2).

The worker runs recovery once at boot, before scheduling.  It normalizes any
state left behind by a crash or an unclean restart:

- RUNNING pair with a fully written, verifiable 2-game PGN -> completed and
  scored (no re-run).
- RUNNING pair without a usable PGN -> attempt interrupted (old directory is
  preserved), attempt number incremented, pair reset to PENDING to re-run the
  whole pair from scratch.  Missing single games are never patched in.
- INTERRUPTED pair in a non-CANCELLED tournament -> attempt incremented and
  reset to PENDING so the whole pair is re-run (this is what makes graceful
  worker shutdown / deploy restarts safe).
- An orphaned cutechess process recorded in worker_state is terminated before
  its pair is re-run (P1.4).
- RUNNING tournament -> back to QUEUED (or terminal only when the strict
  completion conditions hold).
- PAUSING tournament -> PAUSED or CANCELLED (cancel takes priority), unless
  its current pair can be verified as complete, in which case the pair is
  scored first.
- force_cancel_requested -> the pair is interrupted (orphan killed) and the
  tournament is CANCELLED; nothing is scored.

A tournament may only be marked COMPLETED when every one of the following
holds (P1.2):
  completed_pairs == requested_pairs
  every PairJob.status == COMPLETED
  the number of verified games == requested_pairs * 2

Idempotency: recovery only transitions each row once; re-running it changes
nothing (INTERRUPTED -> PENDING happens once, RUNNING recovery only touches
RUNNING rows, scoring happens only when a pair is verified).
"""

from __future__ import annotations

import json
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


def can_mark_completed(session, tournament) -> bool:
    """Strict completion check (P1.2): all pairs COMPLETED + exact game count."""
    if tournament.completed_pairs != tournament.requested_pairs:
        return False
    pairs = (
        session.query(PairJob)
        .filter(PairJob.tournament_id == tournament.id)
        .all()
    )
    if not pairs or any(p.status != COMPLETED for p in pairs):
        return False
    game_count = (
        session.query(Game)
        .filter(Game.tournament_id == tournament.id)
        .count()
    )
    return game_count == tournament.requested_pairs * 2


def atomic_complete_or_cancel(session, tournament, now) -> str:
    """Atomically transition a finished tournament to COMPLETED or CANCELLED.

    Returns the final status.  Every path that can mark a tournament
    COMPLETED must go through this helper so the global invariant holds: the
    row can never end as COMPLETED while a cancel/force-cancel flag is
    present.  The COMPLETED write is a conditional UPDATE guarded by the
    flags, so a cancel committed concurrently (which is itself a conditional
    UPDATE) makes this match zero rows under the SQLite write lock, and the
    CANCELLED branch is applied instead (P2.2).
    """
    from sqlalchemy import update

    result = session.execute(
        update(Tournament)
        .where(Tournament.id == tournament.id)
        .where(Tournament.cancel_requested.is_(False))
        .where(Tournament.force_cancel_requested.is_(False))
        .values(
            status=COMPLETED,
            finished_at=now,
            cancel_requested=False,
            pause_requested=False,
            force_cancel_requested=False,
        )
    )
    if result.rowcount == 1:
        return COMPLETED
    session.execute(
        update(Tournament)
        .where(Tournament.id == tournament.id)
        .values(
            status=CANCELLED,
            finished_at=now,
            cancel_requested=True,
            pause_requested=False,
            force_cancel_requested=False,
        )
    )
    return CANCELLED


def reschedule_interrupted_pairs(session, tournament) -> int:
    """INTERRUPTED pairs in a non-CANCELLED tournament become PENDING again.

    attempt is incremented and run_directory cleared so the whole pair is
    re-run from scratch in a fresh attempt directory.  The old attempt
    directory is preserved on disk.  Returns the number rescheduled.
    """
    if tournament.status == CANCELLED:
        return 0
    pairs = (
        session.query(PairJob)
        .filter(
            PairJob.tournament_id == tournament.id,
            PairJob.status == INTERRUPTED,
        )
        .all()
    )
    for pair in pairs:
        _record_event(
            session,
            tournament.id,
            "pair_interrupted",
            pair_job_id=pair.id,
            attempt=pair.attempt,
            reason="rescheduled by recovery",
        )
        pair.attempt += 1
        pair.status = PENDING
        pair.run_directory = None
        # P1: exit evidence from the old attempt must not leak into the new one.
        pair.return_code = None
        pair.return_code_attempt = None
    return len(pairs)


def run_recovery(settings, session_factory) -> None:
    """Normalize tournament state after a worker (re)start."""
    with session_factory() as session:
        tournaments = (
            session.query(Tournament)
            .filter(Tournament.status.in_([RUNNING, "PAUSING", PAUSED]))
            .all()
        )
        for tournament in tournaments:
            _recover_tournament(settings, session, tournament)
        session.commit()
    logger.info("recovery complete")


def _recover_tournament(settings, session, tournament) -> None:
    logger.info("recovering tournament %s (status=%s)", tournament.id, tournament.status)

    # P1.3: an explicit force-cancel wins over any other state.
    if tournament.force_cancel_requested:
        _cancel_with_force(settings, session, tournament)
        return

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

    # P1.2: pairs interrupted by a previous worker shutdown are re-run.
    reschedule_interrupted_pairs(session, tournament)

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
            # P2.1: cancel takes priority over pause.
            if tournament.cancel_requested:
                tournament.status = CANCELLED
                tournament.finished_at = utcnow()
            else:
                tournament.status = PAUSED
                tournament.pause_requested = False
        elif tournament.status == RUNNING:
            tournament.status = "QUEUED"
        # PAUSED tournaments stay PAUSED (they were never auto-resumed);
        # their INTERRUPTED pairs were just rescheduled to PENDING above so a
        # later resume picks them up.
        return

    # No pending pairs: strictly validate before calling it COMPLETED.
    if can_mark_completed(session, tournament):
        final = atomic_complete_or_cancel(session, tournament, utcnow())
        session.expire(tournament)
        if final == COMPLETED:
            _record_event(session, tournament.id, "tournament_completed")
            _generate_artifacts(session, tournament)
        else:
            _record_event(
                session, tournament.id, "tournament_cancelled",
                reason="concurrent cancel",
            )
    else:
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
            tournament.status = FAILED
            tournament.finished_at = utcnow()
            tournament.failure_reason = (
                f"pairs incomplete: {tournament.completed_pairs}/"
                f"{tournament.requested_pairs} completed; recovery cannot "
                "safely complete this tournament"
            )
            _record_event(session, tournament.id, "tournament_failed",
                          reason=tournament.failure_reason)


def _cancel_with_force(settings, session, tournament) -> None:
    """Honor an explicit force-cancel: kill orphans, interrupt pairs, cancel."""
    running_pairs = (
        session.query(PairJob)
        .filter(
            PairJob.tournament_id == tournament.id,
            PairJob.status == RUNNING,
        )
        .all()
    )
    for pair in running_pairs:
        _kill_orphaned_pair(session, pair, settings)
        pair.status = INTERRUPTED
        pair.finished_at = utcnow()
        pair.failure_reason = "force-cancelled"
        _record_event(session, tournament.id, "pair_interrupted",
                      pair_job_id=pair.id, reason="force-cancelled")
    tournament.force_cancel_requested = False
    tournament.cancel_requested = True
    tournament.status = CANCELLED
    tournament.finished_at = utcnow()
    _record_event(session, tournament.id, "tournament_cancelled",
                  reason="force")


def _recover_running_pair(settings, session, tournament, pair) -> bool:
    """Recover one RUNNING pair.

    Returns False if the pair/tournament reached a terminal state (nothing
    else to schedule), True otherwise.
    """
    run_dir = Path(pair.run_directory) if pair.run_directory else None
    match_pgn = run_dir / "match.pgn" if run_dir is not None else None

    # P1: terminate an orphaned cutechess recorded in worker_state before
    # doing anything with this pair.
    _kill_orphaned_pair(session, pair, settings)

    # P1: the manager exit code must have been recorded by the worker for the
    # CURRENT attempt before we may ever score a pair from its PGN.  If the
    # worker died before recording it (or the evidence belongs to an older
    # attempt), there is no trustworthy evidence: interrupt and re-run.
    evidence_ok = (
        pair.return_code is not None
        and pair.return_code_attempt is not None
        and pair.return_code_attempt == pair.attempt
    )
    if not evidence_ok:
        return _interrupt_pair(session, tournament, pair, run_dir,
                               "no manager exit-code evidence for this attempt")

    pgn_ok = (
        match_pgn is not None
        and match_pgn.exists()
        and len(verifier.parse_pgn(match_pgn)) == 2
    )

    if not (pgn_ok and run_dir is not None):
        return _interrupt_pair(
            session, tournament, pair, run_dir, "no complete 2-game PGN"
        )

    if pair.return_code != 0:
        logger.warning(
            "pair %s has recorded non-zero exit code %s; failing the pair "
            "and the tournament without scoring",
            pair.id, pair.return_code,
        )
        return _fail_pair(session, tournament, pair,
                          f"cutechess exited with code {pair.return_code}",
                          run_dir=run_dir)

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
        return _fail_pair(session, tournament, pair,
                          f"verification failed during recovery: {exc}",
                          verification={"verified": False, "reason": str(exc)},
                          run_dir=run_dir)
    except Exception as exc:
        logger.exception("recovery verification error for pair %s", pair.id)
        return _fail_pair(session, tournament, pair,
                          f"verification error during recovery: {exc}",
                          verification={"verified": False, "reason": f"error: {exc}"},
                          run_dir=run_dir)

    verification["return_code"] = pair.return_code
    _complete_pair(session, tournament, pair, run_dir, verification)
    logger.info(
        "recovered RUNNING pair %s as COMPLETED (exit 0 + 2-game PGN verified)",
        pair.id,
    )
    return True


def _interrupt_pair(session, tournament, pair, run_dir, reason: str) -> bool:
    """Mark the attempt interrupted and reset the pair for a full re-run."""
    logger.warning("interrupting pair %s attempt %d (%s)", pair.id, pair.attempt,
                   reason)
    _record_event(
        session,
        tournament.id,
        "pair_interrupted",
        pair_job_id=pair.id,
        attempt=pair.attempt,
        run_directory=str(run_dir) if run_dir else None,
        reason=reason,
    )
    pair.status = INTERRUPTED
    pair.finished_at = utcnow()
    pair.failure_reason = reason
    pair.attempt += 1
    pair.status = PENDING
    pair.run_directory = None
    # P1: exit evidence from the old attempt must not leak into the new one.
    pair.return_code = None
    pair.return_code_attempt = None
    return True


def _fail_pair(session, tournament, pair, reason, verification=None,
               run_dir=None) -> bool:
    """Fail the pair and the tournament; artifacts are retained (P1.5)."""
    pair.status = FAILED
    pair.finished_at = utcnow()
    pair.failure_reason = reason
    pair.verification = verification or {
        "verified": False,
        "reason": reason,
    }
    # P1: exit evidence must reach verification.json and the events on the
    # failure path too, not only on the success path.
    if pair.return_code is not None:
        pair.verification["return_code"] = pair.return_code
    tournament.status = FAILED
    tournament.finished_at = utcnow()
    tournament.failure_reason = reason
    _record_event(session, tournament.id, "pair_failed",
                  pair_job_id=pair.id, reason=reason,
                  return_code=pair.return_code)
    _record_event(session, tournament.id, "tournament_failed",
                  pair_job_id=pair.id, reason=reason,
                  return_code=pair.return_code)
    if run_dir is not None:
        artifacts.write_json(run_dir, "verification.json", pair.verification or {})
    return False


def _kill_orphaned_pair(session, pair, settings=None) -> None:
    """Terminate an orphaned cutechess process group recorded for this pair.

    worker_state.pid is the cutechess process PID (and process group leader,
    since it was launched with start_new_session).  Cleanup is GROUP-aware:

    - If the leader is still alive, its recorded identity (starttime +
      cmdline on Linux) is verified first so a recycled PID can never kill an
      unrelated process.
    - If the leader is already gone but children survive in the same PGID,
      the group is still cleaned (a dead leader must not leak engine
      children).
    - Identity is cleared ONLY after the whole process group is confirmed
      gone; a group that survives SIGKILL keeps its identity so it is not
      silently lost (P1 gaps A/B).
    """
    from ..models import WorkerState
    from . import cutechess as cc

    state = session.get(WorkerState, 1)
    if state is None or state.pair_job_id != pair.id or not state.pid:
        return
    pid = state.pid
    grace = (
        getattr(settings, "shutdown_grace_seconds", 15.0)
        if settings is not None
        else 15.0
    )

    if not _pid_alive(pid):
        # Gap A: leader gone but children may remain in the same PGID.
        if not _group_alive(pid):
            _clear_worker_identity(session, state)
            return
        logger.warning(
            "orphaned cutechess leader pid %d is gone but its process group "
            "still has members; cleaning the group",
            pid,
        )
        if _terminate_pid_group_wait(pid, grace):
            _clear_worker_identity(session, state)
        else:
            logger.error(
                "orphan group %d survived SIGKILL; retaining its identity",
                pid,
            )
        return

    recorded_marker = state.pid_start_marker
    recorded_cmdline = None
    if state.pid_cmdline:
        try:
            recorded_cmdline = json.loads(state.pid_cmdline)
        except ValueError:
            recorded_cmdline = None

    if not cc.verify_process_identity(pid, recorded_marker, recorded_cmdline):
        # The PID no longer refers to the recorded cutechess process (it was
        # recycled) - never kill it, just drop the stale identity.
        logger.warning(
            "worker_state pid %d does not match recorded identity for pair %s; "
            "not killing (PID reuse guard)",
            pid, pair.id,
        )
        _clear_worker_identity(session, state)
        return

    logger.warning(
        "terminating orphaned cutechess pid %d for pair %s", pid, pair.id
    )
    if _terminate_pid_group_wait(pid, grace):
        _clear_worker_identity(session, state)
    else:
        logger.error(
            "orphan group %d survived SIGKILL; retaining its identity", pid
        )


def _clear_worker_identity(session, state) -> None:
    state.pid = None
    state.pair_job_id = None
    state.pid_start_marker = None
    state.pid_cmdline = None


def _pid_alive(pid: int) -> bool:
    """True when a process with ``pid`` exists (safe on Windows).

    Windows os.kill(pid, 0) would terminate the process, so the shared safe
    probe is used on every platform.
    """
    from . import cutechess as cc

    return cc._pid_alive(pid)


def _group_alive(pgid: int) -> bool:
    """True when ANY member of the process group still exists (P1).

    This is what the cleanup wait must poll: the group leader may exit while
    a child (engine process) survives, and only a group-wide check keeps
    sending SIGKILL until the whole group is gone.
    """
    import os

    if pgid <= 0:
        return False
    if hasattr(os, "killpg"):
        try:
            os.killpg(pgid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False
    # Non-POSIX fallback: the leader is the whole group for our launches.
    return _pid_alive(pgid)


def _terminate_pid_group_wait(pgid: int, grace: float) -> bool:
    """Group-aware termination shared with active force-cancel (P1 gap C).

    Returns True only when the entire process group is confirmed gone;
    False when a member survived SIGKILL, in which case the caller must
    retain the process identity.
    """
    from . import cutechess as cc

    return cc.terminate_process_group_by_pid(pgid, grace)


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
    pair.return_code = verification.get("return_code", 0)
    # Persist the verification for this attempt (P1: recovery must produce the
    # same provenance artifacts the normal scheduler path produces).
    artifacts.write_json(run_dir, "verification.json", verification)
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
        # P2.2: the final transition is atomic - cancel wins if it landed.
        final = atomic_complete_or_cancel(session, tournament, now)
        session.expire(tournament)
        if final == COMPLETED:
            _record_event(session, tournament.id, "tournament_completed")
            artifacts.generate_tournament_artifacts(tournament)
        else:
            _record_event(
                session, tournament.id, "tournament_cancelled",
                reason="concurrent cancel",
            )


def _generate_artifacts(session, tournament) -> None:
    artifacts.generate_tournament_artifacts(tournament)
