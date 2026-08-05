"""Single-worker pair scheduler (sections 10, 11).

The scheduler owns exactly one cutechess process at a time.  A tick does one
unit of work: poll the active process, or claim the next PENDING pair from the
earliest QUEUED/RUNNING tournament and launch it.

Pause / cancel semantics:
- pause_requested: the current pair finishes normally, then the tournament is
  set to PAUSED and no new pair starts.
- cancel_requested: the current pair finishes normally, then the tournament is
  set to CANCELLED.  Cancel wins over both pause and automatic completion.
- force_cancel_requested: the running process group is killed immediately and
  the attempt is marked INTERRUPTED (not scored).  The flag lives in the
  database (not in-process memory) because the API and the worker are separate
  processes in production (P1.3).
- A non-zero cutechess exit code fails the pair and the tournament even when
  the artifacts look complete (P1.5).

Pair boundaries are the only scoring points; a half-completed pair is never
counted.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from ..config import Settings
from ..models import (
    CANCELLED,
    COMPLETED,
    FAILED,
    INTERRUPTED,
    PAUSED,
    PENDING,
    QUEUED,
    RUNNING,
    EngineBuild,
    Event,
    Game,
    OpeningSet,
    PairJob,
    Tournament,
    utcnow,
)
from . import artifacts
from . import cutechess as cc
from . import recovery
from . import verifier


def _record_event(session, tournament_id, event_type, pair_job_id=None,
                  game_id=None, **payload) -> None:
    session.add(
        Event(
            tournament_id=tournament_id,
            pair_job_id=pair_job_id,
            game_id=game_id,
            event_type=event_type,
            payload=dict(payload),
        )
    )


class Scheduler:
    def __init__(self, settings: Settings, session_factory):
        self.settings = settings
        self.session_factory = session_factory
        self.active_tournament_id: Optional[str] = None
        self.active_pair_job_id: Optional[str] = None
        self.active_proc: Optional[subprocess.Popen] = None
        self.active_run_dir: Optional[Path] = None

    # ------------------------------------------------------------------
    # Tick
    # ------------------------------------------------------------------
    def tick(self) -> str:
        """One unit of work.  Returns a short description for logging."""
        with self.session_factory() as session:
            if self.active_proc is not None:
                return self._poll_active(session)
            return self._find_and_launch(session)

    # ------------------------------------------------------------------
    # Active process handling
    # ------------------------------------------------------------------
    def _poll_active(self, session) -> str:
        tournament = session.get(Tournament, self.active_tournament_id)
        if tournament is not None and tournament.force_cancel_requested:
            return self._force_kill_active(session)

        proc = self.active_proc
        if proc.poll() is None:
            return f"pair running: {self.active_pair_job_id}"

        rc = proc.returncode
        self._close_output_handles(proc)
        pair = session.get(PairJob, self.active_pair_job_id)
        tournament = session.get(Tournament, self.active_tournament_id)
        if pair is not None and tournament is not None and pair.status == RUNNING:
            self._finish_pair(session, tournament, pair, return_code=rc)
        else:
            # Already interrupted/cancelled externally; nothing to score.
            pass

        self._clear_active()
        session.commit()
        return f"pair finished: {self.active_pair_job_id} rc={rc}"

    def _force_kill_active(self, session) -> str:
        pair_id = self.active_pair_job_id
        tournament_id = self.active_tournament_id
        proc = self.active_proc
        if proc is not None:
            cc.terminate_process_group(proc, self.settings.shutdown_grace_seconds)
            self._close_output_handles(proc)
        pair = session.get(PairJob, pair_id)
        tournament = session.get(Tournament, tournament_id)
        if pair is not None and pair.status == RUNNING:
            pair.status = INTERRUPTED
            pair.finished_at = utcnow()
            pair.failure_reason = "force-cancelled"
            _record_event(
                session,
                pair.tournament_id,
                "pair_interrupted",
                pair_job_id=pair.id,
                reason="force-cancelled",
            )
        if tournament is not None:
            tournament.force_cancel_requested = False
            tournament.cancel_requested = True
            tournament.status = CANCELLED
            tournament.finished_at = utcnow()
            _record_event(
                session, tournament.id, "tournament_cancelled", reason="force"
            )
        self._clear_active()
        session.commit()
        return f"force-cancelled: {tournament_id}"

    def _close_output_handles(self, proc: subprocess.Popen) -> None:
        for attr in ("_stdout_fh", "_stderr_fh"):
            fh = getattr(proc, attr, None)
            if fh is not None:
                try:
                    fh.close()
                except OSError:
                    pass

    def _clear_active(self) -> None:
        self.active_proc = None
        self.active_tournament_id = None
        self.active_pair_job_id = None
        self.active_run_dir = None

    # ------------------------------------------------------------------
    # Pair completion
    # ------------------------------------------------------------------
    def _finish_pair(self, session, tournament: Tournament, pair: PairJob,
                     return_code: int | None = None) -> None:
        """Verify the completed pair and update DB state (section 14, 10.1).

        ``return_code`` is the cutechess process exit code.  A non-zero exit
        code fails the pair even when the artifacts look complete (P1.5); the
        verifier still runs to produce diagnostics, but nothing is scored.
        """
        run_dir = Path(pair.run_directory) if pair.run_directory else None
        if run_dir is None or not run_dir.exists():
            self._fail_pair(
                session, tournament, pair, "pair run directory missing",
                return_code=return_code,
            )
            return

        engine_a = session.query(EngineBuild).filter(
            EngineBuild.build_id == tournament.engine_a_build_id
        ).first()
        engine_b = session.query(EngineBuild).filter(
            EngineBuild.build_id == tournament.engine_b_build_id
        ).first()
        opening_set = session.query(OpeningSet).filter(
            OpeningSet.opening_set_id == tournament.opening_set_id
        ).first()

        pair.status = "VERIFYING"
        session.flush()
        verification, error = self._run_verifier(
            session, tournament, pair, run_dir, engine_a, engine_b, opening_set
        )
        if error:
            self._fail_pair(
                session, tournament, pair, error,
                verification=verification, return_code=return_code,
            )
            return
        if return_code not in (None, 0):
            # Artifacts look valid but the manager crashed -> never score.
            self._fail_pair(
                session, tournament, pair,
                f"cutechess exited with code {return_code}",
                verification=verification, return_code=return_code,
            )
            return
        self._complete_pair(session, tournament, pair, run_dir, verification)

    def _run_verifier(self, session, tournament, pair, run_dir,
                      engine_a, engine_b, opening_set):
        """Run verification; returns (verification_dict, error_string|None)."""
        try:
            verification = verifier.verify_pair(
                self.settings,
                tournament=tournament,
                pair_job=pair,
                run_dir=run_dir,
                engine_a_build=engine_a,
                engine_b_build=engine_b,
                opening_set=opening_set,
            )
            return verification, None
        except verifier.VerificationFailure as exc:
            return {"verified": False, "reason": str(exc)}, (
                f"verification failed: {exc}"
            )
        except Exception as exc:  # unexpected error -> treat as failure
            return {"verified": False, "reason": f"error: {exc}"}, (
                f"verification error: {exc}"
            )

    def _fail_pair(self, session, tournament, pair, reason, verification=None,
                   return_code: int | None = None) -> None:
        pair.status = FAILED
        pair.finished_at = utcnow()
        pair.failure_reason = reason
        pair.return_code = return_code
        pair.verification = verification or {
            "verified": False,
            "reason": reason,
        }
        if return_code is not None:
            pair.verification["return_code"] = return_code
        tournament.status = FAILED
        tournament.finished_at = utcnow()
        tournament.failure_reason = reason
        _record_event(
            session,
            tournament.id,
            "pair_failed",
            pair_job_id=pair.id,
            reason=reason,
            return_code=return_code,
        )
        _record_event(
            session,
            tournament.id,
            "tournament_failed",
            pair_job_id=pair.id,
            reason=reason,
            return_code=return_code,
        )
        self._write_pair_verification(pair)

    def _complete_pair(self, session, tournament, pair, run_dir, verification) -> None:
        """Record games, aggregate score, and finish the tournament if done."""
        from ..config import ENGINE_A_NAME, ENGINE_B_NAME

        pgn_path = run_dir / "match.pgn"
        colors = [
            {"white": ENGINE_A_NAME, "black": ENGINE_B_NAME},
            {"white": ENGINE_B_NAME, "black": ENGINE_A_NAME},
        ]
        results = verification["results"]
        terminations = verification["terminations"]
        now = utcnow()
        game_records: list[Game] = []
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
            game_records.append(game)
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

        # Strict color swap -> game 0 is A as White, game 1 is A as Black.
        pair.engine_a_white_game_id = game_records[0].id
        pair.engine_a_black_game_id = game_records[1].id
        pair.status = COMPLETED
        pair.finished_at = now
        pair.verification = verification
        self._write_pair_verification(pair)

        computed = verification["candidate_perspective"]
        tournament.completed_pairs += 1
        tournament.candidate_wins += computed["wins"]
        tournament.candidate_losses += computed["losses"]
        tournament.draws += computed["draws"]

        _record_event(
            session,
            tournament.id,
            "pair_completed",
            pair_job_id=pair.id,
            result=computed,
        )
        _record_event(
            session,
            tournament.id,
            "verification_completed",
            pair_job_id=pair.id,
        )

        if tournament.completed_pairs >= tournament.requested_pairs:
            # P2.2: an explicit cancel wins over automatic completion.
            if tournament.cancel_requested:
                tournament.status = CANCELLED
                tournament.finished_at = now
                tournament.pause_requested = False
            else:
                tournament.status = COMPLETED
                tournament.finished_at = now
                tournament.cancel_requested = False
                tournament.pause_requested = False
                _record_event(session, tournament.id, "tournament_completed")
                artifacts.generate_tournament_artifacts(tournament)

    def _write_pair_verification(self, pair: PairJob) -> None:
        if not pair.run_directory:
            return
        artifacts.write_json(
            Path(pair.run_directory), "verification.json", pair.verification or {}
        )

    # ------------------------------------------------------------------
    # Launch next pair
    # ------------------------------------------------------------------
    def _find_and_launch(self, session) -> str:
        # A PAUSING tournament has no running pair (it would be polled above);
        # finalize the pause now.  P2.1: cancel wins over pause.
        pausing = (
            session.query(Tournament)
            .filter(Tournament.status == "PAUSING")
            .order_by(Tournament.created_at.asc())
            .first()
        )
        if pausing is not None:
            if pausing.cancel_requested:
                pausing.status = CANCELLED
                pausing.finished_at = utcnow()
                _record_event(
                    session, pausing.id, "tournament_cancelled", reason="worker"
                )
            else:
                pausing.status = PAUSED
                pausing.pause_requested = False
                _record_event(session, pausing.id, "tournament_paused")
            session.commit()
            return f"paused: {pausing.id}"

        tournament = self._next_tournament(session)
        if tournament is None:
            return "idle"

        if tournament.force_cancel_requested:
            tournament.force_cancel_requested = False
            tournament.cancel_requested = True
            tournament.status = CANCELLED
            tournament.finished_at = utcnow()
            _record_event(
                session, tournament.id, "tournament_cancelled", reason="force"
            )
            session.commit()
            return f"force-cancelled (no running pair): {tournament.id}"

        if tournament.cancel_requested:
            tournament.status = CANCELLED
            tournament.finished_at = utcnow()
            _record_event(session, tournament.id, "tournament_cancelled", reason="worker")
            session.commit()
            return f"cancelled (no running pair): {tournament.id}"

        if tournament.pause_requested:
            tournament.status = PAUSED
            tournament.pause_requested = False
            _record_event(session, tournament.id, "tournament_paused")
            session.commit()
            return f"paused (no running pair): {tournament.id}"

        # P1.2: pairs interrupted by an earlier worker shutdown are re-run.
        recovery.reschedule_interrupted_pairs(session, tournament)

        pair = self._next_pending_pair(session, tournament)
        if pair is None:
            return self._finish_without_pending(session, tournament)

        pair.status = RUNNING
        pair.started_at = utcnow()
        tournament.status = RUNNING
        tournament.started_at = tournament.started_at or utcnow()
        session.flush()

        run_dir = artifacts.pair_run_dir(tournament.id, pair.pair_index, pair.attempt)
        pair.run_directory = str(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        try:
            self._prepare_and_launch(session, tournament, pair, run_dir)
        except cc.CutechessLaunchError as exc:
            pair.status = FAILED
            pair.finished_at = utcnow()
            pair.failure_reason = str(exc)
            tournament.status = FAILED
            tournament.finished_at = utcnow()
            tournament.failure_reason = str(exc)
            _record_event(
                session, tournament.id, "pair_failed", pair_job_id=pair.id,
                reason=str(exc),
            )
            _record_event(
                session, tournament.id, "tournament_failed", pair_job_id=pair.id,
                reason=str(exc),
            )
            session.commit()
            return f"launch failed: {str(exc)}"

        _record_event(
            session,
            tournament.id,
            "pair_started",
            pair_job_id=pair.id,
            pair_index=pair.pair_index,
            opening_index=pair.opening_index,
            attempt=pair.attempt,
            run_directory=str(run_dir),
        )
        session.commit()
        return f"launched pair {pair.pair_index} attempt {pair.attempt}"

    def _next_tournament(self, session) -> Optional[Tournament]:
        return (
            session.query(Tournament)
            .filter(Tournament.status.in_([QUEUED, RUNNING]))
            .order_by(Tournament.created_at.asc())
            .first()
        )

    def _next_pending_pair(self, session, tournament) -> Optional[PairJob]:
        return (
            session.query(PairJob)
            .filter(
                PairJob.tournament_id == tournament.id,
                PairJob.status == PENDING,
            )
            .order_by(PairJob.pair_index.asc())
            .first()
        )

    def _finish_without_pending(self, session, tournament) -> str:
        """All pairs terminal: strictly validate before completing (P1.2)."""
        if tournament.cancel_requested:
            tournament.status = CANCELLED
            tournament.finished_at = utcnow()
            session.commit()
            return f"cancelled: {tournament.id}"

        if recovery.can_mark_completed(session, tournament):
            tournament.status = COMPLETED
            tournament.finished_at = utcnow()
            tournament.cancel_requested = False
            tournament.pause_requested = False
            _record_event(session, tournament.id, "tournament_completed")
            artifacts.generate_tournament_artifacts(tournament)
            session.commit()
            return f"tournament completed: {tournament.id}"

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
            _record_event(session, tournament.id, "tournament_failed",
                          reason="pairs failed")
        else:
            tournament.status = FAILED
            tournament.finished_at = utcnow()
            tournament.failure_reason = (
                f"pairs incomplete: {tournament.completed_pairs}/"
                f"{tournament.requested_pairs} completed"
            )
            _record_event(session, tournament.id, "tournament_failed",
                          reason=tournament.failure_reason)
        session.commit()
        return f"tournament failed (incomplete pairs): {tournament.id}"

    # ------------------------------------------------------------------
    # Launch internals
    # ------------------------------------------------------------------
    def _prepare_and_launch(self, session, tournament, pair, run_dir) -> None:
        opening_set = session.query(OpeningSet).filter(
            OpeningSet.opening_set_id == tournament.opening_set_id
        ).first()
        engine_a = session.query(EngineBuild).filter(
            EngineBuild.build_id == tournament.engine_a_build_id
        ).first()
        engine_b = session.query(EngineBuild).filter(
            EngineBuild.build_id == tournament.engine_b_build_id
        ).first()

        if opening_set is None or engine_a is None or engine_b is None:
            raise cc.CutechessLaunchError("referenced build/opening not found")

        opening_fen = _opening_fen_for_index(opening_set, pair.opening_index)
        opening_epd = run_dir / "opening.epd"
        opening_epd.write_text(opening_fen + "\n", encoding="utf-8")

        from ..config import TIME_CONTROLS

        tc = TIME_CONTROLS[tournament.time_control]["cutechess_tc"]

        argv = cc.build_pair_command(
            self.settings,
            engine_a={
                "build_id": engine_a.build_id,
                "binary_path": engine_a.binary_path,
                "profile": tournament.engine_a_profile,
            },
            engine_b={
                "build_id": engine_b.build_id,
                "binary_path": engine_b.binary_path,
                "profile": tournament.engine_b_profile,
            },
            time_control=tc,
            hash_mb=self.settings.hash_mb,
            opening_epd=opening_epd,
            pgn_out=run_dir / "match.pgn",
        )

        # Pre-flight checks (section 12)
        cc.check_cutechess(self.settings)
        cc.check_engine_binary(
            {"binary_path": engine_a.binary_path, "binary_sha256": engine_a.binary_sha256,
             "build_id": engine_a.build_id}
        )
        cc.check_engine_binary(
            {"binary_path": engine_b.binary_path, "binary_sha256": engine_b.binary_sha256,
             "build_id": engine_b.build_id}
        )

        cc.write_command_artifacts(
            run_dir,
            argv,
            extra={
                "tournament_id": tournament.id,
                "pair_index": pair.pair_index,
                "attempt": pair.attempt,
                "engine_a": {
                    "build_id": engine_a.build_id,
                    "binary_sha256": engine_a.binary_sha256,
                },
                "engine_b": {
                    "build_id": engine_b.build_id,
                    "binary_sha256": engine_b.binary_sha256,
                },
                "time_control": tournament.time_control,
                "hash_mb": self.settings.hash_mb,
            },
        )

        self.active_proc = cc.launch_cutechess(argv, run_dir)
        self.active_tournament_id = tournament.id
        self.active_pair_job_id = pair.id
        self.active_run_dir = run_dir

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------
    def shutdown(self) -> None:
        """Stop accepting new pairs and terminate the active process group.

        The active attempt is marked INTERRUPTED (not scored); the next worker
        boot re-runs it from scratch via recovery (P1.2).
        """
        if self.active_proc is None:
            return
        with self.session_factory() as session:
            pair = session.get(PairJob, self.active_pair_job_id)
            if pair is not None and pair.status == RUNNING:
                cc.terminate_process_group(
                    self.active_proc, self.settings.shutdown_grace_seconds
                )
                self._close_output_handles(self.active_proc)
                pair.status = INTERRUPTED
                pair.finished_at = utcnow()
                pair.failure_reason = "worker shutdown"
                _record_event(
                    session,
                    pair.tournament_id,
                    "pair_interrupted",
                    pair_job_id=pair.id,
                    reason="worker shutdown",
                )
                session.commit()
            else:
                cc.terminate_process_group(
                    self.active_proc, self.settings.shutdown_grace_seconds
                )
                self._close_output_handles(self.active_proc)
        self._clear_active()

    def current_activity(self) -> Dict[str, Any]:
        return {
            "tournament_id": self.active_tournament_id,
            "pair_job_id": self.active_pair_job_id,
            "pid": self.active_proc.pid if self.active_proc else None,
        }


def _opening_fen_for_index(opening_set, opening_index: int) -> str:
    lines = [
        ln.strip()
        for ln in Path(opening_set.file_path).read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
    if opening_index >= len(lines):
        raise cc.CutechessLaunchError(
            f"opening_index {opening_index} out of range ({len(lines)} lines)"
        )
    return lines[opening_index].split(";")[0].strip()
