"""Tournament API and admin pages (sections 16.4-16.6, 17).

Creation and all POST actions validate every reference through the database;
nothing accepts raw paths or arbitrary cutechess parameters.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    RedirectResponse,
    Response,
)
from sqlalchemy.orm import Session

from ..config import ENGINE_A_NAME, ENGINE_B_NAME, TIME_CONTROLS, Settings, get_settings
from ..db import get_db
from ..models import (
    CANCELLED,
    COMPLETED,
    DRAFT,
    FAILED,
    PAUSED,
    PAUSING,
    PENDING,
    QUEUED,
    RUNNING,
    TOURNAMENT_TRANSITIONS,
    EngineBuild,
    Event,
    Game,
    OpeningSet,
    PairJob,
    Tournament,
    WorkerState,
    coerce_utc,
    utcnow,
)
from ..schemas import (
    EventOut,
    GameOut,
    PairJobOut,
    TournamentCreate,
    TournamentDetailOut,
    TournamentOut,
)
from ..services import artifacts

router = APIRouter(tags=["tournaments"])
admin_router = APIRouter(tags=["admin"], include_in_schema=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _record_event(session, tournament_id, event_type, pair_job_id=None,
                  game_id=None, **payload) -> Event:
    event = Event(
        tournament_id=tournament_id,
        pair_job_id=pair_job_id,
        game_id=game_id,
        event_type=event_type,
        payload=dict(payload),
    )
    session.add(event)
    return event


def _get_tournament_or_404(session, tournament_id) -> Tournament:
    tournament = session.get(Tournament, tournament_id)
    if tournament is None:
        raise HTTPException(status_code=404, detail="tournament not found")
    return tournament


def _get_enabled_build_or_422(session, build_id, label) -> EngineBuild:
    build = (
        session.query(EngineBuild)
        .filter(EngineBuild.build_id == build_id, EngineBuild.enabled.is_(True))
        .first()
    )
    if build is None:
        raise HTTPException(
            status_code=422,
            detail=f"{label}: unknown or disabled build '{build_id}'",
        )
    return build


def _get_enabled_opening_or_422(session, opening_set_id) -> OpeningSet:
    opening = (
        session.query(OpeningSet)
        .filter(
            OpeningSet.opening_set_id == opening_set_id,
            OpeningSet.enabled.is_(True),
        )
        .first()
    )
    if opening is None:
        raise HTTPException(
            status_code=422,
            detail=f"unknown or disabled opening set '{opening_set_id}'",
        )
    return opening


def _validate_profile_or_422(build: EngineBuild, profile: str, label: str) -> None:
    if profile not in build.supported_profiles:
        raise HTTPException(
            status_code=422,
            detail=(
                f"{label}: profile '{profile}' not in build "
                f"{build.build_id} supported_profiles {build.supported_profiles}"
            ),
        )


def _require_transition(session, tournament: Tournament, new_status: str,
                        action: str, from_statuses: set[str] | None = None) -> None:
    allowed = TOURNAMENT_TRANSITIONS.get(tournament.status, set())
    if new_status not in allowed:
        raise HTTPException(
            status_code=409,
            detail=(
                f"cannot {action} tournament in status '{tournament.status}'"
                f" (allowed next states: {sorted(allowed)})"
            ),
        )
    if from_statuses is not None and tournament.status not in from_statuses:
        raise HTTPException(
            status_code=409,
            detail=(
                f"cannot {action} tournament from status '{tournament.status}'"
                f" (expected one of {sorted(from_statuses)})"
            ),
        )


def _score_percent(tournament: Tournament) -> Optional[float]:
    played = tournament.candidate_wins + tournament.candidate_losses + tournament.draws
    if played == 0:
        return None
    return round(
        (tournament.candidate_wins + 0.5 * tournament.draws) / played * 100, 2
    )


def _to_out(tournament: Tournament, detail: bool = False):
    data = {
        "id": tournament.id,
        "name": tournament.name,
        "status": tournament.status,
        "engine_a_build_id": tournament.engine_a_build_id,
        "engine_a_profile": tournament.engine_a_profile,
        "engine_b_build_id": tournament.engine_b_build_id,
        "engine_b_profile": tournament.engine_b_profile,
        "opening_set_id": tournament.opening_set_id,
        "time_control": tournament.time_control,
        "requested_pairs": tournament.requested_pairs,
        "completed_pairs": tournament.completed_pairs,
        "candidate_wins": tournament.candidate_wins,
        "candidate_losses": tournament.candidate_losses,
        "draws": tournament.draws,
        "score_percent": _score_percent(tournament),
        "created_at": tournament.created_at,
        "started_at": tournament.started_at,
        "finished_at": tournament.finished_at,
        "failure_reason": tournament.failure_reason,
        "config_snapshot": tournament.config_snapshot,
        "pause_requested": tournament.pause_requested,
        "cancel_requested": tournament.cancel_requested,
    }
    if detail:
        data["pairs"] = sorted(
            tournament.pair_jobs, key=lambda p: p.pair_index
        )
    return data


# ---------------------------------------------------------------------------
# Creation (section 16.4)
# ---------------------------------------------------------------------------
@router.post("/tournaments", response_model=TournamentOut, status_code=201)
def create_tournament(
    body: TournamentCreate,
    session: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    build_a = _get_enabled_build_or_422(session, body.engine_a.build_id, "engine_a")
    _validate_profile_or_422(build_a, body.engine_a.profile, "engine_a")
    build_b = _get_enabled_build_or_422(session, body.engine_b.build_id, "engine_b")
    _validate_profile_or_422(build_b, body.engine_b.profile, "engine_b")

    opening = _get_enabled_opening_or_422(session, body.opening_set_id)
    if body.time_control not in TIME_CONTROLS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"time_control must be one of {sorted(TIME_CONTROLS)}"
            ),
        )
    if body.pairs > opening.position_count:
        raise HTTPException(
            status_code=422,
            detail=(
                f"pairs {body.pairs} exceeds opening set capacity "
                f"{opening.position_count}"
            ),
        )

    config_snapshot = {
        "engine_a": {
            "build_id": build_a.build_id,
            "profile": body.engine_a.profile,
            "git_sha": build_a.git_sha,
            "binary_sha256": build_a.binary_sha256,
        },
        "engine_b": {
            "build_id": build_b.build_id,
            "profile": body.engine_b.profile,
            "git_sha": build_b.git_sha,
            "binary_sha256": build_b.binary_sha256,
        },
        "opening_set": {
            "opening_set_id": opening.opening_set_id,
            "sha256": opening.sha256,
        },
        "time_control": body.time_control,
        "hash_mb": settings.hash_mb,
        "concurrency": settings.max_concurrency,
        "requested_pairs": body.pairs,
    }

    tournament = Tournament(
        name=body.name,
        status=DRAFT,
        engine_a_build_id=build_a.build_id,
        engine_a_profile=body.engine_a.profile,
        engine_b_build_id=build_b.build_id,
        engine_b_profile=body.engine_b.profile,
        opening_set_id=opening.opening_set_id,
        time_control=body.time_control,
        requested_pairs=body.pairs,
        config_snapshot=config_snapshot,
    )
    session.add(tournament)
    session.flush()  # obtain tournament.id

    for pair_index in range(body.pairs):
        session.add(
            PairJob(
                tournament_id=tournament.id,
                pair_index=pair_index,
                opening_index=pair_index,
                status=PENDING,
                attempt=1,
            )
        )
    _record_event(
        session,
        tournament.id,
        "tournament_created",
        name=tournament.name,
        requested_pairs=body.pairs,
        time_control=body.time_control,
    )
    session.flush()
    return _to_out(tournament)


# ---------------------------------------------------------------------------
# Lifecycle actions (sections 16.5, 10.1)
# ---------------------------------------------------------------------------
@router.post("/tournaments/{tournament_id}/start", response_model=TournamentOut)
def start_tournament(tournament_id: str, session: Session = Depends(get_db)):
    tournament = _get_tournament_or_404(session, tournament_id)
    _require_transition(session, tournament, QUEUED, "start", from_statuses={DRAFT})
    tournament.status = QUEUED
    _record_event(session, tournament.id, "tournament_started")
    return _to_out(tournament)


@router.post("/tournaments/{tournament_id}/pause", response_model=TournamentOut)
def pause_tournament(tournament_id: str, session: Session = Depends(get_db)):
    tournament = _get_tournament_or_404(session, tournament_id)
    _require_transition(session, tournament, PAUSING, "pause")
    tournament.status = PAUSING
    tournament.pause_requested = True
    # The worker emits tournament_paused when the pause actually takes effect
    # (after the current pair completes).
    return _to_out(tournament)


@router.post("/tournaments/{tournament_id}/resume", response_model=TournamentOut)
def resume_tournament(tournament_id: str, session: Session = Depends(get_db)):
    tournament = _get_tournament_or_404(session, tournament_id)
    _require_transition(session, tournament, QUEUED, "resume", from_statuses={PAUSED})
    tournament.status = QUEUED
    tournament.pause_requested = False
    _record_event(session, tournament.id, "tournament_resumed")
    return _to_out(tournament)


@router.post("/tournaments/{tournament_id}/cancel", response_model=TournamentOut)
def cancel_tournament(tournament_id: str, session: Session = Depends(get_db)):
    tournament = _get_tournament_or_404(session, tournament_id)
    _require_transition(session, tournament, CANCELLED, "cancel")
    tournament.cancel_requested = True
    _record_event(session, tournament.id, "tournament_cancelled", reason="requested")
    if tournament.status in (QUEUED, PAUSED):
        # Nothing is running; cancel immediately.
        tournament.status = CANCELLED
        tournament.finished_at = utcnow()
    # RUNNING/PAUSING: the worker completes the current pair, then sets
    # CANCELLED.  Status stays RUNNING/PAUSING until then.
    return _to_out(tournament)


@router.post("/tournaments/{tournament_id}/force-cancel", response_model=TournamentOut)
def force_cancel_tournament(
    tournament_id: str,
    confirm: bool = Query(default=False),
    session: Session = Depends(get_db),
):
    """Immediate cancellation that kills the running cutechess process group.

    Requires an explicit ``confirm=true`` so a plain cancel can never trigger
    it by mistake (section 16.5).
    """
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="force-cancel requires confirm=true",
        )
    tournament = _get_tournament_or_404(session, tournament_id)
    if tournament.status not in (RUNNING, PAUSING, QUEUED, PAUSED):
        raise HTTPException(
            status_code=409,
            detail=f"cannot force-cancel tournament in status '{tournament.status}'",
        )
    _record_event(
        session, tournament.id, "tournament_cancelled", reason="force"
    )
    running_pair = (
        session.query(PairJob)
        .filter(
            PairJob.tournament_id == tournament.id,
            PairJob.status == RUNNING,
        )
        .first()
    )
    if running_pair is not None:
        from ..services.scheduler import request_force_kill

        request_force_kill(running_pair.id)
        # The worker performs the actual kill; mark the pair interrupted here
        # so a restart does not try to recover a dead pair as a retry.
        running_pair.status = "INTERRUPTED"
        running_pair.finished_at = utcnow()
        running_pair.failure_reason = "force-cancelled"
        _record_event(
            session,
            tournament.id,
            "pair_failed",
            pair_job_id=running_pair.id,
            reason="force-cancelled",
        )
    tournament.cancel_requested = True
    tournament.status = CANCELLED
    tournament.finished_at = utcnow()
    return _to_out(tournament)


# ---------------------------------------------------------------------------
# Queries (section 16.6)
# ---------------------------------------------------------------------------
@router.get("/tournaments", response_model=list[TournamentOut])
def list_tournaments(
    limit: int = Query(default=50, ge=1, le=200),
    session: Session = Depends(get_db),
):
    rows = (
        session.query(Tournament)
        .order_by(Tournament.created_at.desc())
        .limit(limit)
        .all()
    )
    return [_to_out(t) for t in rows]


@router.get("/tournaments/{tournament_id}", response_model=TournamentDetailOut)
def get_tournament(tournament_id: str, session: Session = Depends(get_db)):
    tournament = _get_tournament_or_404(session, tournament_id)
    return _to_out(tournament, detail=True)


@router.get("/tournaments/{tournament_id}/pairs", response_model=list[PairJobOut])
def get_tournament_pairs(tournament_id: str, session: Session = Depends(get_db)):
    _get_tournament_or_404(session, tournament_id)
    return (
        session.query(PairJob)
        .filter(PairJob.tournament_id == tournament_id)
        .order_by(PairJob.pair_index)
        .all()
    )


@router.get("/tournaments/{tournament_id}/games", response_model=list[GameOut])
def get_tournament_games(tournament_id: str, session: Session = Depends(get_db)):
    _get_tournament_or_404(session, tournament_id)
    return (
        session.query(Game)
        .filter(Game.tournament_id == tournament_id)
        .order_by(Game.game_number)
        .all()
    )


@router.get("/tournaments/{tournament_id}/events", response_model=list[EventOut])
def get_tournament_events(
    tournament_id: str,
    limit: int = Query(default=100, ge=1, le=500),
    session: Session = Depends(get_db),
):
    _get_tournament_or_404(session, tournament_id)
    return (
        session.query(Event)
        .filter(Event.tournament_id == tournament_id)
        .order_by(Event.id.desc())
        .limit(limit)
        .all()
    )


# ---------------------------------------------------------------------------
# Artifact downloads (sections 16.6, 13)
# ---------------------------------------------------------------------------
@router.get("/tournaments/{tournament_id}/pgn")
def download_combined_pgn(tournament_id: str, session: Session = Depends(get_db)):
    tournament = _get_tournament_or_404(session, tournament_id)
    combined = artifacts.tournament_run_dir(tournament.id) / "combined.pgn"
    if not combined.exists():
        raise HTTPException(status_code=404, detail="combined PGN not ready")
    return FileResponse(
        combined,
        media_type="application/x-chess-pgn",
        filename=f"tournament-{tournament.id}.pgn",
    )


@router.get("/tournaments/{tournament_id}/summary")
def download_summary(tournament_id: str, session: Session = Depends(get_db)):
    tournament = _get_tournament_or_404(session, tournament_id)
    summary = artifacts.tournament_run_dir(tournament.id) / "summary.json"
    if not summary.exists():
        raise HTTPException(status_code=404, detail="summary not ready")
    return FileResponse(
        summary,
        media_type="application/json",
        filename=f"tournament-{tournament.id}-summary.json",
    )


@router.get("/tournaments/{tournament_id}/artifacts")
def download_artifact_manifest(tournament_id: str, session: Session = Depends(get_db)):
    tournament = _get_tournament_or_404(session, tournament_id)
    manifest = artifacts.tournament_run_dir(tournament.id) / "artifact-manifest.json"
    if not manifest.exists():
        raise HTTPException(status_code=404, detail="artifact manifest not ready")
    return FileResponse(
        manifest,
        media_type="application/json",
        filename=f"tournament-{tournament.id}-artifact-manifest.json",
    )


@router.get("/tournaments/{tournament_id}/artifacts/raw")
def download_raw_artifact(
    tournament_id: str,
    path: str = Query(...),
    session: Session = Depends(get_db),
):
    """Download a raw pair artifact by relative path (path traversal safe)."""
    tournament = _get_tournament_or_404(session, tournament_id)
    resolved = artifacts.download_path(tournament.id, path)
    if resolved is None or not resolved.is_file():
        raise HTTPException(status_code=404, detail="artifact not found")
    return FileResponse(resolved)


# ---------------------------------------------------------------------------
# Admin pages (section 17)
# ---------------------------------------------------------------------------
def _admin_required():
    # v1 protects everything through Nginx Basic Auth; the app itself is
    # reachable only on 127.0.0.1.
    return None


@admin_router.get("/admin/", response_class=HTMLResponse)
def admin_dashboard(request: Request, session: Session = Depends(get_db)):
    templates = request.app.state.templates
    settings: Settings = request.app.state.settings

    worker = session.get(WorkerState, 1)
    worker_online = (
        worker is not None
        and (datetime.now(timezone.utc) - coerce_utc(worker.heartbeat_at)).total_seconds()
        <= settings.worker_stale_seconds
    )

    active = None
    if worker is not None and worker.tournament_id:
        active = _get_tournament_or_404(session, worker.tournament_id)

    recent = (
        session.query(Tournament).order_by(Tournament.created_at.desc()).limit(20).all()
    )
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "worker_online": worker_online,
            "worker": worker,
            "active": active,
            "recent": recent,
            "settings": settings,
        },
    )


@admin_router.get("/admin/tournaments/new", response_class=HTMLResponse)
def admin_tournament_new(request: Request, session: Session = Depends(get_db)):
    templates = request.app.state.templates
    builds = (
        session.query(EngineBuild)
        .filter(EngineBuild.enabled.is_(True))
        .order_by(EngineBuild.created_at.desc())
        .all()
    )
    openings = (
        session.query(OpeningSet)
        .filter(OpeningSet.enabled.is_(True))
        .order_by(OpeningSet.created_at.desc())
        .all()
    )
    return templates.TemplateResponse(
        request,
        "tournament_new.html",
        {
            "builds": builds,
            "openings": openings,
            "time_controls": TIME_CONTROLS,
            "settings": request.app.state.settings,
        },
    )


@admin_router.post("/admin/tournaments", response_class=RedirectResponse)
def admin_tournament_create(request: Request, session: Session = Depends(get_db)):
    form = dict(request.form())
    body = TournamentCreate(
        name=form["name"],
        engine_a={"build_id": form["engine_a_build"], "profile": form["engine_a_profile"]},
        engine_b={"build_id": form["engine_b_build"], "profile": form["engine_b_profile"]},
        opening_set_id=form["opening_set_id"],
        time_control=form["time_control"],
        pairs=int(form["pairs"]),
    )
    # Reuse the API creation logic by calling it directly.
    created = create_tournament(body, session, request.app.state.settings)
    session.flush()
    return RedirectResponse(
        url=f"{request.app.state.settings.base_path}/admin/tournaments/{created['id']}",
        status_code=303,
    )


@admin_router.get("/admin/tournaments/{tournament_id}", response_class=HTMLResponse)
def admin_tournament_detail(
    request: Request, tournament_id: str, session: Session = Depends(get_db)
):
    templates = request.app.state.templates
    tournament = _get_tournament_or_404(session, tournament_id)
    pairs = sorted(tournament.pair_jobs, key=lambda p: p.pair_index)
    games = (
        session.query(Game)
        .filter(Game.tournament_id == tournament_id)
        .order_by(Game.game_number)
        .all()
    )
    events = (
        session.query(Event)
        .filter(Event.tournament_id == tournament_id)
        .order_by(Event.id.desc())
        .limit(30)
        .all()
    )
    run_dir = artifacts.tournament_run_dir(tournament.id)
    has_combined = (run_dir / "combined.pgn").exists()
    has_summary = (run_dir / "summary.json").exists()
    has_manifest = (run_dir / "artifact-manifest.json").exists()
    return templates.TemplateResponse(
        request,
        "tournament_detail.html",
        {
            "tournament": tournament,
            "pairs": pairs,
            "games": games,
            "events": events,
            "score_percent": _score_percent(tournament),
            "has_combined": has_combined,
            "has_summary": has_summary,
            "has_manifest": has_manifest,
            "settings": request.app.state.settings,
        },
    )


@admin_router.post("/admin/tournaments/{tournament_id}/action/{action}",
                   response_class=RedirectResponse)
def admin_tournament_action(
    request: Request,
    tournament_id: str,
    action: str,
    session: Session = Depends(get_db),
):
    actions = {
        "start": start_tournament,
        "pause": pause_tournament,
        "resume": resume_tournament,
        "cancel": cancel_tournament,
    }
    handler = actions.get(action)
    if handler is None:
        raise HTTPException(status_code=404, detail="unknown action")
    handler(tournament_id, session)
    session.flush()
    return RedirectResponse(
        url=(
            f"{request.app.state.settings.base_path}/admin/tournaments/"
            f"{tournament_id}"
        ),
        status_code=303,
    )


@admin_router.get("/admin/tournaments/{tournament_id}/status",
                  response_class=HTMLResponse)
def admin_tournament_status_fragment(
    request: Request, tournament_id: str, session: Session = Depends(get_db)
):
    """HTMX fragment auto-refreshed every 5 seconds (section 17.1)."""
    templates = request.app.state.templates
    tournament = _get_tournament_or_404(session, tournament_id)
    pairs = sorted(tournament.pair_jobs, key=lambda p: p.pair_index)
    return templates.TemplateResponse(
        request,
        "_tournament_status.html",
        {
            "tournament": tournament,
            "pairs": pairs,
            "score_percent": _score_percent(tournament),
            "settings": request.app.state.settings,
        },
    )
