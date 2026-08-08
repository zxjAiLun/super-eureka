"""Tournament API and admin pages (sections 16.4-16.6, 17).

Creation and all POST actions validate every reference through the database;
nothing accepts raw paths or arbitrary cutechess parameters.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
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
    EngineBuild,
    EnginePreset,
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
from ..security import require_same_origin, validate_csrf_token
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


def _get_enabled_preset_or_422(session, preset_id, label):
    """Resolve an enabled EnginePreset and its EngineBuild (422 on any miss)."""
    preset = (
        session.query(EnginePreset)
        .filter(
            EnginePreset.preset_id == preset_id,
            EnginePreset.enabled.is_(True),
        )
        .first()
    )
    if preset is None:
        raise HTTPException(
            status_code=422,
            detail=f"{label}: unknown or disabled engine preset '{preset_id}'",
        )
    build = (
        session.query(EngineBuild)
        .filter(
            EngineBuild.build_id == preset.build_id,
            EngineBuild.enabled.is_(True),
        )
        .first()
    )
    if build is None:
        raise HTTPException(
            status_code=422,
            detail=f"{label}: preset '{preset_id}' references missing build "
            f"'{preset.build_id}'",
        )
    return preset, build


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
        "engine_a_preset_id": tournament.engine_a_preset_id,
        "engine_b_preset_id": tournament.engine_b_preset_id,
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
        "force_cancel_requested": tournament.force_cancel_requested,
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
@router.post("/tournaments", response_model=TournamentOut, status_code=201,
    dependencies=[Depends(require_same_origin)],
)
def create_tournament(
    body: TournamentCreate,
    session: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    preset_a, build_a = _get_enabled_preset_or_422(
        session, body.engine_a.preset_id, "engine_a"
    )
    preset_b, build_b = _get_enabled_preset_or_422(
        session, body.engine_b.preset_id, "engine_b"
    )
    if (
        body.engine_a.preset_id == body.engine_b.preset_id
        and not body.allow_intentional_self_play
    ):
        raise HTTPException(
            status_code=422,
            detail=(
                "both sides use the same engine preset; set "
                "allow_intentional_self_play to run it deliberately"
            ),
        )

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

    def _snapshot_engine(preset, build) -> dict:
        args = list(preset.command_args or [])
        if len(args) >= 2:
            # Project engines carry "--profile <profile>".
            profile = args[1]
        else:
            # External engines (e.g. Stockfish) have no internal profile; the
            # historical audit column records the preset id instead.
            profile = f"preset:{preset.preset_id}"
        return {
            "preset_id": preset.preset_id,
            "display_name": preset.display_name,
            "build_id": build.build_id,
            "profile": profile,
            "command_args": args,
            "uci_options": dict(preset.uci_options or {}),
            # Freeze the engine's capability schema into the tournament
            # snapshot so command construction and verifier rebuild are
            # immune to later live backfill/re-probe of the EngineBuild row.
            "uci_options_schema": build.uci_options_schema or {},
            "git_sha": build.git_sha,
            "binary_sha256": build.binary_sha256,
        }

    config_snapshot = {
        "engine_a": _snapshot_engine(preset_a, build_a),
        "engine_b": _snapshot_engine(preset_b, build_b),
        "opening_set": {
            "opening_set_id": opening.opening_set_id,
            "sha256": opening.sha256,
        },
        "time_control": body.time_control,
        "hash_mb": settings.hash_mb,
        "threads": settings.threads,
        "concurrency": settings.max_concurrency,
        "requested_pairs": body.pairs,
    }

    tournament = Tournament(
        name=body.name,
        status=DRAFT,
        engine_a_build_id=build_a.build_id,
        engine_a_profile=config_snapshot["engine_a"]["profile"],
        engine_b_build_id=build_b.build_id,
        engine_b_profile=config_snapshot["engine_b"]["profile"],
        engine_a_preset_id=preset_a.preset_id,
        engine_b_preset_id=preset_b.preset_id,
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
#
# Each action is an atomic conditional UPDATE: it only applies when the
# tournament is in the expected status at write time.  Combined with the
# worker's conditional COMPLETED transition this closes the cancel/force-cancel
# race - whichever side acquires the SQLite write lock first wins, and the
# loser's WHERE clause fails (the end state is never COMPLETED with a pending
# cancel flag).
# ---------------------------------------------------------------------------
def _conditional_update(session, tournament_id: str, from_statuses, values):
    from sqlalchemy import update

    return session.execute(
        update(Tournament)
        .where(Tournament.id == tournament_id)
        .where(Tournament.status.in_(list(from_statuses)))
        .values(**values)
    ).rowcount


def _reload_tournament(session, tournament_id) -> Tournament:
    session.expire_all()
    tournament = session.get(Tournament, tournament_id)
    if tournament is None:
        raise HTTPException(status_code=404, detail="tournament not found")
    return tournament


@router.post("/tournaments/{tournament_id}/start", response_model=TournamentOut,
    dependencies=[Depends(require_same_origin)],
)
def start_tournament(tournament_id: str, session: Session = Depends(get_db)):
    if _conditional_update(
        session, tournament_id, {DRAFT}, {"status": QUEUED}
    ) != 1:
        tournament = _reload_tournament(session, tournament_id)
        raise HTTPException(
            status_code=409,
            detail=f"cannot start tournament in status '{tournament.status}'",
        )
    tournament = _reload_tournament(session, tournament_id)
    _record_event(session, tournament.id, "tournament_started")
    return _to_out(tournament)


@router.post("/tournaments/{tournament_id}/pause", response_model=TournamentOut,
    dependencies=[Depends(require_same_origin)],
)
def pause_tournament(tournament_id: str, session: Session = Depends(get_db)):
    if _conditional_update(
        session, tournament_id, {RUNNING},
        {"status": PAUSING, "pause_requested": True},
    ) != 1:
        tournament = _reload_tournament(session, tournament_id)
        raise HTTPException(
            status_code=409,
            detail=f"cannot pause tournament in status '{tournament.status}'",
        )
    # The worker emits tournament_paused when the pause actually takes effect
    # (after the current pair completes).
    tournament = _reload_tournament(session, tournament_id)
    return _to_out(tournament)


@router.post("/tournaments/{tournament_id}/resume", response_model=TournamentOut,
    dependencies=[Depends(require_same_origin)],
)
def resume_tournament(tournament_id: str, session: Session = Depends(get_db)):
    if _conditional_update(
        session, tournament_id, {PAUSED},
        {"status": QUEUED, "pause_requested": False},
    ) != 1:
        tournament = _reload_tournament(session, tournament_id)
        raise HTTPException(
            status_code=409,
            detail=f"cannot resume tournament in status '{tournament.status}'",
        )
    tournament = _reload_tournament(session, tournament_id)
    _record_event(session, tournament.id, "tournament_resumed")
    return _to_out(tournament)


@router.post("/tournaments/{tournament_id}/cancel", response_model=TournamentOut,
    dependencies=[Depends(require_same_origin)],
)
def cancel_tournament(tournament_id: str, session: Session = Depends(get_db)):
    # QUEUED/PAUSED -> cancelled immediately (atomic); RUNNING/PAUSING ->
    # cancel_requested flag set atomically for the worker to apply.
    if _conditional_update(
        session, tournament_id, {QUEUED, PAUSED},
        {"status": CANCELLED, "cancel_requested": True, "finished_at": utcnow()},
    ) == 1:
        tournament = _reload_tournament(session, tournament_id)
        _record_event(session, tournament.id, "tournament_cancelled", reason="requested")
        return _to_out(tournament)
    if _conditional_update(
        session, tournament_id, {RUNNING, PAUSING}, {"cancel_requested": True}
    ) == 1:
        tournament = _reload_tournament(session, tournament_id)
        _record_event(
            session, tournament.id, "tournament_cancelled", reason="requested"
        )
        return _to_out(tournament)
    tournament = _reload_tournament(session, tournament_id)
    raise HTTPException(
        status_code=409,
        detail=f"cannot cancel tournament in status '{tournament.status}'",
    )


@router.post("/tournaments/{tournament_id}/force-cancel", response_model=TournamentOut,
    dependencies=[Depends(require_same_origin)],
)
def force_cancel_tournament(
    tournament_id: str,
    confirm: bool = Query(default=False),
    session: Session = Depends(get_db),
):
    """Request an immediate cancellation that kills the running process group.

    The API only sets ``force_cancel_requested`` in the database (atomically);
    the worker polls it, kills the cutechess process group, and then marks the
    pair INTERRUPTED and the tournament CANCELLED (P1.3).  The response
    reflects the state before the worker acts.
    """
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="force-cancel requires confirm=true",
        )
    if _conditional_update(
        session, tournament_id, {RUNNING, PAUSING, QUEUED, PAUSED},
        {"force_cancel_requested": True},
    ) != 1:
        tournament = _reload_tournament(session, tournament_id)
        raise HTTPException(
            status_code=409,
            detail=f"cannot force-cancel tournament in status '{tournament.status}'",
        )
    tournament = _reload_tournament(session, tournament_id)
    _record_event(
        session, tournament.id, "tournament_cancelled", reason="force-requested"
    )
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
    # _tournament_status.html is included whenever a match is active and
    # requires the full tournament/pairs/score_percent contract (it uses
    # {{ tournament.* }} and iterates {{ pairs }}), so pass them explicitly.
    active_pairs = (
        sorted(active.pair_jobs, key=lambda p: p.pair_index)
        if active is not None
        else []
    )
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "worker_online": worker_online,
            "worker": worker,
            "active": active,
            "tournament": active,
            "pairs": active_pairs,
            "score_percent": _score_percent(active) if active else None,
            "recent": recent,
            "settings": settings,
        },
    )


@admin_router.get("/admin/tournaments/new", response_class=HTMLResponse)
def admin_tournament_new(request: Request, session: Session = Depends(get_db)):
    templates = request.app.state.templates
    presets = (
        session.query(EnginePreset)
        .filter(EnginePreset.enabled.is_(True))
        .order_by(EnginePreset.category, EnginePreset.created_at.desc())
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
            "presets": presets,
            "openings": openings,
            "time_controls": TIME_CONTROLS,
            "settings": request.app.state.settings,
        },
    )


@admin_router.post("/admin/tournaments", response_class=RedirectResponse)
async def admin_tournament_create(request: Request, session: Session = Depends(get_db)):
    form = dict(await request.form())
    validate_csrf_token(request, form)
    body = TournamentCreate(
        name=form["name"],
        engine_a={"preset_id": form["engine_a_preset"]},
        engine_b={"preset_id": form["engine_b_preset"]},
        opening_set_id=form["opening_set_id"],
        time_control=form["time_control"],
        pairs=int(form["pairs"]),
        allow_intentional_self_play=form.get("allow_intentional_self_play") == "on",
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

    from chessarena.services.runtime_status import derive_runtime_status

    runtime: dict[str, dict] = {}
    for p in pairs:
        if p.run_directory and Path(p.run_directory).is_dir():
            runtime[p.id] = derive_runtime_status(
                Path(p.run_directory), total_games=2
            )
    return templates.TemplateResponse(
        request,
        "tournament_detail.html",
        {
            "tournament": tournament,
            "pairs": pairs,
            "runtime": runtime,
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
async def admin_tournament_action(
    request: Request,
    tournament_id: str,
    action: str,
    session: Session = Depends(get_db),
):
    form = dict(await request.form())
    validate_csrf_token(request, form)
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


@admin_router.post("/admin/tournaments/{tournament_id}/action/force-cancel",
                   response_class=RedirectResponse)
async def admin_tournament_force_cancel(
    request: Request,
    tournament_id: str,
    session: Session = Depends(get_db),
):
    """Force-cancel from the admin UI: requires the CSRF token AND an explicit
    ``confirm`` form field so it can never be triggered by a plain cancel."""
    form = dict(await request.form())
    validate_csrf_token(request, form)
    if form.get("confirm") != "1":
        raise HTTPException(
            status_code=400,
            detail="force-cancel requires explicit confirmation",
        )
    force_cancel_tournament(
        tournament_id, confirm=True, session=session
    )
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


@admin_router.get("/admin/tournaments/{tournament_id}/pairs",
                  response_class=HTMLResponse)
def admin_tournament_pairs_fragment(
    request: Request, tournament_id: str, session: Session = Depends(get_db)
):
    """HTMX pair-progress fragment (2s refresh) including live per-game
    runtime status derived from cutechess stdout (P4.F1 A3)."""
    templates = request.app.state.templates
    tournament = _get_tournament_or_404(session, tournament_id)
    pairs = sorted(tournament.pair_jobs, key=lambda p: p.pair_index)

    from chessarena.services import artifacts
    from chessarena.services.runtime_status import derive_runtime_status

    runtime: dict[str, dict] = {}
    for p in pairs:
        if p.run_directory and Path(p.run_directory).is_dir():
            runtime[p.id] = derive_runtime_status(
                Path(p.run_directory), total_games=2
            )
    return templates.TemplateResponse(
        request,
        "_pair_progress.html",
        {
            "tournament": tournament,
            "pairs": pairs,
            "runtime": runtime,
            "settings": request.app.state.settings,
        },
    )
