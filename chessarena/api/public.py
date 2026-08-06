"""Public, read-only replay API (anonymous).

Only COMPLETED tournaments and their verified games are exposed, with a
whitelist of display fields.  No build ids, binary SHAs, server paths,
commands, logs, manifests or provenance are ever returned.  Write endpoints
stay behind the authenticated ``/api/v1`` tree (enforced by nginx).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import PlainTextResponse
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import COMPLETED, EngineBuild, EnginePreset, Game, Tournament
from ..schemas import PublicGameOut, PublicMatchDetailOut, PublicMatchOut
from ..services.replay import ReplayError, read_single_game_pgn

router = APIRouter(tags=["public-replay"])
pages_router = APIRouter(include_in_schema=False, tags=["public-pages"])


def _engine_label(
    session: Session, preset_id: str | None, build_id: str, profile: str
) -> str:
    """Display label for one side.  Prefers the preset's friendly display name;
    falls back to the engine name + profile.  build_id is never exposed."""
    if preset_id:
        preset = (
            session.query(EnginePreset)
            .filter(EnginePreset.preset_id == preset_id)
            .first()
        )
        if preset is not None:
            return preset.display_name
    name = "ChessEngine"
    if build_id:
        build = (
            session.query(EngineBuild)
            .filter(EngineBuild.build_id == build_id)
            .first()
        )
        if build is not None:
            name = build.engine_name
    return f"{name} ({profile})" if profile else name


def _score_percent(t: Tournament) -> float | None:
    played = t.candidate_wins + t.candidate_losses + t.draws
    if not played:
        return None
    return round((t.candidate_wins + 0.5 * t.draws) / played * 100, 2)


def _public_match(session: Session, t: Tournament) -> PublicMatchOut:
    return PublicMatchOut(
        id=t.id,
        name=t.name,
        status=t.status,
        time_control=t.time_control,
        requested_pairs=t.requested_pairs,
        completed_pairs=t.completed_pairs,
        candidate_wins=t.candidate_wins,
        candidate_losses=t.candidate_losses,
        draws=t.draws,
        score_percent=_score_percent(t),
        finished_at=t.finished_at,
        engine_a_label=_engine_label(
            session, t.engine_a_preset_id, t.engine_a_build_id, t.engine_a_profile
        ),
        engine_b_label=_engine_label(
            session, t.engine_b_preset_id, t.engine_b_build_id, t.engine_b_profile
        ),
        opening_set_id=t.opening_set_id,
    )


@router.get("/matches", response_model=list[PublicMatchOut])
def list_public_matches(
    limit: int = 50,
    session: Session = Depends(get_db),
):
    """Public list of COMPLETED tournaments (newest first)."""
    limit = max(1, min(limit, 200))
    rows = (
        session.query(Tournament)
        .filter(Tournament.status == COMPLETED)
        .order_by(Tournament.finished_at.desc())
        .limit(limit)
        .all()
    )
    return [_public_match(session, t) for t in rows]


@router.get("/matches/{tournament_id}", response_model=PublicMatchDetailOut)
def get_public_match(
    tournament_id: str,
    session: Session = Depends(get_db),
):
    """Public detail of a COMPLETED tournament plus its verified games."""
    t = (
        session.query(Tournament)
        .filter(Tournament.id == tournament_id)
        .first()
    )
    if t is None or t.status != COMPLETED:
        raise HTTPException(status_code=404, detail="match not found")
    games = (
        session.query(Game)
        .filter(Game.tournament_id == t.id, Game.verified.is_(True))
        .order_by(Game.game_number)
        .all()
    )
    match = _public_match(session, t)
    return PublicMatchDetailOut(
        **match.model_dump(),
        games=[PublicGameOut.model_validate(g) for g in games],
    )


@router.get("/games/{game_id}/pgn")
def get_public_game_pgn(
    game_id: str,
    session: Session = Depends(get_db),
):
    """Single verified game's PGN text (anonymous replay)."""
    game = session.query(Game).filter(Game.id == game_id).first()
    if game is None or not game.verified:
        raise HTTPException(status_code=404, detail="game not found")
    tournament = (
        session.query(Tournament)
        .filter(Tournament.id == game.tournament_id)
        .first()
    )
    if tournament is None or tournament.status != COMPLETED:
        raise HTTPException(status_code=404, detail="game not found")
    try:
        pgn = read_single_game_pgn(game)
    except ReplayError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    filename = f"game-{game.game_number}.pgn"
    return PlainTextResponse(
        pgn,
        media_type="application/x-chess-pgn",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ---------------------------------------------------------------------------
# Public HTML pages
# ---------------------------------------------------------------------------
def _render(request: Request, template: str, **ctx):
    return request.app.state.templates.TemplateResponse(
        request, template, ctx
    )


def _recent_matches(session: Session, limit: int = 12):
    rows = (
        session.query(Tournament)
        .filter(Tournament.status == COMPLETED)
        .order_by(Tournament.finished_at.desc())
        .limit(limit)
        .all()
    )
    return [_public_match(session, t) for t in rows]


@pages_router.get("/")
def public_home(request: Request, session: Session = Depends(get_db)):
    return _render(
        request,
        "public_home.html",
        settings=request.app.state.settings,
        matches=_recent_matches(session, limit=12),
    )


@pages_router.get("/matches/")
def public_matches(request: Request, session: Session = Depends(get_db)):
    return _render(
        request,
        "public_matches.html",
        settings=request.app.state.settings,
        matches=_recent_matches(session, limit=200),
    )


@pages_router.get("/matches/{tournament_id}")
def public_match_detail(
    tournament_id: str,
    request: Request,
    session: Session = Depends(get_db),
):
    t = (
        session.query(Tournament)
        .filter(Tournament.id == tournament_id)
        .first()
    )
    if t is None or t.status != COMPLETED:
        raise HTTPException(status_code=404, detail="match not found")
    games = (
        session.query(Game)
        .filter(Game.tournament_id == t.id, Game.verified.is_(True))
        .order_by(Game.game_number)
        .all()
    )
    match = _public_match(session, t)
    detail = PublicMatchDetailOut(
        **match.model_dump(),
        games=[PublicGameOut.model_validate(g) for g in games],
    )
    return _render(
        request,
        "public_match_detail.html",
        settings=request.app.state.settings,
        match=detail,
    )


@pages_router.get("/games/{game_id}")
def public_game(
    game_id: str,
    request: Request,
    session: Session = Depends(get_db),
):
    game = session.query(Game).filter(Game.id == game_id).first()
    if game is None or not game.verified:
        raise HTTPException(status_code=404, detail="game not found")
    tournament = (
        session.query(Tournament)
        .filter(Tournament.id == game.tournament_id)
        .first()
    )
    if tournament is None or tournament.status != COMPLETED:
        raise HTTPException(status_code=404, detail="game not found")
    try:
        pgn_text = read_single_game_pgn(game)
    except ReplayError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _render(
        request,
        "public_game.html",
        settings=request.app.state.settings,
        game=game,
        pgn_text=pgn_text,
    )
