"""FastAPI application factory for the arena.

The app listens on 127.0.0.1:8787 behind Nginx; every route is prefixed with
the configured base path (``/chessarena``) so the reverse proxy can pass the
URI through unchanged.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .api import builds, health, openings, tournaments
from .config import Settings, get_settings
from .db import bind_session_factory, make_engine, make_session_factory
from .services import artifacts

_PACKAGE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = _PACKAGE_DIR / "templates"
STATIC_DIR = _PACKAGE_DIR / "static"


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    engine = make_engine(settings.db_url)
    session_factory = make_session_factory(engine)
    bind_session_factory(session_factory)

    # Resolve artifact paths through the settings that were actually used to
    # build this app instance (tests inject their own tmp run root).
    artifacts.configure_artifact_service(settings)

    app = FastAPI(title="ChessArena", version="0.1.0")

    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    app.state.settings = settings
    app.state.templates = templates
    app.state.session_factory = session_factory

    # P2.4: per-browser CSRF cookie (Secure only when the public URL is HTTPS).
    from .security import CsrfCookieMiddleware

    app.add_middleware(
        CsrfCookieMiddleware, secure=settings.public_url.startswith("https")
    )

    bp = settings.base_path
    api_prefix = f"{bp}/api/v1"

    app.include_router(health.router, prefix=api_prefix)
    app.include_router(builds.router, prefix=api_prefix)
    app.include_router(openings.router, prefix=api_prefix)
    app.include_router(tournaments.router, prefix=api_prefix)
    app.include_router(tournaments.admin_router, prefix=bp)

    app.mount(f"{bp}/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get(bp, include_in_schema=False)
    @app.get(f"{bp}/", include_in_schema=False)
    def arena_root():
        return RedirectResponse(url=f"{bp}/admin/")

    return app
