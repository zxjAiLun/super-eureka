"""Request security helpers (P2.4).

CSRF protection uses the double-submit-cookie pattern:
- each browser gets its own random ``arena_csrf`` cookie (set by
  ``CsrfCookieMiddleware`` with SameSite=Lax, HttpOnly, Secure behind HTTPS),
- every admin form embeds that random token as a hidden ``_csrf_token`` field,
- validation compares the submitted token against the cookie value, so a
  token stolen from one browser cannot be replayed with another browser's
  session, and there is no single global token shared by all clients.

State-changing API requests additionally reject cross-site Origin/Referer
headers; clients that send no Origin (curl, scripts, tests) are allowed.

The cookie middleware must be installed by the application factory.
"""

from __future__ import annotations

import secrets
from urllib.parse import urlparse

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

CSRF_COOKIE = "arena_csrf"


class CsrfCookieMiddleware(BaseHTTPMiddleware):
    """Ensure every client carries its own random CSRF cookie."""

    def __init__(self, app, *, secure: bool):
        super().__init__(app)
        self._secure = secure

    async def dispatch(self, request: Request, call_next):
        token = request.cookies.get(CSRF_COOKIE)
        if token is None:
            token = secrets.token_hex(32)
            request.state.csrf_token = token
            response: Response = await call_next(request)
            response.set_cookie(
                CSRF_COOKIE,
                token,
                max_age=60 * 60 * 8,
                path="/",
                httponly=True,
                samesite="lax",
                secure=self._secure,
            )
            return response
        request.state.csrf_token = token
        return await call_next(request)


def _origin_of(request: Request) -> str | None:
    origin = request.headers.get("origin")
    if origin:
        return origin
    referer = request.headers.get("referer")
    if referer:
        return referer
    return None


def require_same_origin(request: Request) -> None:
    """Reject state-changing requests that originate from another site."""
    settings = request.app.state.settings
    candidate = _origin_of(request)
    if candidate is None:
        return  # non-browser client; nothing to cross-check
    expected = urlparse(settings.public_url)
    actual = urlparse(candidate)
    if (actual.scheme, actual.netloc) != (expected.scheme, expected.netloc):
        raise HTTPException(
            status_code=403,
            detail="cross-origin request rejected",
        )


def validate_csrf_token(request: Request, form_fields: dict) -> None:
    """Validate the hidden ``_csrf_token`` field against this browser's cookie.

    The token is per-browser (set by CsrfCookieMiddleware); a global HMAC of
    the server secret is not accepted here.
    """
    expected = getattr(request.state, "csrf_token", None)
    token = form_fields.get("_csrf_token")
    if not expected or not token or token != expected:
        raise HTTPException(status_code=403, detail="invalid CSRF token")
