"""Request security helpers (P2.4).

- ``require_same_origin``: rejects state-changing API requests that carry a
  cross-site Origin/Referer.  Browsers send Origin on every POST; a mismatched
  one is a CSRF attempt.  Clients that send no Origin (curl, tests, scripts)
  are allowed.
- CSRF token validation for the Jinja2 admin forms lives directly in the admin
  handlers (the token is submitted as a hidden form field), because a FastAPI
  dependency cannot safely read the request body that a form handler needs.

The token derives from ``settings.csrf_secret`` and is embedded in every admin
form; cross-origin pages cannot read it, so this is a synchronizer token.
"""

from __future__ import annotations

from urllib.parse import urlparse

from fastapi import HTTPException, Request


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
    """Validate the hidden ``_csrf_token`` form field against the app secret."""
    settings = request.app.state.settings
    token = form_fields.get("_csrf_token")
    if not token or token != settings.csrf_token:
        raise HTTPException(status_code=403, detail="invalid CSRF token")
