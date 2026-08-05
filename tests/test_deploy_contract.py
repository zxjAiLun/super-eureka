"""Deploy-artifact contract test.

Regression: the API unit must start the application through its factory
(chessarena.main:create_app --factory).  The API module exposes create_app(),
not a module-level app, so the obsolete chessarena.main:app target crashes
the service at bootstrap (uvicorn: Attribute "app" not found).
"""

from __future__ import annotations

from pathlib import Path

API_UNIT = (
    Path(__file__).resolve().parents[1] / "deploy" / "chessarena-api.service"
)


def test_api_unit_starts_through_the_application_factory():
    assert API_UNIT.is_file(), f"missing {API_UNIT}"
    content = API_UNIT.read_text(encoding="utf-8")
    exec_line = next(
        line for line in content.splitlines() if line.startswith("ExecStart=")
    )

    assert "chessarena.main:create_app" in exec_line
    assert "--factory" in exec_line
    assert "chessarena.main:app" not in exec_line
