"""Regression: the deployment verification database probe must be executable
under SQLAlchemy 2.x.

verify_install.py probes the database with a bare ``session.execute("SELECT 1")``;
SQLAlchemy 2.x rejects textual SQL that is not wrapped in ``text()``, so the
probe fails with "Textual SQL expression ... should be explicitly declared as
text(...)" against the declared ``sqlalchemy>=2.0.30`` dependency.  This test
runs the real probe against a real SQLite engine to lock in the fix.
"""

from __future__ import annotations

import sys
from pathlib import Path

from chessarena.db import make_engine, make_session_factory
from chessarena.models import Base

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import verify_install  # noqa: E402


def test_database_probe_executes_under_sqlalchemy_2(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path / 'probe.db'}")
    Base.metadata.create_all(engine)
    session_factory = make_session_factory(engine)

    verify_install.failures.clear()
    try:
        verify_install.check_database(session_factory)
    finally:
        db_failures = [
            f for f in verify_install.failures
            if f.startswith("database unreachable:")
        ]
        verify_install.failures.clear()

    # The regression target is the SQL probe itself: it must execute without
    # raising the SQLAlchemy 2.x textual-SQL error.  An empty database may
    # legitimately report no enabled builds/openings.
    assert db_failures == []
