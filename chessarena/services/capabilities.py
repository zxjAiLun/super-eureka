"""Deployment gate helpers for UCI capability backfill (P4.F1 B3b).

Migration 0006 leaves pre-existing enabled builds with
``uci_options_schema=NULL``.  Running tournaments against such builds would
silently omit runtime options (Hash/Threads/Ponder/...) that the frozen
snapshot expects.  The deployment gate therefore fails health / refuses to
start the worker until every enabled build has been backfilled.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from ..models import EngineBuild


def enabled_builds_without_uci_schema(session: Session) -> int:
    return (
        session.query(EngineBuild)
        .filter(
            EngineBuild.enabled.is_(True),
            EngineBuild.uci_options_schema.is_(None),
        )
        .count()
    )
