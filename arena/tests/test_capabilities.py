"""Deployment gate: enabled builds with NULL uci_options_schema block
worker start and degrade health until backfilled (P4.F1 B3b)."""

from __future__ import annotations


def test_health_reports_capability_gap(app_client, engine_factory, registered):
    # registered installs a build without a capability schema (install_build
    # does not probe UCI), so the gap must be visible.
    r = app_client.get("/chessarena/api/v1/health")
    body = r.json()
    assert body["uci_capability_gap"] >= 1
    assert body["status"] == "degraded"


def test_health_ok_when_all_enabled_builds_have_schema(
    app_client, engine_factory, registered
):
    from chessarena.models import EngineBuild

    with engine_factory() as session:
        for build in session.query(EngineBuild):
            build.uci_options_schema = {"Hash": {"type": "spin"}}
        session.commit()
    r = app_client.get("/chessarena/api/v1/health")
    body = r.json()
    assert body["uci_capability_gap"] == 0
    # status may be degraded for other reasons (worker offline in tests), but
    # the capability gap itself must not be the cause.
    assert body["status"] in ("ok", "degraded")


def test_worker_refuses_to_start_with_null_schema(
    settings, engine_factory, registered
):
    from chessarena.worker import run_worker

    rc = run_worker(settings, engine_factory)
    assert rc == 1
