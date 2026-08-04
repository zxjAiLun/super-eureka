"""API tests (spec section 22.4).

Covers creation validation, lifecycle actions, invalid state transitions (409)
and read/query endpoints.  Concurrent execution is guaranteed by the
single-worker scheduler; a second match enqueues as QUEUED and never runs
before the first finishes (covered in test_scheduler.py).
"""

from __future__ import annotations

import pytest


def _create_payload(app_client, pairs=2, **overrides):
    build = app_client.get("/chessarena/api/v1/builds").json()[0]
    opening = app_client.get("/chessarena/api/v1/opening-sets").json()[0]
    payload = {
        "name": "api test",
        "engine_a": {"build_id": build["build_id"], "profile": "current-final"},
        "engine_b": {"build_id": build["build_id"], "profile": "current"},
        "opening_set_id": opening["opening_set_id"],
        "time_control": "blitz_3_2",
        "pairs": pairs,
    }
    payload.update(overrides)
    return payload


def _make(app_client, **overrides):
    response = app_client.post(
        "/chessarena/api/v1/tournaments", json=_create_payload(app_client, **overrides)
    )
    assert response.status_code == 201, response.text
    return response.json()


# ---------------------------------------------------------------------------
# Creation
# ---------------------------------------------------------------------------
def test_create_tournament(app_client):
    data = _make(app_client, pairs=5)
    assert data["status"] == "DRAFT"
    assert data["requested_pairs"] == 5
    assert data["completed_pairs"] == 0
    assert data["time_control"] == "blitz_3_2"
    assert data["config_snapshot"]["requested_pairs"] == 5
    assert data["config_snapshot"]["engine_a"]["profile"] == "current-final"
    assert data["config_snapshot"]["hash_mb"] == 32

    pairs = app_client.get(
        f"/chessarena/api/v1/tournaments/{data['id']}/pairs"
    ).json()
    assert len(pairs) == 5
    assert [p["pair_index"] for p in pairs] == list(range(5))
    assert all(p["status"] == "PENDING" for p in pairs)


def test_create_invalid_build(app_client):
    payload = _create_payload(app_client)
    payload["engine_a"]["build_id"] = "does-not-exist"
    response = app_client.post("/chessarena/api/v1/tournaments", json=payload)
    assert response.status_code == 422


def test_create_invalid_profile(app_client):
    payload = _create_payload(app_client)
    payload["engine_a"]["profile"] = "bogus-profile"
    response = app_client.post("/chessarena/api/v1/tournaments", json=payload)
    assert response.status_code == 422


def test_create_invalid_time_control(app_client):
    payload = _create_payload(app_client)
    payload["time_control"] = "custom-999+999"
    response = app_client.post("/chessarena/api/v1/tournaments", json=payload)
    assert response.status_code == 422


def test_create_pairs_over_capacity(app_client):
    payload = _create_payload(app_client, pairs=9999)
    response = app_client.post("/chessarena/api/v1/tournaments", json=payload)
    assert response.status_code == 422


def test_create_pairs_zero(app_client):
    payload = _create_payload(app_client, pairs=0)
    response = app_client.post("/chessarena/api/v1/tournaments", json=payload)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Lifecycle and state machine
# ---------------------------------------------------------------------------
def test_start_then_invalid_repeat_start(app_client):
    data = _make(app_client)
    tournament_id = data["id"]
    assert app_client.post(
        f"/chessarena/api/v1/tournaments/{tournament_id}/start"
    ).json()["status"] == "QUEUED"
    # Already QUEUED: start is not a valid transition -> 409
    assert app_client.post(
        f"/chessarena/api/v1/tournaments/{tournament_id}/start"
    ).status_code == 409


def test_pause_from_draft_is_409(app_client):
    data = _make(app_client)
    assert app_client.post(
        f"/chessarena/api/v1/tournaments/{data['id']}/pause"
    ).status_code == 409


def test_resume_from_draft_is_409(app_client):
    data = _make(app_client)
    assert app_client.post(
        f"/chessarena/api/v1/tournaments/{data['id']}/resume"
    ).status_code == 409


def test_resume_after_pause_roundtrip(app_client):
    data = _make(app_client)
    tournament_id = data["id"]
    # DRAFT -> QUEUED -> (worker would set RUNNING -> PAUSING -> PAUSED).
    # Simulate the worker-visible states directly for the transition test.
    assert app_client.post(
        f"/chessarena/api/v1/tournaments/{tournament_id}/start"
    ).status_code == 200
    # Force the DB into PAUSED via the model (as the worker would) and resume.
    from chessarena.db import get_db

    # Use the app's session via dependency override is overkill; flip directly:
    from sqlalchemy.orm import Session

    from chessarena.models import PAUSED, Tournament

    settings = app_client.app.state.settings
    from chessarena.db import make_session_factory

    session_factory = make_session_factory(
        make_engine_safe(settings.db_url)
    )
    with session_factory() as session:
        tournament = session.get(Tournament, tournament_id)
        tournament.status = PAUSED
        tournament.pause_requested = False
        session.commit()
    assert app_client.post(
        f"/chessarena/api/v1/tournaments/{tournament_id}/resume"
    ).json()["status"] == "QUEUED"


def make_engine_safe(db_url):
    from chessarena.db import make_engine

    return make_engine(db_url)


def test_cancel_queued_tournament(app_client):
    data = _make(app_client)
    tournament_id = data["id"]
    app_client.post(f"/chessarena/api/v1/tournaments/{tournament_id}/start")
    result = app_client.post(
        f"/chessarena/api/v1/tournaments/{tournament_id}/cancel"
    ).json()
    assert result["status"] == "CANCELLED"
    assert result["finished_at"] is not None


def test_cancel_draft_is_409(app_client):
    data = _make(app_client)
    assert app_client.post(
        f"/chessarena/api/v1/tournaments/{data['id']}/cancel"
    ).status_code == 409


def test_force_cancel_requires_confirm(app_client):
    data = _make(app_client)
    tournament_id = data["id"]
    assert app_client.post(
        f"/chessarena/api/v1/tournaments/{tournament_id}/force-cancel"
    ).status_code == 400
    app_client.post(f"/chessarena/api/v1/tournaments/{tournament_id}/start")
    result = app_client.post(
        f"/chessarena/api/v1/tournaments/{tournament_id}/force-cancel?confirm=true"
    ).json()
    assert result["status"] == "CANCELLED"


def test_second_tournament_enqueues_not_runs(app_client):
    first = _make(app_client, name="first")
    second = _make(app_client, name="second")
    app_client.post(f"/chessarena/api/v1/tournaments/{first['id']}/start")
    app_client.post(f"/chessarena/api/v1/tournaments/{second['id']}/start")
    # Both are QUEUED; only the single worker may promote one to RUNNING.
    assert app_client.get(
        f"/chessarena/api/v1/tournaments/{first['id']}"
    ).json()["status"] == "QUEUED"
    assert app_client.get(
        f"/chessarena/api/v1/tournaments/{second['id']}"
    ).json()["status"] == "QUEUED"


def test_unknown_tournament_404(app_client):
    assert app_client.get("/chessarena/api/v1/tournaments/nope").status_code == 404
    assert app_client.post(
        "/chessarena/api/v1/tournaments/nope/start"
    ).status_code == 404


# ---------------------------------------------------------------------------
# Queries and artifacts
# ---------------------------------------------------------------------------
def test_list_and_detail(app_client):
    data = _make(app_client, pairs=3)
    listing = app_client.get("/chessarena/api/v1/tournaments").json()
    assert any(t["id"] == data["id"] for t in listing)
    detail = app_client.get(
        f"/chessarena/api/v1/tournaments/{data['id']}"
    ).json()
    assert len(detail["pairs"]) == 3


def test_events_recorded_on_create(app_client):
    data = _make(app_client)
    events = app_client.get(
        f"/chessarena/api/v1/tournaments/{data['id']}/events"
    ).json()
    assert events[0]["event_type"] == "tournament_created"


def test_artifact_downloads_404_before_completion(app_client):
    data = _make(app_client)
    tournament_id = data["id"]
    assert app_client.get(
        f"/chessarena/api/v1/tournaments/{tournament_id}/pgn"
    ).status_code == 404
    assert app_client.get(
        f"/chessarena/api/v1/tournaments/{tournament_id}/summary"
    ).status_code == 404
    assert app_client.get(
        f"/chessarena/api/v1/tournaments/{tournament_id}/artifacts"
    ).status_code == 404


def test_raw_artifact_path_traversal_blocked(app_client):
    data = _make(app_client)
    tournament_id = data["id"]
    assert app_client.get(
        f"/chessarena/api/v1/tournaments/{tournament_id}/artifacts/raw",
        params={"path": "../../../etc/passwd"},
    ).status_code == 404
    assert app_client.get(
        f"/chessarena/api/v1/tournaments/{tournament_id}/artifacts/raw",
        params={"path": ".."},
    ).status_code == 404


# ---------------------------------------------------------------------------
# Health / registry reads
# ---------------------------------------------------------------------------
def test_health_endpoint(app_client):
    health = app_client.get("/chessarena/api/v1/health").json()
    assert health["database"] == "ok"
    assert health["status"] in ("ok", "degraded")


def test_registry_reads(app_client):
    builds = app_client.get("/chessarena/api/v1/builds").json()
    assert len(builds) == 1
    build_id = builds[0]["build_id"]
    detail = app_client.get(f"/chessarena/api/v1/builds/{build_id}").json()
    assert detail["binary_sha256"]
    assert app_client.get("/chessarena/api/v1/builds/nope").status_code == 404

    openings = app_client.get("/chessarena/api/v1/opening-sets").json()
    assert openings[0]["position_count"] == 20
    assert app_client.get(
        f"/chessarena/api/v1/opening-sets/{openings[0]['opening_set_id']}"
    ).json()["sha256"]
