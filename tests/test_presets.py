"""Engine preset tests (P4.2).

Covers:
- scheduler preset resolution (args/options, fallback, missing preset);
- the create API freezes the full preset config into config_snapshot;
- same-preset creation is rejected unless explicitly allowed.
"""

from __future__ import annotations

import pytest

from chessarena.models import EngineBuild
from chessarena.services.scheduler import _resolve_engine_cfg


def test_resolve_engine_cfg_uses_preset(engine_factory, registered):
    with engine_factory() as session:
        build = session.query(EngineBuild).first()
        cfg = _resolve_engine_cfg(
            session, build, "chessengine-production", "current-final", "EngineA"
        )
        assert cfg["binary_path"] == build.binary_path
        assert cfg["display_name"] == "ChessEngine Production"
        assert cfg["command_args"] == ["--profile", "current-final"]
        assert cfg["uci_options"] == {}


def test_resolve_engine_cfg_falls_back_to_profile(engine_factory, registered):
    with engine_factory() as session:
        build = session.query(EngineBuild).first()
        cfg = _resolve_engine_cfg(session, build, None, "current-final", "EngineA")
        assert cfg["display_name"] == "EngineA"
        assert cfg["command_args"] == ["--profile", "current-final"]
        assert cfg["uci_options"] == {}


def test_resolve_engine_cfg_missing_preset_raises(engine_factory, registered):
    from chessarena.services.cutechess import CutechessLaunchError

    with engine_factory() as session:
        build = session.query(EngineBuild).first()
        with pytest.raises(CutechessLaunchError, match="preset not found"):
            _resolve_engine_cfg(session, build, "stockfish-2000", "current", "EngineA")


def test_config_snapshot_freezes_preset(app_client):
    opening = app_client.get("/chessarena/api/v1/opening-sets").json()[0]
    created = app_client.post(
        "/chessarena/api/v1/tournaments",
        json={
            "name": "preset snapshot",
            "engine_a": {"preset_id": "chessengine-production"},
            "engine_b": {"preset_id": "chessengine-legacy-current"},
            "opening_set_id": opening["opening_set_id"],
            "time_control": "blitz_3_2",
            "pairs": 2,
        },
    )
    assert created.status_code == 201
    snapshot = created.json()["config_snapshot"]
    assert snapshot["engine_a"]["preset_id"] == "chessengine-production"
    assert snapshot["engine_a"]["command_args"] == ["--profile", "current-final"]
    assert snapshot["engine_a"]["uci_options"] == {}
    assert snapshot["engine_a"]["binary_sha256"]
    assert snapshot["engine_b"]["preset_id"] == "chessengine-legacy-current"
    assert created.json()["engine_a_preset_id"] == "chessengine-production"
    assert created.json()["engine_b_preset_id"] == "chessengine-legacy-current"


def test_same_preset_requires_allowance(app_client):
    opening = app_client.get("/chessarena/api/v1/opening-sets").json()[0]

    def _create(allow: bool):
        return app_client.post(
            "/chessarena/api/v1/tournaments",
            json={
                "name": "selfplay",
                "engine_a": {"preset_id": "chessengine-production"},
                "engine_b": {"preset_id": "chessengine-production"},
                "opening_set_id": opening["opening_set_id"],
                "time_control": "bullet_1_0",
                "pairs": 2,
                "allow_intentional_self_play": allow,
            },
        )

    assert _create(False).status_code == 422
    assert _create(True).status_code == 201
