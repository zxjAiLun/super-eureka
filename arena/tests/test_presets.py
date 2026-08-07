"""Engine preset tests (P4.2).

Covers:
- scheduler execution driven by the frozen config_snapshot (args/options,
  legacy fallback, missing build);
- the create API freezes the full preset config into config_snapshot;
- same-preset creation is rejected unless explicitly allowed;
- changing a preset after creation must NOT change an existing tournament's
  execution config.
"""

from __future__ import annotations

import pytest

from chessarena.models import EngineBuild
from chessarena.services.scheduler import _engine_cfg_from_snapshot


def _snapshot(build_id, **side):
    return {"engine_a": {"build_id": build_id, **side}}


def test_engine_cfg_from_frozen_snapshot(engine_factory, registered):
    with engine_factory() as session:
        build = session.query(EngineBuild).first()
        cfg = _engine_cfg_from_snapshot(
            session,
            _snapshot(
                build.build_id,
                display_name="Stockfish Limited 2000",
                command_args=[],
                uci_options={"UCI_LimitStrength": True, "UCI_Elo": 2000},
                profile="preset:stockfish-limited-2000",
            ),
            "engine_a",
            "current-final",
            "EngineA",
        )
        assert cfg["binary_path"] == build.binary_path
        assert cfg["display_name"] == "Stockfish Limited 2000"
        assert cfg["command_args"] == []
        assert cfg["uci_options"] == {"UCI_LimitStrength": True, "UCI_Elo": 2000}


def test_engine_cfg_legacy_snapshot_falls_back_to_profile(engine_factory, registered):
    with engine_factory() as session:
        build = session.query(EngineBuild).first()
        # Pre-preset snapshot has no command_args/uci_options/display_name.
        cfg = _engine_cfg_from_snapshot(
            session,
            _snapshot(build.build_id, profile="current-final"),
            "engine_a",
            "current-final",
            "EngineA",
        )
        assert cfg["display_name"] == "EngineA"
        assert cfg["command_args"] == ["--profile", "current-final"]
        assert cfg["uci_options"] == {}


def test_engine_cfg_missing_build_raises(engine_factory, registered):
    from chessarena.services.cutechess import CutechessLaunchError

    with engine_factory() as session:
        with pytest.raises(CutechessLaunchError, match="missing build"):
            _engine_cfg_from_snapshot(
                session,
                _snapshot("does-not-exist", command_args=[], uci_options={}),
                "engine_a",
                "current-final",
                "EngineA",
            )


def test_engine_cfg_snapshot_without_build_id_raises(engine_factory, registered):
    from chessarena.services.cutechess import CutechessLaunchError

    with engine_factory() as session:
        with pytest.raises(CutechessLaunchError, match="no build_id"):
            _engine_cfg_from_snapshot(
                session, {"engine_a": {}}, "engine_a", "current-final", "EngineA"
            )


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


def test_preset_change_after_create_does_not_affect_snapshot(
    app_client, engine_factory
):
    """P1 regression: the config_snapshot is frozen at creation; later edits
    to the live EnginePreset must not rewrite an existing tournament."""
    opening = app_client.get("/chessarena/api/v1/opening-sets").json()[0]
    created = app_client.post(
        "/chessarena/api/v1/tournaments",
        json={
            "name": "frozen",
            "engine_a": {"preset_id": "chessengine-production"},
            "engine_b": {"preset_id": "chessengine-legacy-current"},
            "opening_set_id": opening["opening_set_id"],
            "time_control": "blitz_3_2",
            "pairs": 2,
        },
    )
    assert created.status_code == 201
    tid = created.json()["id"]
    frozen = created.json()["config_snapshot"]["engine_a"]

    from chessarena.models import EnginePreset

    with engine_factory() as session:
        preset = (
            session.query(EnginePreset)
            .filter(EnginePreset.preset_id == "chessengine-production")
            .one()
        )
        preset.display_name = "Changed Name"
        preset.command_args = ["--profile", "changed"]
        session.commit()

    detail = app_client.get(
        f"/chessarena/api/v1/tournaments/{tid}"
    ).json()
    assert detail["config_snapshot"]["engine_a"] == frozen


# ---------------------------------------------------------------------------
# P4.2 Repair round 2: frozen provenance is enforced at launch
# ---------------------------------------------------------------------------
def _launch_context(engine_factory, tournament_factory, status="QUEUED"):
    from chessarena.models import Tournament
    from chessarena.services import artifacts

    tid = tournament_factory(name="pinned", pairs=1, status=status)

    def _enter(session):
        t = (
            session.query(Tournament)
            .filter(Tournament.id == tid)
            .one()
        )
        pair = t.pair_jobs[0]
        run_dir = artifacts.pair_run_dir(t.id, pair.pair_index, pair.attempt)
        run_dir.mkdir(parents=True, exist_ok=True)
        return t, pair, run_dir

    return tid, _enter


def test_snapshot_sha_mismatch_fails_before_launch(
    settings, scheduler, engine_factory, tournament_factory
):
    """P1: if the live EngineBuild SHA differs from the frozen snapshot, the
    tournament must fail before Popen — no binary runs."""
    from chessarena.services.cutechess import CutechessLaunchError

    tid, _enter = _launch_context(engine_factory, tournament_factory)
    with engine_factory() as session:
        t, pair, run_dir = _enter(session)
        t.config_snapshot["engine_a"]["binary_sha256"] = "0" * 64
        session.commit()
        with pytest.raises(CutechessLaunchError, match="binary SHA differs"):
            scheduler._prepare_and_launch(session, t, pair, run_dir)
        assert scheduler.active_proc is None, "Popen must not have been called"


def test_snapshot_git_sha_mismatch_fails_before_launch(
    settings, scheduler, engine_factory, tournament_factory
):
    from chessarena.services.cutechess import CutechessLaunchError

    tid, _enter = _launch_context(engine_factory, tournament_factory)
    with engine_factory() as session:
        t, pair, run_dir = _enter(session)
        t.config_snapshot["engine_a"]["git_sha"] = "deadbeef"
        session.commit()
        with pytest.raises(CutechessLaunchError, match="git_sha"):
            scheduler._prepare_and_launch(session, t, pair, run_dir)
        assert scheduler.active_proc is None


def test_snapshot_hash_threads_used_in_command_and_verifier(
    settings, scheduler, engine_factory, tournament_factory
):
    """P1: Hash/Threads must come from the frozen snapshot (16/2), not the
    live Settings (32/1); the verifier rebuild must use the same values."""
    import json

    from chessarena.models import EngineBuild

    tid, _enter = _launch_context(engine_factory, tournament_factory)
    with engine_factory() as session:
        t, pair, run_dir = _enter(session)
        t.config_snapshot["hash_mb"] = 16
        t.config_snapshot["threads"] = 2
        session.commit()

        scheduler._prepare_and_launch(session, t, pair, run_dir)
        if scheduler.active_proc is not None:
            scheduler.active_proc.terminate()

        command = json.loads((run_dir / "command.json").read_text(encoding="utf-8"))
        joined = " ".join(command["argv"])
        # Hash must come from the frozen snapshot (16), not live settings (32).
        assert "option.Hash=16" in joined
        # Threads is not forced (ChessEngine lacks the option); it is frozen
        # as snapshot metadata only.
        assert "option.Threads" not in joined
        assert command["hash_mb"] == 16
        assert command["threads"] == 2

        from chessarena.services.verifier import _check_command_provenance

        engine_a = session.query(EngineBuild).first()
        # Rebuild must match (it uses the snapshot's 16/2, not settings 32/1).
        _check_command_provenance(
            settings,
            run_dir / "command.json",
            t.config_snapshot,
            run_dir,
            engine_a,
            engine_a,
        )
