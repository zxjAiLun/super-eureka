"""Stockfish install + preset registration script tests (P4.2 Phase B)."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from chessarena.models import EngineBuild, EnginePreset

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
FIXTURES = Path(__file__).resolve().parent / "fixtures"
FAKE_UCI_ENGINE = FIXTURES / "fake_uci_engine.py"


def _run_script(name: str, *args, env_extra: dict | None = None):
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, str(SCRIPTS / name), *map(str, args)],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stage_binary(tmp_path: Path) -> Path:
    binary = tmp_path / "stockfish"
    binary.write_bytes(FAKE_UCI_ENGINE.read_bytes())
    binary.chmod(0o755)
    return binary


def test_install_external_build_registers(settings, engine_factory, tmp_path):
    binary = _stage_binary(tmp_path)
    sha = _sha256(binary)
    result = _run_script(
        "install_external_build.py",
        tmp_path,
        "--build-id", "stockfish-17.1-linux-x86_64",
        "--engine-name", "Stockfish",
        "--binary-name", "stockfish",
        "--binary-sha256", sha,
        "--platform", "linux-x86_64",
        env_extra={"ARENA_DB_URL": settings.db_url},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "registered external build" in result.stdout
    assert "FakeStockfish 17.1" in result.stdout

    with engine_factory() as session:
        build = (
            session.query(EngineBuild)
            .filter(EngineBuild.build_id == "stockfish-17.1-linux-x86_64")
            .one()
        )
        assert build.engine_name == "Stockfish"
        assert build.binary_sha256 == sha
        assert build.supported_profiles == []


def test_install_external_build_idempotent_same_sha(
    settings, engine_factory, tmp_path
):
    binary = _stage_binary(tmp_path)
    sha = _sha256(binary)
    args = [
        "install_external_build.py",
        tmp_path,
        "--build-id", "stockfish-17.1-linux-x86_64",
        "--engine-name", "Stockfish",
        "--binary-name", "stockfish",
        "--binary-sha256", sha,
        "--platform", "linux-x86_64",
    ]
    first = _run_script(*args, env_extra={"ARENA_DB_URL": settings.db_url})
    assert first.returncode == 0, first.stderr
    second = _run_script(*args, env_extra={"ARENA_DB_URL": settings.db_url})
    assert second.returncode == 0
    assert "already registered" in second.stdout


def test_install_external_build_rejects_sha_mismatch(settings, tmp_path):
    binary = _stage_binary(tmp_path)
    result = _run_script(
        "install_external_build.py",
        tmp_path,
        "--build-id", "stockfish-x",
        "--engine-name", "Stockfish",
        "--binary-name", "stockfish",
        "--binary-sha256", "0" * 64,
        "--platform", "linux-x86_64",
        env_extra={"ARENA_DB_URL": settings.db_url},
    )
    assert result.returncode != 0
    assert "SHA mismatch" in result.stderr


def test_install_external_build_rejects_elo_out_of_range(
    settings, engine_factory, tmp_path
):
    binary = _stage_binary(tmp_path)
    sha = _sha256(binary)
    result = _run_script(
        "install_external_build.py",
        tmp_path,
        "--build-id", "stockfish-narrow",
        "--engine-name", "Stockfish",
        "--binary-name", "stockfish",
        "--binary-sha256", sha,
        "--platform", "linux-x86_64",
        "--uci-elos", "1800,2000",
        env_extra={
            "ARENA_DB_URL": settings.db_url,
            "FAKE_UCI_ELO_MIN": "1000",
            "FAKE_UCI_ELO_MAX": "1500",
        },
    )
    assert result.returncode != 0
    assert "engine maximum" in result.stderr


def test_register_stockfish_presets_and_idempotent(
    settings, engine_factory, tmp_path
):
    binary = _stage_binary(tmp_path)
    sha = _sha256(binary)
    build_id = "stockfish-17.1-linux-x86_64"
    install = _run_script(
        "install_external_build.py",
        tmp_path,
        "--build-id", build_id,
        "--engine-name", "Stockfish",
        "--binary-name", "stockfish",
        "--binary-sha256", sha,
        "--platform", "linux-x86_64",
        env_extra={"ARENA_DB_URL": settings.db_url},
    )
    assert install.returncode == 0, install.stderr

    args = [
        "register_stockfish_presets.py",
        "--build-id", build_id,
    ]
    first = _run_script(*args, env_extra={"ARENA_DB_URL": settings.db_url})
    assert first.returncode == 0, first.stdout + first.stderr
    assert "created preset stockfish-limited-1800" in first.stdout
    assert "created preset stockfish-limited-2400" in first.stdout

    with engine_factory() as session:
        for elo in (1800, 2000, 2200, 2400):
            preset = (
                session.query(EnginePreset)
                .filter(EnginePreset.preset_id == f"stockfish-limited-{elo}")
                .one()
            )
            assert preset.build_id == build_id
            assert preset.display_name == f"Stockfish Limited {elo}"
            assert preset.command_args == []
            assert preset.uci_options == {
                "UCI_LimitStrength": True,
                "UCI_Elo": elo,
            }

    second = _run_script(*args, env_extra={"ARENA_DB_URL": settings.db_url})
    assert second.returncode == 0
    assert "updated preset" in second.stdout


def test_register_stockfish_presets_requires_build(settings, engine_factory):
    result = _run_script(
        "register_stockfish_presets.py",
        "--build-id", "not-registered",
        env_extra={"ARENA_DB_URL": settings.db_url},
    )
    assert result.returncode != 0
    assert "not registered" in result.stderr


def test_pgn_display_names_from_presets(settings, engine_factory, app_client):
    """A tournament created from the stockfish presets must carry the preset
    display names in the snapshot (the PGN-facing names)."""
    opening = app_client.get("/chessarena/api/v1/opening-sets").json()[0]

    # Register a stockfish build + presets in the app's DB directly.
    from chessarena.models import EngineBuild as EB, EnginePreset as EP

    with engine_factory() as session:
        build = session.query(EB).first()  # the registered dummy build
        session.add(
            EP(
                preset_id="stockfish-limited-2000",
                build_id=build.build_id,
                display_name="Stockfish Limited 2000",
                command_args=[],
                uci_options={"UCI_LimitStrength": True, "UCI_Elo": 2000},
                category="external",
                public_visible=True,
                enabled=True,
            )
        )
        session.commit()

    created = app_client.post(
        "/chessarena/api/v1/tournaments",
        json={
            "name": "sf",
            "engine_a": {"preset_id": "chessengine-production"},
            "engine_b": {"preset_id": "stockfish-limited-2000"},
            "opening_set_id": opening["opening_set_id"],
            "time_control": "blitz_3_2",
            "pairs": 2,
        },
    )
    assert created.status_code == 201, created.text
    snapshot = created.json()["config_snapshot"]
    assert snapshot["engine_a"]["display_name"] == "ChessEngine Production"
    assert snapshot["engine_b"]["display_name"] == "Stockfish Limited 2000"
    assert snapshot["engine_b"]["command_args"] == []
    assert snapshot["engine_b"]["uci_options"] == {
        "UCI_LimitStrength": True,
        "UCI_Elo": 2000,
    }
    # Historical audit profile column records the external preset id.
    assert snapshot["engine_b"]["profile"] == "preset:stockfish-limited-2000"
