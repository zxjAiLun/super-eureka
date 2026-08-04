"""Tests for the registration / install scripts (section 8, 20.1)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _run_script(name: str, *args, env_extra: dict | None = None):
    import os

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


def test_install_build_registers(settings, registered, tmp_path: Path):
    # registered build dir is already valid; re-register via the script.
    result = _run_script(
        "install_build.py",
        registered["build_dir"],
        "--overwrite",
        env_extra={"ARENA_DB_URL": settings.db_url},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "registered build" in result.stdout


def test_install_build_rejects_sha_mismatch(settings, tmp_path: Path):
    build_dir = tmp_path / "builds" / "broken"
    build_dir.mkdir(parents=True)
    (build_dir / "engine").write_bytes(b"content")
    manifest = {
        "schema_version": 1,
        "build_id": "broken",
        "engine_name": "X",
        "git_sha": "abc",
        "binary_sha256": "0" * 64,
        "platform": "linux-x86_64",
        "rustc_version": "1",
        "cargo_lock_sha256": "0" * 64,
        "supported_profiles": ["current"],
        "uci_id_name": "X",
        "uci_id_author": "r",
        "created_utc": "2026-08-05",
    }
    (build_dir / "manifest.json").write_text(json.dumps(manifest))
    result = _run_script(
        "install_build.py", build_dir, env_extra={"ARENA_DB_URL": settings.db_url}
    )
    assert result.returncode != 0
    assert "SHA mismatch" in result.stderr


def test_install_build_rejects_dir_name_mismatch(settings, tmp_path: Path):
    build_dir = tmp_path / "builds" / "other-name"
    build_dir.mkdir(parents=True)
    content = b"x"
    (build_dir / "engine").write_bytes(content)
    import hashlib

    manifest = {
        "schema_version": 1,
        "build_id": "different",  # does not match directory name
        "engine_name": "X",
        "git_sha": "abc",
        "binary_sha256": hashlib.sha256(content).hexdigest(),
        "platform": "linux-x86_64",
        "rustc_version": "1",
        "cargo_lock_sha256": "0" * 64,
        "supported_profiles": ["current"],
        "uci_id_name": "X",
        "uci_id_author": "r",
        "created_utc": "2026-08-05",
    }
    (build_dir / "manifest.json").write_text(json.dumps(manifest))
    result = _run_script(
        "install_build.py", build_dir, env_extra={"ARENA_DB_URL": settings.db_url}
    )
    assert result.returncode != 0
    assert "does not match" in result.stderr


def test_register_openings_registers(settings, registered, tmp_path: Path):
    epd = registered["opening_dir"] / "openings.epd"
    manifest = registered["opening_dir"] / "manifest.json"
    result = _run_script(
        "register_openings.py",
        epd,
        manifest,
        "--overwrite",
        env_extra={"ARENA_DB_URL": settings.db_url},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "registered opening set" in result.stdout


def test_register_openings_rejects_duplicate_positions(settings, tmp_path: Path):
    opening_dir = tmp_path / "openings" / "dup"
    opening_dir.mkdir(parents=True)
    import hashlib

    from chessarena.config import get_settings

    epd = opening_dir / "openings.epd"
    # Two identical lines -> duplicate position key.
    epd.write_text(
        "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 1 2\n"
        "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 1 2\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "opening_set_id": "dup",
        "format": "epd",
        "count": 2,
        "sha256": hashlib.sha256(epd.read_bytes()).hexdigest(),
        "unique_position_keys": True,
        "non_terminal": True,
        "created_utc": "2026-08-05",
    }
    manifest_path = opening_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    result = _run_script(
        "register_openings.py",
        epd,
        manifest_path,
        env_extra={"ARENA_DB_URL": settings.db_url},
    )
    assert result.returncode != 0
    assert "duplicate position" in result.stderr
