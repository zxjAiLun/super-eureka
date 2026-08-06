"""Cutechess command construction tests (spec sections 6, 12, 22.1).

- time-control preset mapping is fixed and exact,
- argv is a plain list with no shell, and cannot carry arbitrary user input,
- the pair command contains the strict-color-swap settings.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chessarena.config import ENGINE_A_NAME, ENGINE_B_NAME, TIME_CONTROLS, Settings
from chessarena.services.cutechess import (
    build_pair_command,
    check_engine_binary,
    launch_cutechess,
    write_command_artifacts,
)


def test_time_control_presets_are_fixed():
    assert TIME_CONTROLS == {
        "bullet_1_0": {"label": "Bullet 1+0", "cutechess_tc": "60"},
        "blitz_3_2": {"label": "Blitz 3+2", "cutechess_tc": "180+2"},
        "rapid_5_3": {"label": "Rapid 5+3", "cutechess_tc": "300+3"},
    }


@pytest.mark.parametrize(
    "preset,expected",
    [
        ("bullet_1_0", "60"),
        ("blitz_3_2", "180+2"),
        ("rapid_5_3", "300+3"),
    ],
)
def test_preset_maps_to_cutechess_tc(preset, expected):
    assert TIME_CONTROLS[preset]["cutechess_tc"] == expected


def _build(settings: Settings):
    return build_pair_command(
        settings,
        engine_a={
            "binary_path": "/opt/chessarena/builds/20260805-bde9085-linux-x86_64/engine",
            "command_args": ["--profile", "current-final"],
            "uci_options": {},
        },
        engine_b={
            "binary_path": "/opt/chessarena/builds/20260805-bde9085-linux-x86_64/engine",
            "command_args": ["--profile", "current"],
            "uci_options": {},
        },
        time_control="180+2",
        hash_mb=32,
        opening_epd=Path("/var/lib/chessarena/runs/t/opening.epd"),
        pgn_out=Path("/var/lib/chessarena/runs/t/match.pgn"),
    )


def test_pair_command_structure(settings: Settings):
    argv = _build(settings)
    assert argv[0] == str(settings.cutechess)
    joined = " ".join(argv)

    assert f"name={ENGINE_A_NAME}" in joined
    assert f"name={ENGINE_B_NAME}" in joined
    assert "proto=uci" in joined
    assert "arg=--profile" in joined
    assert "arg=current-final" in joined
    assert "arg=current" in joined
    assert "tc=180+2" in joined
    assert "option.Hash=32" in joined
    assert "option.Threads=1" in joined
    assert "-rounds 2" in joined
    assert "-repeat 2" in joined
    assert "-concurrency 1" in joined
    assert "-variant standard" in joined
    assert "format=epd" in joined
    assert "order=sequential" in joined
    # No debug/recover in v1 (spec section 12).
    assert "-debug" not in joined
    assert "-recover" not in joined


def test_engine_without_profile_uses_uci_options(settings: Settings):
    """An engine like Stockfish takes no --profile; its UCI options are
    emitted as option.<name>=<value> under -each."""
    argv = build_pair_command(
        settings,
        engine_a={
            "binary_path": "/opt/chessarena/builds/stockfish/stockfish",
            "command_args": [],
            "uci_options": {
                "UCI_LimitStrength": True,
                "UCI_Elo": 2000,
                "Threads": 1,
            },
        },
        engine_b={
            "binary_path": "/opt/chessarena/builds/20260805-bde9085-linux-x86_64/engine",
            "command_args": ["--profile", "current-final"],
            "uci_options": {},
        },
        time_control="180+2",
        hash_mb=32,
        opening_epd=Path("/var/lib/chessarena/runs/t/opening.epd"),
        pgn_out=Path("/var/lib/chessarena/runs/t/match.pgn"),
    )
    joined = " ".join(argv)
    assert "option.UCI_LimitStrength=true" in joined
    assert "option.UCI_Elo=2000" in joined
    assert "arg=--profile" in joined  # engine_b still uses its preset args
    # preset Threads is overridden by the fixed arena constraint Threads=1
    assert joined.count("option.Threads=1") == 2


def test_no_shell_injection_possible(settings: Settings):
    # Even a hostile arg string cannot escape the argv array: there is no
    # shell anywhere in the launch path.
    malicious = "current-final; rm -rf /"
    argv = build_pair_command(
        settings,
        engine_a={
            "binary_path": "/opt/x/engine",
            "command_args": ["--profile", malicious],
            "uci_options": {},
        },
        engine_b={
            "binary_path": "/opt/y/engine",
            "command_args": ["--profile", "current"],
            "uci_options": {},
        },
        time_control="60",
        hash_mb=32,
        opening_epd=Path("/tmp/o.epd"),
        pgn_out=Path("/tmp/m.pgn"),
    )
    # The malicious string is one argv element, never parsed by a shell.
    assert f"arg={malicious}" in argv
    assert all("&&" not in arg and "|" not in arg for arg in argv)


def test_launch_uses_no_shell(settings: Settings, tmp_path: Path, monkeypatch):
    pair_dir = tmp_path / "pair"
    pair_dir.mkdir(parents=True)
    recorded = {}

    class FakePopen:
        def __init__(self, argv, cwd, stdin, stdout, stderr, start_new_session, shell):
            recorded["shell"] = shell
            recorded["argv"] = argv
            recorded["cwd"] = cwd
            recorded["start_new_session"] = start_new_session
            self._stdout_fh = stdout
            self._stderr_fh = stderr

    monkeypatch.setattr("chessarena.services.cutechess.subprocess.Popen", FakePopen)
    argv = _build(settings)
    launch_cutechess(argv, pair_dir)
    assert recorded["shell"] is False
    assert recorded["start_new_session"] is True
    assert recorded["cwd"] == str(pair_dir)


def test_write_command_artifacts(settings: Settings, tmp_path: Path):
    pair_dir = tmp_path / "pair"
    write_command_artifacts(pair_dir, ["a", "b"], extra={"k": "v"})
    assert (pair_dir / "command.txt").read_text() == "a b\n"
    import json

    command = json.loads((pair_dir / "command.json").read_text())
    assert command["argv"] == ["a", "b"]
    assert command["shell"] is False
    assert command["k"] == "v"


def test_check_engine_binary_rejects_mismatch(settings: Settings, tmp_path: Path):
    binary = tmp_path / "engine"
    binary.write_bytes(b"some content")
    with pytest.raises(RuntimeError, match="SHA"):
        check_engine_binary(
            {"binary_path": str(binary), "binary_sha256": "0" * 64,
             "build_id": "x"}
        )
