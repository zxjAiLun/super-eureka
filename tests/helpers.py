"""Helpers for constructing pair artifacts used by verifier/recovery tests."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from chessarena.config import Settings
from chessarena.services import artifacts
from chessarena.services.cutechess import build_pair_command, write_command_artifacts


def opening_fen_for_index(opening_set, opening_index: int) -> str:
    """The normalized FEN of opening_index-th position of the opening set."""
    import chess

    lines = [
        ln.strip()
        for ln in Path(opening_set.file_path).read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
    return chess.Board(lines[opening_index].split(";")[0].strip()).fen()


def run_fake_pair(
    settings: Settings,
    *,
    tournament,
    pair_job,
    engine_a_build,
    engine_b_build,
    opening_set,
    env_extra: dict[str, str] | None = None,
) -> Path:
    """Create and populate a pair run directory with the fake cutechess.

    Returns the run directory.  ``env_extra`` controls the fake's failure
    modes (see tests/fixtures/fake_cutechess.py).
    """
    run_dir = artifacts.pair_run_dir(
        tournament.id, pair_job.pair_index, pair_job.attempt
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    opening_epd = run_dir / "opening.epd"
    opening_epd.write_text(
        opening_fen_for_index(opening_set, pair_job.opening_index) + "\n",
        encoding="utf-8",
    )

    from chessarena.config import TIME_CONTROLS

    tc = TIME_CONTROLS[tournament.time_control]["cutechess_tc"]
    argv = build_pair_command(
        settings,
        engine_a={
            "binary_path": engine_a_build.binary_path,
            "profile": tournament.engine_a_profile,
        },
        engine_b={
            "binary_path": engine_b_build.binary_path,
            "profile": tournament.engine_b_profile,
        },
        time_control=tc,
        hash_mb=settings.hash_mb,
        opening_epd=opening_epd,
        pgn_out=run_dir / "match.pgn",
    )
    write_command_artifacts(
        run_dir,
        argv,
        extra={
            "tournament_id": tournament.id,
            "pair_index": pair_job.pair_index,
            "attempt": pair_job.attempt,
        },
    )
    env = dict(os.environ)
    env_extra = dict(env_extra or {})
    env.setdefault("FAKE_CUTECHESS_RESULTS", "1-0,0-1")
    env.update(env_extra)
    result = subprocess.run(argv, capture_output=True, text=True, timeout=60, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"fake cutechess failed rc={result.returncode}: {result.stderr}")
    return run_dir


def write_pair_artifacts(
    run_dir: Path,
    *,
    games: list[dict],
    stdout_lines: list[str],
    stderr_lines: list[str] = (),
    command_argv: list[str],
) -> None:
    """Directly write a pair directory (used when hand-crafting failures)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    pgn_parts = []
    for idx, game in enumerate(games, start=1):
        moves = game.get("moves", "")
        movetext = f"{moves} {game['result']}".strip()
        block = (
            "\n".join(
                [
                    f'[Event "pair {idx}"]',
                    f'[White "{game["white"]}"]',
                    f'[Black "{game["black"]}"]',
                    f'[Result "{game["result"]}"]',
                    f'[TimeControl "{game.get("tc", "60")}"]',
                    f'[Termination "{game.get("termination", "Normal")}"]',
                    f'[FEN "{game["fen"]}"]',
                    '[SetUp "1"]',
                    "",
                    movetext,
                    "",
                ]
            )
        )
        pgn_parts.append(block)
    (run_dir / "match.pgn").write_text("\n".join(pgn_parts), encoding="utf-8")
    (run_dir / "stdout.log").write_text(
        "\n".join(stdout_lines) + "\n", encoding="utf-8"
    )
    (run_dir / "stderr.log").write_text(
        "\n".join(stderr_lines) + "\n" if stderr_lines else "", encoding="utf-8"
    )
    write_command_artifacts(
        run_dir,
        command_argv,
        extra={"tournament_id": "test", "pair_index": 0, "attempt": 1},
    )
