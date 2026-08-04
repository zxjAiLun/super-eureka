#!/usr/bin/env python3
"""Fake cutechess-cli used by the arena test suite.

It mimics the parts of cutechess-cli the worker depends on:
- answers ``-version``,
- writes a ``match.pgn`` with two color-swapped games for the opening in the
  openings file,
- writes a ``stdout.log`` containing the final ``Score of A vs B`` line,
- writes a (usually empty) ``stderr.log``,
- exits 0.

Behavior is controlled through environment variables so the verifier test
matrix can inject every failure mode described in spec section 22.2.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import chess

ENV = os.environ


def _parse_option_value(argv: list[str], option: str) -> str | None:
    for arg in argv:
        if arg.startswith(option + "="):
            return arg[len(option) + 1:]
    return None


def _flag_value(argv: list[str], flag: str) -> str | None:
    """Value following a flag given as a separate argv element (e.g. -pgnout)."""
    for idx, arg in enumerate(argv):
        if arg == flag and idx + 1 < len(argv) and not argv[idx + 1].startswith("-"):
            return argv[idx + 1]
    return None


def _arg_in(argv: list[str], name: str) -> list[str]:
    """Return values following the named flag (e.g. -engine -> name=... cmd=...)."""
    values: list[str] = []
    capture = False
    for arg in argv:
        if arg == name:
            capture = True
            continue
        if capture:
            if arg.startswith("-"):
                break
            values.append(arg)
    return values


def _engine_info(argv: list[str]) -> tuple[str, str]:
    engine_a = engine_b = ""
    current: list[str] = []
    engines: list[list[str]] = []
    for arg in argv:
        if arg == "-engine":
            if current:
                engines.append(current)
            current = []
        elif current is not None and not arg.startswith("-"):
            current.append(arg)
    if current:
        engines.append(current)
    if len(engines) >= 1:
        engine_a = _parse_option_value(engines[0], "name") or "EngineA"
    if len(engines) >= 2:
        engine_b = _parse_option_value(engines[1], "name") or "EngineB"
    return engine_a, engine_b


def _opening_fen(argv: list[str]) -> str:
    openings_file = _parse_option_value(argv, "file")
    if not openings_file or not Path(openings_file).exists():
        return chess.Board().fen()
    for line in Path(openings_file).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        return chess.Board(line.split(";")[0].strip()).fen()
    return chess.Board().fen()


def _time_control(argv: list[str]) -> str:
    tc = _parse_option_value(argv, "tc")
    return tc or "60"


def _results() -> list[str]:
    raw = ENV.get("FAKE_CUTECHESS_RESULTS", "1-0,0-1")
    return [r.strip() for r in raw.split(",") if r.strip()]


def main() -> int:
    argv = sys.argv[1:]

    if "-version" in argv:
        print("cutechess-cli 1.5.1 (fake test fixture)")
        return 0

    sleep_ms = float(ENV.get("FAKE_CUTECHESS_SLEEP_MS", "0"))
    if sleep_ms > 0:
        time.sleep(sleep_ms / 1000.0)

    engine_a, engine_b = _engine_info(argv)
    fen = _opening_fen(argv)
    tc = _time_control(argv)

    results = _results()
    if ENV.get("FAKE_CUTECHESS_GAMES"):
        results = results[: int(ENV["FAKE_CUTECHESS_GAMES"])]
        if not results:
            results = ["1/2-1/2"]

    pgn_out = _parse_option_value(argv, "pgnout") or _flag_value(argv, "-pgnout") or "match.pgn"
    pair_dir = Path(pgn_out).parent
    pair_dir.mkdir(parents=True, exist_ok=True)

    empty_pgn = ENV.get("FAKE_CUTECHESS_EMPTY_PGN") == "1"
    if not empty_pgn:
        _write_pgn(pair_dir / Path(pgn_out).name, engine_a, engine_b, fen, tc, results)
    _write_stdout(pair_dir / "stdout.log", engine_a, engine_b, results)
    _write_stderr(pair_dir / "stderr.log")

    return 0


def _game_text(round_no: int, white: str, black: str, fen: str, tc: str,
               result: str, bad_move: bool) -> str:
    board = chess.Board(fen)
    moves: list[str] = []
    if bad_move:
        # "Ke2" from the opening position: either illegal or unparsable by
        # python-chess depending on the position - both fail verification.
        moves.append("Ke2")
    else:
        legal = list(board.legal_moves)
        if legal:
            moves.append(board.san(legal[0]))
    termination = {
        "1-0": "White mates",
        "0-1": "Black mates",
        "1/2-1/2": "Draw by 3-fold repetition",
    }.get(result, "Normal")
    header = [
        f'[Event "fake pair {round_no}"]',
        '[Site "localhost"]',
        f'[Round "{round_no}"]',
        f'[White "{white}"]',
        f'[Black "{black}"]',
        f'[Result "{result}"]',
        f'[TimeControl "{tc}"]',
        f'[Termination "{termination}"]',
        f'[FEN "{board.fen()}"]',
        '[SetUp "1"]',
    ]
    if moves:
        header.append("")
        header.append(f"1. {moves[0]}")
    header.append(result)
    return "\n".join(header) + "\n\n"


def _write_pgn(path: Path, engine_a: str, engine_b: str, fen: str, tc: str,
               results: list[str]) -> None:
    same_colors = ENV.get("FAKE_CUTECHESS_SAME_COLORS") == "1"
    bad_move = ENV.get("FAKE_CUTECHESS_BAD_MOVE") == "1"
    different_opening = ENV.get("FAKE_CUTECHESS_DIFFERENT_OPENING") == "1"
    wrong_tc = ENV.get("FAKE_CUTECHESS_TC_WRONG") == "1"

    parts: list[str] = []
    for idx, result in enumerate(results):
        if same_colors:
            white, black = engine_a, engine_b
        elif idx % 2 == 0:
            white, black = engine_a, engine_b
        else:
            white, black = engine_b, engine_a
        game_fen = fen
        if different_opening and idx >= 1:
            # A distinct but legal position (different move counters).
            game_fen = chess.Board(fen).fen().replace(" - 0 1", " - 3 5")
            board = chess.Board(fen)
            if list(board.legal_moves):
                board.push(list(board.legal_moves)[0])
                game_fen = board.fen()
        game_tc = "1+99" if wrong_tc else tc
        parts.append(
            _game_text(idx + 1, white, black, game_fen, game_tc, result,
                       bad_move and idx == 0)
        )
    path.write_text("".join(parts), encoding="utf-8", newline="\n")


def _write_stdout(path: Path, engine_a: str, engine_b: str, results: list[str]) -> None:
    wins = losses = draws = 0
    for idx, result in enumerate(results):
        if result == "1/2-1/2":
            draws += 1
        elif idx % 2 == 0:  # A is white
            wins += 1 if result == "1-0" else 0
            losses += 1 if result == "0-1" else 0
        else:  # A is black
            wins += 1 if result == "0-1" else 0
            losses += 1 if result == "1-0" else 0
        print(f"Finished game {idx + 1} ({engine_a} vs {engine_b}): {result}",
              file=sys.stderr)  # shown in probe logs only
    total = wins + losses + draws
    score = (wins + 0.5 * draws) / total if total else 0.0
    lines = [
        f"Finished game 1 ({engine_a} vs {engine_b}): {results[0] if results else '1/2-1/2'}",
        f"Score of {engine_a} vs {engine_b}: {wins} - {losses} - {draws}  [{score:.3f}] {total}",
    ]
    if ENV.get("FAKE_CUTECHESS_STDOUT_FORBIDDEN") == "1":
        lines.append("Warning: illegal move rejected")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _write_stderr(path: Path) -> None:
    if ENV.get("FAKE_CUTECHESS_STDERR_BAD") == "1":
        path.write_text("ERROR: engine crashed\n", encoding="utf-8")
    else:
        path.write_text("", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
