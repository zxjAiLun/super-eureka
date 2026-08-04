#!/usr/bin/env python3
"""Verify the bounded S3-FINAL Current vs CurrentFinal match artifact.

The verifier treats the PGN and manager logs as an audit artifact.  It does
not calculate Elo or SPRT decisions.  It checks the fixed 50-pair opening
contract, candidate-first color reversal, legal movetext, and manager failure
signals before producing a candidate-perspective W/D/L summary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import chess
import chess.pgn


PROFILE_CANDIDATE = "CurrentFinal"
PROFILE_BASELINE = "Current"
PROFILE_CANDIDATE_ARG = "current-final"
PROFILE_BASELINE_ARG = "current"
EXPECTED_GAME_COUNT = 100
EXPECTED_PAIR_COUNT = 50
EXPECTED_TIME_CONTROL = "10+0.1"
QUICK_SCREEN_THRESHOLD_PERCENT = 60.0
EXPECTED_STDERR = "Warning: 2 opening repetitions vs 1 games per encounter"
FAILURE_TOKEN_RE = re.compile(r"\b(?:forfeit|illegal|crash|timeout|fatal)\b", re.IGNORECASE)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_sha(repo: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    require(completed.returncode == 0, f"git rev-parse failed: {completed.stderr.strip()}")
    return completed.stdout.strip()


def normalized_fen(fen: str) -> str:
    fields = fen.strip().split()
    require(len(fields) >= 4, f"invalid FEN header: {fen!r}")
    if len(fields) == 4:
        fields.extend(("0", "1"))
    normalized = " ".join(fields[:6])
    chess.Board(normalized)
    return normalized


def position_key(fen: str) -> str:
    """Return the position identity without halfmove/fullmove clocks."""

    return " ".join(normalized_fen(fen).split()[:4])


def load_opening_fens(path: Path, count: int = EXPECTED_PAIR_COUNT) -> list[str]:
    fens: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        # The opening manifest is EPD-compatible; ignore optional operations
        # after the four or six FEN fields.
        line = line.split(";", 1)[0].strip()
        fens.append(normalized_fen(line))
    require(len(fens) >= count, f"opening source has fewer than {count} positions")
    selected = fens[:count]
    require(len(set(selected)) == count, "selected opening positions are not unique")
    return selected


def read_games(path: Path) -> list[chess.pgn.Game]:
    games: list[chess.pgn.Game] = []
    with path.open("r", encoding="utf-8") as handle:
        while True:
            game = chess.pgn.read_game(handle)
            if game is None:
                break
            require(not game.errors, f"PGN parse errors: {game.errors!r}")
            games.append(game)
    return games


def validate_game(game: chess.pgn.Game, index: int, expected_time_control: str) -> str:
    headers = game.headers
    white = headers.get("White")
    black = headers.get("Black")
    require(
        {white, black} == {PROFILE_CANDIDATE, PROFILE_BASELINE},
        f"game {index}: unexpected players {white!r} vs {black!r}",
    )
    result = headers.get("Result")
    require(result in {"1-0", "0-1", "1/2-1/2"}, f"game {index}: invalid result {result!r}")
    require(headers.get("TimeControl") == expected_time_control, f"game {index}: time control mismatch")
    fen = normalized_fen(headers.get("FEN", ""))
    board = chess.Board(fen)
    for ply, move in enumerate(game.mainline_moves()):
        require(board.is_legal(move), f"game {index}: illegal move {move.uci()} at ply {ply}")
        board.push(move)
    return fen


def validate_manager_logs(
    stdout_path: Path,
    stderr_path: Path,
    expected_game_count: int,
) -> dict[str, Any]:
    stdout = stdout_path.read_text(encoding="utf-8")
    stderr = stderr_path.read_text(encoding="utf-8")
    started = [int(value) for value in re.findall(r"^Started game (\d+)", stdout, re.MULTILINE)]
    finished = [int(value) for value in re.findall(r"^Finished game (\d+)", stdout, re.MULTILINE)]
    expected_numbers = list(range(1, expected_game_count + 1))
    require(started == expected_numbers, f"manager started-game sequence mismatch: {started!r}")
    require(finished == expected_numbers, f"manager finished-game sequence mismatch: {finished!r}")
    require(stdout.count("Finished match") == 1, "manager did not report exactly one Finished match")
    require(not FAILURE_TOKEN_RE.search(stdout), "manager stdout contains a failure token")
    stderr_lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    unexpected_stderr = [line for line in stderr_lines if line != EXPECTED_STDERR]
    require(not unexpected_stderr, f"unexpected manager stderr: {unexpected_stderr!r}")
    return {
        "started_games": len(started),
        "finished_games": len(finished),
        "finished_match": True,
        "unexpected_stderr": unexpected_stderr,
        "stdout_sha256": sha256_file(stdout_path),
        "stderr_sha256": sha256_file(stderr_path),
    }


def candidate_result(candidate_color: str, result: str) -> str:
    if result == "1/2-1/2":
        return "draw"
    candidate_won = (candidate_color == "White" and result == "1-0") or (
        candidate_color == "Black" and result == "0-1"
    )
    return "win" if candidate_won else "loss"


def quick_screen_exit_code(summary: dict[str, Any]) -> int:
    """Return 0 for the declared pass line and 3 for a gate rejection."""

    return 0 if float(summary["candidate_score_percent"]) >= QUICK_SCREEN_THRESHOLD_PERCENT else 3


def verify_match(
    pgn_path: Path,
    opening_path: Path,
    stdout_path: Path,
    stderr_path: Path,
    expected_game_count: int = EXPECTED_GAME_COUNT,
    expected_time_control: str = EXPECTED_TIME_CONTROL,
) -> dict[str, Any]:
    require(expected_game_count % 2 == 0, "expected game count must contain complete pairs")
    expected_pair_count = expected_game_count // 2
    expected_openings = load_opening_fens(opening_path, expected_pair_count)
    games = read_games(pgn_path)
    require(len(games) == expected_game_count, f"expected {expected_game_count} games, got {len(games)}")
    manager = validate_manager_logs(stdout_path, stderr_path, expected_game_count)

    expected_keys = {position_key(fen) for fen in expected_openings}
    by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    candidate_counts = Counter()
    candidate_colors = Counter()
    for index, game in enumerate(games, start=1):
        fen = validate_game(game, index, expected_time_control)
        key = position_key(fen)
        require(key in expected_keys, f"game {index}: opening position is not in selected source")
        candidate_color = "White" if game.headers["White"] == PROFILE_CANDIDATE else "Black"
        result = candidate_result(candidate_color, game.headers["Result"])
        candidate_counts[result] += 1
        candidate_colors[candidate_color] += 1
        by_key[key].append(
            {
                "game": index,
                "pgn_fen": fen,
                "candidate_color": candidate_color,
                "result": game.headers["Result"],
                "candidate_result": result,
            }
        )

    require(set(by_key) == expected_keys, "opening source and PGN pair set differ")
    pairs: list[dict[str, Any]] = []
    for source_fen in expected_openings:
        key = position_key(source_fen)
        pair = by_key[key]
        require(len(pair) == 2, f"opening position does not have exactly two games: {source_fen}")
        require(
            {item["candidate_color"] for item in pair} == {"White", "Black"},
            f"opening position does not strictly exchange candidate colors: {source_fen}",
        )
        pairs.append({"source_fen": source_fen, "position_key": key, "games": pair})

    score = (candidate_counts["win"] + 0.5 * candidate_counts["draw"]) / expected_game_count
    return {
        "schema_version": 1,
        "status": "MATCH_INTEGRITY_PASS",
        "analysis": "S3-FINAL bounded quick screen; descriptive W/D/L only",
        "games": expected_game_count,
        "opening_pairs": expected_pair_count,
        "candidate": PROFILE_CANDIDATE,
        "baseline": PROFILE_BASELINE,
        "candidate_wins": candidate_counts["win"],
        "candidate_losses": candidate_counts["loss"],
        "draws": candidate_counts["draw"],
        "candidate_score": round(score, 6),
        "candidate_score_percent": round(score * 100, 3),
        "candidate_color_counts": dict(candidate_colors),
        "pairs": pairs,
        "manager": manager,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pgn", type=Path, default=Path("results/s3-final/match/match.pgn"))
    parser.add_argument(
        "--openings", type=Path, default=Path("tests/data/openings/d1.14-openings-v1.epd")
    )
    parser.add_argument(
        "--stdout", type=Path, default=Path("results/s3-final/match/cutechess.stdout.log")
    )
    parser.add_argument(
        "--stderr", type=Path, default=Path("results/s3-final/match/cutechess.stderr.log")
    )
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument(
        "--cutechess", type=Path, default=Path("tools/.cache/cutechess-1.5.1-win64/cutechess-cli.exe")
    )
    parser.add_argument("--output", type=Path, default=Path("results/s3-final/match-summary.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    paths = [args.pgn, args.openings, args.stdout, args.stderr, args.engine, args.cutechess]
    if not all(path.expanduser().resolve().is_file() for path in paths):
        print("all S3-FINAL match inputs must exist", file=sys.stderr)
        return 2
    pgn = args.pgn.expanduser().resolve()
    openings = args.openings.expanduser().resolve()
    stdout = args.stdout.expanduser().resolve()
    stderr = args.stderr.expanduser().resolve()
    engine = args.engine.expanduser().resolve()
    cutechess = args.cutechess.expanduser().resolve()
    output = args.output.expanduser().resolve()
    try:
        summary = verify_match(pgn, openings, stdout, stderr)
        summary.update(
            {
                "git_sha": git_sha(repo),
                "opening_source": {
                    "path": str(openings),
                    "sha256": sha256_file(openings),
                    "selected_lines": list(range(1, EXPECTED_PAIR_COUNT + 1)),
                },
                "engine": {
                    "path": str(engine),
                    "sha256": sha256_file(engine),
                    "same_binary_for_both_profiles": True,
                    "candidate_argv": [str(engine), "--profile", PROFILE_CANDIDATE_ARG],
                    "baseline_argv": [str(engine), "--profile", PROFILE_BASELINE_ARG],
                    "hash_mb": 16,
                    "threads": 1,
                },
                "match": {
                    "manager": "cutechess-cli",
                    "manager_path": str(cutechess),
                    "manager_sha256": sha256_file(cutechess),
                    "time_control": EXPECTED_TIME_CONTROL,
                    "opening_order": "sequential first 50 source positions, each repeated with color swap",
                    "candidate_first": True,
                },
                "decision": {
                    "quick_screen_threshold_percent": QUICK_SCREEN_THRESHOLD_PERCENT,
                    "quick_screen_pass": quick_screen_exit_code(summary) == 0,
                    "formal_elo_or_sprt": False,
                    "current_promotion": False,
                },
            }
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(
            f"S3-FINAL match integrity PASS: {summary['games']} games, "
            f"candidate score={summary['candidate_score_percent']:.3f}%"
        )
        exit_code = quick_screen_exit_code(summary)
        if exit_code:
            print(
                "S3-FINAL quick-screen gate FAIL: candidate score is below "
                f"{QUICK_SCREEN_THRESHOLD_PERCENT:.1f}%",
                file=sys.stderr,
            )
        return exit_code
    except (OSError, ValueError) as exc:
        print(f"S3-FINAL match verification failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
