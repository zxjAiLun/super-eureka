#!/usr/bin/env python3
"""Verify a completed S3-PROMOTION manager artifact.

This verifier does not reimplement SPRT.  It checks the frozen provenance,
manager lifecycle, early-stop boundary, opening order, color reversal, PGN
legality, and candidate-perspective W/D/L counts, while preserving the
manager's explicit H0/H1 decision as the statistical result.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import chess
import chess.pgn


PROFILE_CANDIDATE = "CurrentFinal"
PROFILE_BASELINE = "Current"
PROFILE_CANDIDATE_ARG = "current-final"
PROFILE_BASELINE_ARG = "current"
FROZEN_ENGINE_GIT_SHA = "91347775906f3f5d3730c9e9596037493429776d"
FROZEN_ENGINE_SHA256 = "b4bf0c3e73158bf3f5c072aa863aca671721275efc5b2e6d5354cf53a0fd0933"
FROZEN_MANAGER_SHA256 = "8889f9582dc688c567704cf083f6025baf77f791cde903698c70b3420caf5d7e"
EXPECTED_MANAGER_VERSION = "cutechess-cli 1.5.1"
EXPECTED_TIME_CONTROL = "10+0.1"
EXPECTED_HASH_MB = 16
EXPECTED_THREADS = 1
EXPECTED_CONCURRENCY = 1
EXPECTED_GAMES_MAX = 1000
EXPECTED_PAIRS_MAX = 500
EXPECTED_STDERR = "Warning: 2 opening repetitions vs 1 games per encounter"
FAILURE_TOKEN_RE = re.compile(r"\b(?:error|forfeit|illegal|crash|timeout|fatal)\b", re.IGNORECASE)
H1_RE = re.compile(r"SPRT:.*-\s*H1 was accepted", re.IGNORECASE)
H0_RE = re.compile(r"SPRT:.*-\s*H0 was accepted", re.IGNORECASE)
SCORE_RE = re.compile(
    rf"Score of {re.escape(PROFILE_CANDIDATE)} vs {re.escape(PROFILE_BASELINE)}:\s*"
    r"(\d+)\s*-\s*(\d+)\s*-\s*(\d+)"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=False
    )
    require(result.returncode == 0, f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def source_matches_recorded_commit(repo: Path, recorded_sha: str) -> bool:
    result = subprocess.run(
        [
            "git",
            "diff",
            "--quiet",
            recorded_sha,
            "--",
            "src",
            "Cargo.toml",
            "Cargo.lock",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def normalized_fen(fen: str) -> str:
    fields = fen.strip().split()
    require(len(fields) >= 4, f"invalid FEN: {fen!r}")
    if len(fields) == 4:
        fields.extend(("0", "1"))
    normalized = " ".join(fields[:6])
    chess.Board(normalized)
    return normalized


def position_key(fen: str) -> str:
    return " ".join(normalized_fen(fen).split()[:4])


def load_runtime_openings(path: Path) -> list[str]:
    openings: list[str] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split(";", 1)[0].strip().split()
        fen = " ".join(fields[:6] if len(fields) >= 6 else fields[:4] + ["0", "1"])
        normalized = normalized_fen(fen)
        require(any(chess.Board(normalized).legal_moves), f"opening line {line_number} is terminal")
        openings.append(normalized)
    require(len(openings) == EXPECTED_PAIRS_MAX, f"expected {EXPECTED_PAIRS_MAX} runtime openings")
    require(len({position_key(fen) for fen in openings}) == EXPECTED_PAIRS_MAX, "runtime openings are not unique")
    return openings


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


def validate_game(game: chess.pgn.Game, index: int) -> tuple[str, str, str]:
    headers = game.headers
    white = headers.get("White")
    black = headers.get("Black")
    require(
        {white, black} == {PROFILE_CANDIDATE, PROFILE_BASELINE},
        f"game {index}: unexpected players {white!r} vs {black!r}",
    )
    result = headers.get("Result")
    require(result in {"1-0", "0-1", "1/2-1/2"}, f"game {index}: invalid result {result!r}")
    require(headers.get("TimeControl") == EXPECTED_TIME_CONTROL, f"game {index}: time control mismatch")
    fen = normalized_fen(headers.get("FEN", ""))
    board = chess.Board(fen)
    for ply, move in enumerate(game.mainline_moves()):
        require(board.is_legal(move), f"game {index}: illegal move {move.uci()} at ply {ply}")
        board.push(move)
    candidate_color = "White" if white == PROFILE_CANDIDATE else "Black"
    return fen, candidate_color, result


def candidate_result(candidate_color: str, result: str) -> str:
    if result == "1/2-1/2":
        return "draw"
    candidate_won = (candidate_color == "White" and result == "1-0") or (
        candidate_color == "Black" and result == "0-1"
    )
    return "win" if candidate_won else "loss"


def validate_manager_logs(stdout_path: Path, stderr_path: Path, games: int) -> dict[str, Any]:
    stdout = stdout_path.read_text(encoding="utf-8")
    stderr = stderr_path.read_text(encoding="utf-8")
    started = [int(value) for value in re.findall(r"^Started game (\d+)", stdout, re.MULTILINE)]
    finished = [int(value) for value in re.findall(r"^Finished game (\d+)", stdout, re.MULTILINE)]
    expected_numbers = list(range(1, games + 1))
    require(started == expected_numbers, f"started-game sequence mismatch: {started!r}")
    require(finished == expected_numbers, f"finished-game sequence mismatch: {finished!r}")
    require(stdout.count("Finished match") == 1, "manager did not report exactly one Finished match")
    require(not FAILURE_TOKEN_RE.search(stdout), "manager stdout contains a failure token")
    stderr_lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    unexpected_stderr = [line for line in stderr_lines if line != EXPECTED_STDERR]
    require(not unexpected_stderr, f"unexpected manager stderr: {unexpected_stderr!r}")

    h1 = bool(H1_RE.search(stdout))
    h0 = bool(H0_RE.search(stdout))
    require(not (h1 and h0), "manager output contains both H1 and H0 decisions")
    decision = "H1_ACCEPTED" if h1 else "H0_ACCEPTED" if h0 else "INCONCLUSIVE"
    if decision == "INCONCLUSIVE":
        require(games == EXPECTED_GAMES_MAX, "inconclusive match did not reach the game limit")
    else:
        require(games < EXPECTED_GAMES_MAX, "boundary decision was reported only at the game limit")

    scores = SCORE_RE.findall(stdout)
    require(scores, "manager final candidate score line is missing")
    candidate_wins, baseline_wins, draws = (int(value) for value in scores[-1])
    require(candidate_wins + baseline_wins + draws == games, "manager score line does not sum to game count")
    return {
        "started_games": games,
        "finished_games": games,
        "finished_match": True,
        "sprt_decision": decision,
        "manager_score": {
            "candidate_wins": candidate_wins,
            "baseline_wins": baseline_wins,
            "draws": draws,
        },
        "unexpected_stderr": unexpected_stderr,
        "stdout_sha256": sha256_file(stdout_path),
        "stderr_sha256": sha256_file(stderr_path),
    }


def verify_artifact(run_dir: Path, repo: Path, engine: Path, manager: Path) -> dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("status") == "MANAGER_COMPLETED", "manifest is not manager-completed")
    require(manifest.get("manager_return_code") == 0, "manager return code is not zero")
    recorded_git_sha = manifest.get("current_git_sha")
    require(recorded_git_sha, "manifest is missing current_git_sha")
    require(source_matches_recorded_commit(repo, recorded_git_sha), "engine source changed after the recorded run")
    require(manifest.get("frozen_engine_git_sha") == FROZEN_ENGINE_GIT_SHA, "frozen engine Git SHA mismatch")

    engine_sha = sha256_file(engine)
    manager_sha = sha256_file(manager)
    require(engine_sha == FROZEN_ENGINE_SHA256, "current engine SHA differs from frozen binary")
    require(manager_sha == FROZEN_MANAGER_SHA256, "current manager SHA differs from frozen manager")
    require(manifest["engine"]["binary"]["sha256"] == engine_sha, "manifest engine hash mismatch")
    require(manifest["manager"]["sha256"] == manager_sha, "manifest manager hash mismatch")
    require(EXPECTED_MANAGER_VERSION in manifest["manager"]["version"], "manager version mismatch")

    candidate_identity = manifest["engine"]["candidate"]["identity"]
    baseline_identity = manifest["engine"]["baseline"]["identity"]
    require(candidate_identity["reported_profile"] == PROFILE_CANDIDATE_ARG, "candidate profile identity mismatch")
    require(baseline_identity["reported_profile"] == PROFILE_BASELINE_ARG, "baseline profile identity mismatch")
    require(candidate_identity["return_code"] == 0 and baseline_identity["return_code"] == 0, "UCI probe failed")

    match = manifest["match"]
    require(match == {
        "protocol": "uci",
        "time_control": EXPECTED_TIME_CONTROL,
        "hash_mb": EXPECTED_HASH_MB,
        "threads": EXPECTED_THREADS,
        "concurrency": EXPECTED_CONCURRENCY,
        "games_max": EXPECTED_GAMES_MAX,
        "candidate_first": True,
    }, f"match contract mismatch: {match!r}")
    require(manifest["sprt"] == {
        "subject": "CurrentFinal minus Current",
        "elo0": 20,
        "elo1": 60,
        "alpha": 0.05,
        "beta": 0.05,
    }, "SPRT contract mismatch")

    command_path = run_dir / "command.txt"
    command_text = command_path.read_text(encoding="utf-8").rstrip("\r\n")
    require(command_text == manifest["command_text"], "command.txt differs from manifest command_text")
    require(manifest["command"][0].lower().endswith("cutechess-cli.exe"), "command manager is not cutechess-cli")

    selection_path = repo / "tests" / "data" / "openings" / "s3-promotion-openings-v1.json"
    runtime_openings_path = run_dir / "openings.epd"
    require(sha256_file(selection_path) == manifest["opening"]["selection_manifest_sha256"], "selection manifest hash mismatch")
    require(sha256_file(runtime_openings_path) == manifest["opening"]["runtime_sha256"], "runtime opening hash mismatch")
    openings = load_runtime_openings(runtime_openings_path)

    pgn_path = run_dir / "match.pgn"
    games_list = read_games(pgn_path)
    games = len(games_list)
    require(games > 0 and games % 2 == 0, "completed match must contain nonzero complete pairs")
    require(games <= EXPECTED_GAMES_MAX, "PGN exceeds the configured game limit")
    pair_count = games // 2
    expected_keys = [position_key(fen) for fen in openings[:pair_count]]

    counts: Counter[str] = Counter()
    colors: Counter[str] = Counter()
    observed_keys: list[str] = []
    for pair_index in range(pair_count):
        first = validate_game(games_list[2 * pair_index], 2 * pair_index + 1)
        second = validate_game(games_list[2 * pair_index + 1], 2 * pair_index + 2)
        first_key, first_color, first_result = first
        second_key, second_color, second_result = second
        require(position_key(first_key) == position_key(second_key), f"pair {pair_index + 1}: FEN mismatch")
        observed_keys.append(position_key(first_key))
        require({first_color, second_color} == {"White", "Black"}, f"pair {pair_index + 1}: color swap missing")
        counts[candidate_result(first_color, first_result)] += 1
        counts[candidate_result(second_color, second_result)] += 1
        colors[first_color] += 1
        colors[second_color] += 1
    require(observed_keys == expected_keys, "PGN opening order does not match the runtime opening prefix")

    stdout_path = run_dir / "manager.stdout.log"
    stderr_path = run_dir / "manager.stderr.log"
    manager_summary = validate_manager_logs(stdout_path, stderr_path, games)
    require(manager_summary["manager_score"]["candidate_wins"] == counts["win"], "manager candidate wins mismatch")
    require(manager_summary["manager_score"]["baseline_wins"] == counts["loss"], "manager baseline wins mismatch")
    require(manager_summary["manager_score"]["draws"] == counts["draw"], "manager draws mismatch")
    require(manager_summary["sprt_decision"] == "H1_ACCEPTED", "S3-PROMOTION did not accept H1")

    score = (counts["win"] + 0.5 * counts["draw"]) / games
    artifacts = {}
    for name in ("manifest.json", "command.txt", "openings.epd", "manager.stdout.log", "manager.stderr.log", "match.pgn"):
        path = run_dir / name
        artifacts[name] = {"path": str(path.resolve()), "sha256": sha256_file(path), "size": path.stat().st_size}
    return {
        "schema_version": 1,
        "status": "VERIFIED",
        "milestone": "S3-PROMOTION",
        "decision": manager_summary["sprt_decision"],
        "decision_source": "cutechess manager stdout",
        "games": games,
        "opening_pairs": pair_count,
        "candidate": PROFILE_CANDIDATE,
        "baseline": PROFILE_BASELINE,
        "candidate_wins": counts["win"],
        "candidate_losses": counts["loss"],
        "draws": counts["draw"],
        "candidate_score_percent": round(score * 100, 3),
        "candidate_color_counts": dict(colors),
        "manager": manager_summary,
        "provenance": {
            "run_git_sha": recorded_git_sha,
            "verification_git_sha": git_output(repo, "rev-parse", "HEAD"),
            "engine_sha256": engine_sha,
            "manager_sha256": manager_sha,
            "source_unchanged_since_run": True,
        },
        "artifacts": artifacts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("results/s3-promotion/run-001"))
    parser.add_argument("--engine", type=Path, default=Path("target/release/eureka.exe"))
    parser.add_argument(
        "--manager", type=Path, default=Path("tools/.cache/cutechess-1.5.1-win64/cutechess-cli.exe")
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    run_dir = (repo / args.run_dir).resolve() if not args.run_dir.is_absolute() else args.run_dir.resolve()
    engine = (repo / args.engine).resolve() if not args.engine.is_absolute() else args.engine.resolve()
    manager = (repo / args.manager).resolve() if not args.manager.is_absolute() else args.manager.resolve()
    output = (
        (repo / args.output).resolve()
        if args.output is not None and not args.output.is_absolute()
        else (args.output.resolve() if args.output is not None else run_dir / "verification.json")
    )
    try:
        summary = verify_artifact(run_dir, repo, engine, manager)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(
            f"S3-PROMOTION verification PASS: {summary['games']} games, "
            f"{summary['candidate_wins']}-{summary['candidate_losses']}-{summary['draws']}, "
            f"decision={summary['decision']}"
        )
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"S3-PROMOTION verification failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
