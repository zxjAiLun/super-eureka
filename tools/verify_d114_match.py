#!/usr/bin/env python3
"""Verify a D1.14 Cute Chess match without reimplementing SPRT.

The verifier checks the artifacts produced by ``run_d114_sprt.py`` and by
Cutechess: provenance, binary identity, opening coverage, profile roles,
pair/color balance, legal PGN replay, manager failure diagnostics, and the
candidate-oriented descriptive result.  Cutechess remains the authority for
the SPRT decision; this tool only maps its explicit H0/H1 boundary output.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

import chess
import chess.pgn


BASELINE_PROFILE = "current"
CANDIDATE_PROFILE = "current-lmr"
BASELINE_LABEL = "Current"
CANDIDATE_LABEL = "CurrentLmr"
VALID_RESULTS = {"1-0", "0-1", "1/2-1/2"}
MANAGER_FINISHED_RE = re.compile(
    r"Finished game (?P<number>\d+) \([^)]*\): "
    r"(?P<result>1-0|0-1|1/2-1/2) \{(?P<reason>[^}]*)\}"
)


class D114VerificationError(RuntimeError):
    """Raised when a D1.14 artifact fails an integrity gate."""


@dataclass(frozen=True)
class VerifiedGame:
    number: int
    fen: str
    white: str
    black: str
    result: str
    candidate_score: float
    natural_terminal: bool
    manager_reason: str


def position_key(fen: str) -> str:
    """Use the board/side/castling/EP fields; Cutechess resets move clocks in PGN."""
    fields = fen.split()
    require(len(fields) in {4, 6}, f"not a four/six-field FEN: {fen!r}")
    return " ".join(fields[:4])


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise D114VerificationError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise D114VerificationError(f"JSON root is not an object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise D114VerificationError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise D114VerificationError(message)


def resolve_artifact(run_dir: Path, relative: str) -> Path:
    path = (run_dir / relative).resolve()
    require(path.is_relative_to(run_dir.resolve()), f"artifact escapes run directory: {relative}")
    return path


def load_opening_lines(manifest: dict[str, Any], run_dir: Path) -> list[str]:
    opening = manifest.get("opening")
    require(isinstance(opening, dict), "manifest opening section is missing")
    runtime_path = Path(str(opening.get("runtime_path", ""))).expanduser()
    if not runtime_path.is_file():
        runtime_path = resolve_artifact(run_dir, runtime_path.name)
    require(runtime_path.is_file(), f"opening runtime file is missing: {runtime_path}")
    expected_hash = str(opening.get("runtime_sha256", ""))
    actual_hash = sha256_file(runtime_path)
    require(actual_hash == expected_hash, "opening runtime SHA-256 mismatch")
    lines = [line.strip() for line in runtime_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    require(len(lines) == int(opening.get("runtime_count", -1)), "opening runtime count mismatch")
    for index, line in enumerate(lines, start=1):
        fields = line.split()
        require(len(fields) == 6, f"opening line {index} is not a six-field FEN")
        try:
            board = chess.Board(line)
        except ValueError as exc:
            raise D114VerificationError(f"invalid opening FEN at line {index}: {exc}") from exc
        require(board.is_valid(), f"invalid opening position at line {index}")
        require(
            not board.is_game_over(claim_draw=False),
            f"terminal opening position at line {index}",
        )
    return lines


def verify_opening_hash_artifact(manifest: dict[str, Any], run_dir: Path) -> None:
    path = resolve_artifact(run_dir, "openings.sha256")
    value = read_json(path)
    opening = manifest["opening"]
    require(value.get("runtime_path") == opening.get("runtime_path"), "opening hash path mismatch")
    require(value.get("runtime_sha256") == opening.get("runtime_sha256"), "opening hash value mismatch")
    require(value.get("runtime_count") == opening.get("runtime_count"), "opening hash count mismatch")
    require(value.get("suite") == opening.get("suite_metadata"), "opening suite metadata mismatch")


def verify_binary_artifact(manifest: dict[str, Any], run_dir: Path) -> None:
    binary = manifest.get("engine", {}).get("binary")
    require(isinstance(binary, dict), "manifest engine binary is missing")
    binary_path = Path(str(binary.get("path", ""))).expanduser()
    require(binary_path.is_file(), f"engine binary is missing: {binary_path}")
    actual_hash = sha256_file(binary_path)
    require(actual_hash == binary.get("sha256"), "engine binary SHA-256 mismatch")
    artifact = read_json(resolve_artifact(run_dir, "binary.sha256"))
    require(artifact.get("binary") == binary, "binary.sha256 does not match manifest")
    engines = manifest.get("engine", {})
    require(
        engines.get("baseline", {}).get("identity", {}).get("reported_profile") == BASELINE_PROFILE,
        "baseline manifest identity is not current",
    )
    require(
        engines.get("candidate", {}).get("identity", {}).get("reported_profile") == CANDIDATE_PROFILE,
        "candidate manifest identity is not current-lmr",
    )


def parse_manager_reasons(run_dir: Path) -> dict[int, tuple[str, str]]:
    path = resolve_artifact(run_dir, "manager.log.gz")
    require(path.is_file(), "manager.log.gz is missing")
    try:
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as source:
            text = source.read()
    except OSError as exc:
        raise D114VerificationError(f"cannot read manager.log.gz: {exc}") from exc
    stderr_marker = "=== stderr ==="
    if stderr_marker in text:
        stderr = text.split(stderr_marker, 1)[1].strip()
        allowed_warnings = {
            "Warning: 2 opening repetitions vs 1 games per encounter",
        }
        unexpected = [line.strip() for line in stderr.splitlines() if line.strip() not in allowed_warnings]
        require(not unexpected, f"cutechess stderr contains unexpected diagnostics: {unexpected[:3]}")
    reasons: dict[int, tuple[str, str]] = {}
    for match in MANAGER_FINISHED_RE.finditer(text):
        number = int(match.group("number"))
        require(number not in reasons, f"duplicate manager result for game {number}")
        reasons[number] = (match.group("result"), match.group("reason").strip())
    return reasons


def candidate_score(result: str, white: str) -> float:
    if result == "1/2-1/2":
        return 0.5
    candidate_won = (white == CANDIDATE_LABEL and result == "1-0") or (
        white == BASELINE_LABEL and result == "0-1"
    )
    return 1.0 if candidate_won else 0.0


def terminal_winner_matches(board: chess.Board, result: str) -> bool:
    if board.is_checkmate():
        winner = chess.BLACK if board.turn == chess.WHITE else chess.WHITE
        return result == ("1-0" if winner == chess.WHITE else "0-1")
    if board.is_stalemate() or board.is_insufficient_material():
        return result == "1/2-1/2"
    return True


def validate_game(
    game: chess.pgn.Game,
    number: int,
    expected_fens: set[str],
    reasons: dict[int, tuple[str, str]],
) -> VerifiedGame:
    headers = game.headers
    fen = headers.get("FEN", "")
    white = headers.get("White", "")
    black = headers.get("Black", "")
    result = headers.get("Result", "")
    fen_key = position_key(fen)
    require(fen_key in expected_fens, f"game {number} has an unknown opening FEN")
    require({white, black} == {BASELINE_LABEL, CANDIDATE_LABEL}, f"game {number} has wrong profile labels")
    require(white != black, f"game {number} has identical players")
    require(result in VALID_RESULTS, f"game {number} has invalid result {result!r}")
    try:
        board = chess.Board(fen)
    except ValueError as exc:
        raise D114VerificationError(f"game {number} has invalid FEN: {exc}") from exc
    require(board.is_valid(), f"game {number} FEN is not a valid position")
    for ply, move in enumerate(game.mainline_moves(), start=1):
        require(board.is_legal(move), f"game {number} has illegal move at ply {ply}: {move}")
        board.push(move)
    manager_result, reason = reasons.get(number, ("", ""))
    require(manager_result == result, f"game {number} PGN result disagrees with manager output")
    lower_reason = reason.lower()
    failure_words = ("illegal", "crash", "disconnect", "aborted", "abandon", "failed", "error")
    require(
        not any(word in lower_reason for word in failure_words),
        f"game {number} manager termination is a failure: {reason}",
    )
    manager_draw_terminal = any(
        word in lower_reason
        for word in ("3-fold repetition", "threefold", "fifty moves", "75 moves", "insufficient mating")
    )
    natural_terminal = board.is_game_over(claim_draw=False) or manager_draw_terminal
    if natural_terminal:
        require(terminal_winner_matches(board, result), f"game {number} result disagrees with terminal board")
    else:
        require(
            any(word in lower_reason for word in ("time", "forfeit")),
            f"game {number} is non-terminal without a time-forfeit manager result",
        )
    return VerifiedGame(
        number=number,
        fen=fen_key,
        white=white,
        black=black,
        result=result,
        candidate_score=candidate_score(result, white),
        natural_terminal=natural_terminal,
        manager_reason=reason,
    )


def load_games(
    run_dir: Path,
    expected_fens: list[str],
    reasons: dict[int, tuple[str, str]],
) -> list[VerifiedGame]:
    path = resolve_artifact(run_dir, "match.pgn.gz")
    require(path.is_file(), "match.pgn.gz is missing")
    games: list[VerifiedGame] = []
    expected_set = {position_key(fen) for fen in expected_fens}
    try:
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as source:
            number = 0
            while True:
                game = chess.pgn.read_game(source)
                if game is None:
                    break
                number += 1
                require(not game.errors, f"game {number} PGN parse errors: {game.errors}")
                games.append(validate_game(game, number, expected_set, reasons))
    except (OSError, UnicodeError) as exc:
        raise D114VerificationError(f"cannot read match.pgn.gz: {exc}") from exc
    require(
        set(reasons) == set(range(1, len(games) + 1)),
        "manager output is missing or has extra finished-game records",
    )
    return games


def verify_pairs(
    games: list[VerifiedGame],
    expected_fens: list[str],
    expected_games: int,
    require_sequential_order: bool,
) -> None:
    require(len(games) == expected_games, f"expected {expected_games} PGN games, got {len(games)}")
    by_fen: dict[str, list[VerifiedGame]] = defaultdict(list)
    for game in games:
        by_fen[position_key(game.fen)].append(game)
    expected_keys = [position_key(fen) for fen in expected_fens]
    require(set(by_fen) == set(expected_keys), "PGN opening set does not match the manifest")
    require(len(by_fen) == len(expected_keys), "duplicate opening FEN in runtime suite")
    first_occurrence = []
    seen: set[str] = set()
    for game in games:
        game_key = position_key(game.fen)
        if game_key not in seen:
            first_occurrence.append(game_key)
            seen.add(game_key)
    if require_sequential_order:
        require(first_occurrence == expected_keys, "PGN opening order is not sequential")
    for fen in expected_keys:
        pair = by_fen[fen]
        require(len(pair) == 2, f"opening pair is incomplete for {fen}")
        require({(game.white, game.black) for game in pair} == {
            (CANDIDATE_LABEL, BASELINE_LABEL),
            (BASELINE_LABEL, CANDIDATE_LABEL),
        }, f"opening pair did not swap colors for {fen}")


def descriptive_statistics(games: list[VerifiedGame], mode: str) -> dict[str, Any]:
    wins = sum(game.candidate_score == 1.0 for game in games)
    draws = sum(game.candidate_score == 0.5 for game in games)
    losses = sum(game.candidate_score == 0.0 for game in games)
    total = len(games)
    score = (wins + 0.5 * draws) / total if total else 0.0
    elo: float | None = None
    elo_error_95: float | None = None
    if 0.0 < score < 1.0:
        import math

        elo = 400.0 * math.log10(score / (1.0 - score))
        standard_error = math.sqrt(score * (1.0 - score) / total)
        elo_error_95 = 1.96 * 400.0 / math.log(10.0) * standard_error / (score * (1.0 - score))
    penta = [0, 0, 0, 0, 0]
    by_fen: dict[str, list[VerifiedGame]] = defaultdict(list)
    for game in games:
        by_fen[game.fen].append(game)
    for pair in by_fen.values():
        if len(pair) == 2:
            penta[round(sum(game.candidate_score for game in pair) * 2)] += 1
    return {
        "games": total,
        "pairs": len(by_fen),
        "candidate": CANDIDATE_LABEL,
        "baseline": BASELINE_LABEL,
        "candidate_wins": wins,
        "draws": draws,
        "candidate_losses": losses,
        "candidate_score": score,
        "descriptive_elo": elo,
        "descriptive_elo_error_95": elo_error_95,
        "descriptive_only": True,
        "pentanomial_candidate_points_0_to_2": penta,
        "time_forfeit_games": sum(
            any(word in game.manager_reason.lower() for word in ("time", "forfeit"))
            for game in games
        ),
        "mode": mode,
    }


def manager_decision(manifest: dict[str, Any], run_dir: Path, games: list[VerifiedGame]) -> str:
    mode = manifest.get("mode")
    if mode == "Smoke":
        return "NOT_APPLICABLE"
    require(mode == "Sprt", f"unknown D1.14 mode {mode!r}")
    output_path = resolve_artifact(run_dir, "sprt-output.txt")
    require(output_path.is_file(), "sprt-output.txt is missing")
    output = output_path.read_text(encoding="utf-8", errors="replace")
    if re.search(r"SPRT: H1 was accepted", output):
        return "PASS"
    if re.search(r"SPRT: H0 was accepted", output):
        return "REJECTED"
    require(len(games) == int(manifest["match"]["games_max"]), "formal match stopped before SPRT boundary or game limit")
    return "INCONCLUSIVE"


def compare_existing_summary(path: Path, summary: dict[str, Any]) -> None:
    if not path.is_file():
        return
    old = read_json(path)
    keys = (
        "decision",
        "games",
        "pairs",
        "candidate_wins",
        "draws",
        "candidate_losses",
        "pentanomial_candidate_points_0_to_2",
    )
    for key in keys:
        require(old.get(key) == summary.get(key), f"existing summary mismatch for {key}")


def verify(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    require(run_dir.is_dir(), f"run directory does not exist: {run_dir}")
    manifest = read_json(run_dir / "manifest.json")
    require(manifest.get("milestone") == "D1.14", "not a D1.14 manifest")
    require(manifest.get("status") in {"MANAGER_COMPLETED", "VERIFIED"}, "match was not manager-completed")
    require(manifest.get("decision") in {"PENDING_VERIFICATION", "NOT_APPLICABLE", "PASS", "REJECTED", "INCONCLUSIVE"}, "invalid manifest decision")
    match = manifest.get("match")
    require(isinstance(match, dict), "manifest match section is missing")
    require(match.get("adjudication") == {"draw": False, "resign": False, "tablebase": False}, "adjudication is enabled")
    require(match.get("hash_mb") == 16, "D1.14 Hash must be 16 MB")
    require(match.get("threads") == 1, "D1.14 engine threads must be 1")
    require(manifest.get("sprt", {}).get("player_1") == "current-lmr", "SPRT player 1 is not candidate")
    require(manifest.get("sprt", {}).get("player_2") == "current", "SPRT player 2 is not baseline")
    verify_opening_hash_artifact(manifest, run_dir)
    expected_fens = load_opening_lines(manifest, run_dir)
    verify_binary_artifact(manifest, run_dir)
    reasons = parse_manager_reasons(run_dir)
    games = load_games(run_dir, expected_fens, reasons)
    verify_pairs(
        games,
        expected_fens,
        int(match["games_max"]),
        require_sequential_order=match.get("concurrency") == 1,
    )
    summary = descriptive_statistics(games, str(manifest["mode"]))
    summary["decision"] = manager_decision(manifest, run_dir, games)
    summary["integrity_status"] = "PASS"
    summary["pgn_status"] = "PASS"
    summary["profile_status"] = "PASS"
    summary["opening_pair_status"] = "PASS"
    summary_path = resolve_artifact(run_dir, "summary.json")
    compare_existing_summary(summary_path, summary)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest["status"] = "VERIFIED"
    manifest["decision"] = summary["decision"]
    manifest["verification"] = {
        "status": "PASS",
        "games": summary["games"],
        "pairs": summary["pairs"],
        "pgn": "match.pgn.gz",
        "verified_utc": datetime_now_utc(),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def datetime_now_utc() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("run_dir", type=Path)
    return command


def main() -> int:
    args = parser().parse_args()
    try:
        summary = verify(args.run_dir)
    except D114VerificationError as exc:
        print(f"INTEGRITY_FAIL: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
