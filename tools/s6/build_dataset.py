#!/usr/bin/env python3
"""S6.0 Dataset builder: PGN sources -> deterministic evaluation dataset.

Rules (S6.0 spec):
- deterministic sampling: hash(game_id, ply, dataset_seed), min ply 12,
  max ply 160, max 8 positions per game;
- eligibility: legal position, both kings, non-terminal, side to move has a
  legal move, side to move NOT in check;
- identity: canonical FEN (first four fields) -> sha256 position_id;
- game-group split 80/10/10 by deterministic hash; every game belongs to
  exactly ONE split; cross-split FEN duplicates removed (keep the copy in
  the earliest game's split by deterministic order);
- phase = sum of piece weights (N/B=1, R=2, Q=4) clamped to 24;
- output: deterministic JSONL shards + manifest (canonical representation
  hash) + stats.

Dataset id: s6-eval-v1-core-shard01 (NOT the 300k FINAL; that is gated on a
second independent source per the S6.0 contract).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import chess
import chess.pgn

DATASET_ID = "s6-eval-v1-core-shard01"
SCHEMA_VERSION = 1
SAMPLING_VERSION = 2
SAMPLING_METHOD = "deterministic_hash_top_k"
DATASET_SEED = 20260812
MIN_PLY = 12
MAX_PLY = 160
MAX_PER_GAME = 8
TARGET = 300_000
PHASE_WEIGHTS = {chess.KNIGHT: 1, chess.BISHOP: 1, chess.ROOK: 2, chess.QUEEN: 4}
PHASE_BUCKETS = {"high": (18, 24), "mid": (8, 17), "low": (1, 7), "zero": (0, 0)}
PHASE_QUOTAS = {"high": 0.25, "mid": 0.45, "low": 0.20, "zero": 0.10}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def phase_of(board: chess.Board) -> int:
    total = 0
    for sq, piece in board.piece_map().items():
        total += PHASE_WEIGHTS.get(piece.piece_type, 0)
    return min(24, total)


def canonical_fen4(board: chess.Board) -> str:
    """FEN first four fields: piece placement, side, castling, en passant."""
    return board.fen().split(" ")[0] + " " + " ".join(board.fen().split(" ")[1:4])


def eligible(board: chess.Board) -> tuple[bool, str]:
    if not board.is_valid():
        return False, "illegal_board"
    if board.is_game_over(claim_draw=False):
        return False, "terminal"
    if board.is_check():
        return False, "in_check"
    if not any(board.legal_moves):
        return False, "no_legal_move"
    if board.king(chess.WHITE) is None or board.king(chess.BLACK) is None:
        return False, "missing_king"
    return True, ""


def sample_candidates(game_id: str, ply: int, seed: int) -> bool:
    h = hashlib.sha256(f"{game_id}:{ply}:{seed}".encode("utf-8")).digest()
    return h[0] < 0x80  # ~50% of plies sampled; top-K selection caps the rest


def ply_priority(game_id: str, ply: int, seed: int) -> int:
    """Sampling v2: deterministic priority for the whole-ply-range top-K.

    Every eligible ply in [MIN_PLY, MAX_PLY] gets a priority from
    sha256(game_id, ply, seed); the LOWEST-priority eligible positions
    (up to MAX_PER_GAME) are kept. This removes the v1 front-loading bias
    (time-order 'first 8' stopped at the opening/middlegame) so each game
    contributes positions across its whole span."""
    return int.from_bytes(
        hashlib.sha256(f"{game_id}:{ply}:{seed}".encode("utf-8")).digest()[:8],
        "big")


def game_split(game_id: str) -> str:
    h = int.from_bytes(
        hashlib.sha256(f"split:{game_id}".encode("utf-8")).digest()[:4], "big"
    )
    r = h % 1000
    if r < 800:
        return "train"
    if r < 900:
        return "validation"
    return "holdout"


def load_games(pgn_path: Path):
    with open(pgn_path, encoding="utf-8", errors="replace") as fh:
        while True:
            game = chess.pgn.read_game(fh)
            if game is None:
                return
            yield game


def game_result_white(game) -> float | None:
    r = game.headers.get("Result")
    return {"1-0": 1.0, "1/2-1/2": 0.5, "0-1": 0.0}.get(r)


def build(args) -> int:
    sources_dir = Path(args.sources)
    manifest_path = sources_dir / "source_manifest.json"
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sampling_version = args.sampling_version
    sampling_method = {
        1: "first_8_in_time_order",
        2: "deterministic_hash_top_k",
    }[sampling_version]

    out_dir = Path(args.out) / DATASET_ID
    out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    per_source: dict[str, int] = {}
    games_parsed = 0
    reject_stats = {
        "terminal": 0, "in_check": 0, "no_legal_move": 0,
        "missing_king": 0, "illegal_board": 0,
    }
    candidates_seen = 0

    # Deterministic source order (manifest key order is insertion order from
    # the server; sort for reproducibility).
    for source_name in sorted(source_manifest):
        src = source_manifest[source_name]
        pgn_path = sources_dir / f"{source_name}.pgn"
        actual_sha = sha256_text(pgn_path.read_text(encoding="utf-8", errors="replace"))
        if actual_sha != src["sha256"]:
            print(f"SOURCE SHA MISMATCH {source_name}: {actual_sha[:12]} != {src['sha256'][:12]}")
            return 2

        game_no = 0
        for game in load_games(pgn_path):
            game_no += 1
            games_parsed += 1
            game_id = f"{src['source_id']}:{game_no}"
            result = game_result_white(game)
            if result is None:
                continue
            moves = list(game.mainline_moves())
            if len(moves) < MIN_PLY:
                continue
            board = game.board()
            split = game_split(game_id)
            if sampling_version == 2:
                # Sampling v2: scan the WHOLE ply range, collect eligible
                # candidates with their priorities, keep the top-K (lowest
                # priorities) per game (deterministic). Removes the v1
                # front-loading bias.
                candidates: list[tuple[int, int, dict]] = []
                for ply in range(1, len(moves) + 1):
                    board.push(moves[ply - 1])
                    if ply < MIN_PLY or ply > MAX_PLY:
                        continue
                    candidates_seen += 1
                    ok, reason = eligible(board)
                    if not ok:
                        reject_stats[reason] += 1
                        continue
                    fen4 = canonical_fen4(board)
                    pos_id = sha256_text(fen4)
                    candidates.append((ply_priority(game_id, ply, DATASET_SEED),
                                       ply, {
                        "schema_version": SCHEMA_VERSION,
                        "position_id": pos_id,
                        "fen": board.fen(),
                        "canonical_fen4": fen4,
                        "source_id": src["source_id"],
                        "source_game_id": game_id,
                        "ply": ply,
                        "game_result_white": result,
                        "phase": phase_of(board),
                        "split": split,
                        "teacher_cp_stm": None,
                        "teacher_mate": None,
                        "teacher_bestmove": None,
                        "teacher_wdl_stm": None,
                    }))
                candidates.sort(key=lambda c: (c[0], c[1]))
                for _, _, rec in candidates[:MAX_PER_GAME]:
                    records.append(rec)
                per_source[source_name] = per_source.get(source_name, 0) + min(
                    len(candidates), MAX_PER_GAME)
            else:
                # Sampling v1 (frozen shard01 contract): time-order, stop at
                # the first MAX_PER_GAME eligible positions. Deterministic.
                taken = 0
                for ply in range(1, len(moves) + 1):
                    board.push(moves[ply - 1])
                    if ply < MIN_PLY or ply > MAX_PLY:
                        continue
                    if not sample_candidates(game_id, ply, DATASET_SEED):
                        continue
                    candidates_seen += 1
                    ok, reason = eligible(board)
                    if not ok:
                        reject_stats[reason] += 1
                        continue
                    if taken >= MAX_PER_GAME:
                        continue
                    fen4 = canonical_fen4(board)
                    pos_id = sha256_text(fen4)
                    records.append({
                        "schema_version": SCHEMA_VERSION,
                        "position_id": pos_id,
                        "fen": board.fen(),
                        "canonical_fen4": fen4,
                        "source_id": src["source_id"],
                        "source_game_id": game_id,
                        "ply": ply,
                        "game_result_white": result,
                        "phase": phase_of(board),
                        "split": split,
                        "teacher_cp_stm": None,
                        "teacher_mate": None,
                        "teacher_bestmove": None,
                        "teacher_wdl_stm": None,
                    })
                    taken += 1
                per_source[source_name] = per_source.get(source_name, 0) + taken
        print(f"{source_name}: {game_no} games -> {per_source.get(source_name, 0)} records")

    # --- dedup: keep ONE copy per position_id (deterministic order) ---
    before = len(records)
    records.sort(key=lambda r: (r["position_id"], r["source_game_id"]))
    unique: list[dict] = []
    seen: set[str] = set()
    for r in records:
        if r["position_id"] in seen:
            continue
        seen.add(r["position_id"])
        unique.append(r)
    dup_removed = before - len(unique)
    records = unique

    # --- phase-bucket stratification (quota-aware; deterministic) ---
    def bucket_of(phase: int) -> str:
        for name, (lo, hi) in PHASE_BUCKETS.items():
            if lo <= phase <= hi:
                return name
        return "mid"

    n_pre = len(records)
    targets = {
        name: round(n_pre * share) for name, share in PHASE_QUOTAS.items()
    }
    quota_pool = {name: [] for name in PHASE_BUCKETS}
    for r in records:
        quota_pool[bucket_of(r["phase"])].append(r)
    stratified: list[dict] = []
    for name in PHASE_BUCKETS:
        pool = sorted(quota_pool[name], key=lambda r: r["position_id"])
        stratified.extend(pool[: targets[name]])
    records = stratified

    # --- split stats + cross-split audit (AFTER stratification) ---
    by_split = {"train": [], "validation": [], "holdout": []}
    for r in records:
        by_split[r["split"]].append(r)

    # --- write shards ---
    n = len(records)
    shard_size = max(1, math.ceil(n / 4))
    shard_hashes: list[str] = []
    for i in range(0, n, shard_size):
        part = records[i:i + shard_size]
        lines = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in part)
        shard_path = out_dir / f"part-{i // shard_size:04d}.jsonl"
        shard_path.write_text(lines, encoding="utf-8")
        shard_hashes.append(sha256_text(lines))

    # --- canonical representation hash (concatenation of sorted JSON lines) ---
    canonical = "".join(
        json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records
    )
    dataset_sha = sha256_text(canonical)

    stats = {
        "dataset_id": DATASET_ID,
        "final": False,
        "final_target": TARGET,
        "not_final_reason": "single source family (Arena historical); "
                            "second independent source required for FINAL",
        "schema_version": SCHEMA_VERSION,
        "sampling_version": sampling_version,
        "sampling_method": sampling_method,
        "dataset_seed": DATASET_SEED,
        "min_ply": MIN_PLY,
        "max_ply": MAX_PLY,
        "max_per_game": MAX_PER_GAME,
        "records_total": n,
        "games_parsed": games_parsed,
        "candidates_seen": candidates_seen,
        "reject_stats": reject_stats,
        "duplicates_removed": dup_removed,
        "per_source": per_source,
        "splits": {s: len(by_split[s]) for s in by_split},
        "phase_buckets": {
            name: sum(1 for r in records if bucket_of(r["phase"]) == name)
            for name in PHASE_BUCKETS
        },
        "phase_quotas": PHASE_QUOTAS,
        "phase_targets": targets,
        "phase_quota_shortfall": {
            name: max(0, targets[name] - sum(1 for r in records
                                             if bucket_of(r["phase"]) == name))
            for name in PHASE_BUCKETS
        },
        "wdl": {
            "white_wins": sum(1 for r in records if r["game_result_white"] == 1.0),
            "draws": sum(1 for r in records if r["game_result_white"] == 0.5),
            "black_wins": sum(1 for r in records if r["game_result_white"] == 0.0),
        },
        "piece_counts": {
            name: sum(1 for r in records for sq, p in
                      chess.Board(r["fen"]).piece_map().items()
                      if p.piece_type == name)
            for name in (chess.PAWN, chess.KNIGHT, chess.BISHOP,
                         chess.ROOK, chess.QUEEN)
        },
        "side_to_move": {
            "white": sum(1 for r in records
                         if chess.Board(r["fen"]).turn == chess.WHITE),
            "black": sum(1 for r in records
                         if chess.Board(r["fen"]).turn == chess.BLACK),
        },
        "castling_rights": {
            "none": sum(1 for r in records
                        if chess.Board(r["fen"]).castling_rights == 0),
            "some": sum(1 for r in records
                        if chess.Board(r["fen"]).castling_rights != 0),
        },
        "shards": [f"part-{i:04d}.jsonl" for i in range(math.ceil(n / shard_size))],
        "shard_hashes": shard_hashes,
        "dataset_sha256": dataset_sha,
        "sources": {k: v["sha256"] for k, v in source_manifest.items()},
    }
    (out_dir / "dataset_manifest.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", required=True,
                        help="dir with PGNs + source_manifest.json")
    parser.add_argument("--sampling-version", type=int, default=2,
                        choices=(1, 2),
                        help="1 = first-8-in-time-order (frozen shard01); "
                             "2 = deterministic hash top-K (FINAL contract)")
    parser.add_argument("--out", default="data/s6")
    return build(parser.parse_args(sys.argv[1:]))


if __name__ == "__main__":
    raise SystemExit(main())
