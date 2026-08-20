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
FINAL_SPLIT_TARGETS = {"train": 240_000, "validation": 30_000, "holdout": 30_000}
FINAL_PHASE_TARGETS = {"high": 75_000, "mid": 135_000, "low": 60_000, "zero": 30_000}
MIN_FAMILIES = 2

PHASE_WEIGHTS = {chess.KNIGHT: 1, chess.BISHOP: 1, chess.ROOK: 2, chess.QUEEN: 4}
PHASE_BUCKETS = {"high": (18, 24), "mid": (8, 17), "low": (1, 7), "zero": (0, 0)}
PHASE_QUOTAS = {"high": 0.25, "mid": 0.45, "low": 0.20, "zero": 0.10}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_source_catalog(dir_paths: list[Path]) -> dict[str, dict]:
    """Load one or more source catalogs into a single keyed catalog.

    Accepts BOTH schemas:
      - aggregate `source_manifest.json` (map: source_key -> {source_id,
        source_family, sha256, ...});
      - source-local `source-manifest.json` (single Lichess-style entry with
        source_id / source_family / pgn_sha256).

    Every PGN is verified against the ACTUAL FILE BYTES SHA-256 (the
    lichess `pgn_sha256` convention; for ASCII PGNs it equals the old
    text-based hash). Duplicate source keys OR duplicate source_ids across
    all catalogs fail closed.
    """
    catalog: dict[str, dict] = {}
    seen_source_ids: set[str] = set()

    def check_unique(name: str, entry: dict, dir_path: Path) -> None:
        if name in catalog:
            raise SystemExit(
                f"FAIL CLOSED: duplicate source key '{name}'")
        src_id = entry.get("source_id")
        if not src_id:
            raise SystemExit(f"FAIL CLOSED: source '{name}' has no source_id")
        if src_id in seen_source_ids:
            raise SystemExit(
                f"FAIL CLOSED: duplicate source_id '{src_id}' (key '{name}')")
        seen_source_ids.add(src_id)

    def verify_sha(pgn_path: Path, expected: str, name: str) -> None:
        if not pgn_path.is_file():
            raise SystemExit(
                f"FAIL CLOSED: source PGN missing {pgn_path}")
        actual = sha256_bytes(pgn_path.read_bytes())
        if actual != expected:
            raise SystemExit(
                f"FAIL CLOSED: source SHA mismatch {name}: "
                f"{actual[:12]} != {expected[:12]}")

    for dir_path in map(Path, dir_paths):
        agg = dir_path / "source_manifest.json"
        local = dir_path / "source-manifest.json"
        if agg.is_file():
            manifest = json.loads(agg.read_text(encoding="utf-8"))
            for name, entry in manifest.items():
                check_unique(name, entry, dir_path)
                verify_sha(dir_path / f"{name}.pgn", entry["sha256"], name)
                catalog[name] = dict(entry)
        elif local.is_file():
            entry = json.loads(local.read_text(encoding="utf-8"))
            name = entry.get("source_id") or entry.get("source_family")
            if not name:
                raise SystemExit(
                    f"FAIL CLOSED: source-local manifest in {dir_path} has "
                    f"no source_id/source_family")
            check_unique(name, entry, dir_path)
            verify_sha(dir_path / f"{entry['source_id']}.pgn",
                       entry["pgn_sha256"], name)
            catalog[name] = {**entry, "source_key": name,
                             "sha256": entry["pgn_sha256"]}
        else:
            raise SystemExit(
                f"FAIL CLOSED: no source manifest (aggregate or local) in "
                f"{dir_path}")
    return catalog


def family_of(src_id: str, source_manifest: dict, final_mode: bool = False) -> str:
    """FINAL contract: the family is a REQUIRED manifest field; the
    arena- prefix fallback exists only for legacy exploratory shards."""
    for src in source_manifest.values():
        if src.get("source_id") == src_id:
            fam = src.get("source_family")
            if fam is None and final_mode:
                raise ValueError(
                    f"source {src_id} has no source_family "
                    f"(REQUIRED in FINAL mode)")
            if fam:
                return fam
    if src_id.startswith("arena-"):
        return "arena"
    return src_id.split("-", 1)[0]


def phase_of(board: chess.Board) -> int:
    total = 0
    for sq, piece in board.piece_map().items():
        total += PHASE_WEIGHTS.get(piece.piece_type, 0)
    return min(24, total)


def bucket_of(phase: int) -> str:
    for name, (lo, hi) in PHASE_BUCKETS.items():
        if lo <= phase <= hi:
            return name
    return "mid"


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


def find_pgn(dir_paths: list[Path], source_name: str, src: dict) -> Path:
    candidates = [f"{source_name}.pgn", f"{src['source_id']}.pgn"]
    for dir_path in map(Path, dir_paths):
        for name in candidates:
            path = dir_path / name
            if path.is_file():
                return path
    raise SystemExit(
        f"FAIL CLOSED: cannot locate PGN for source '{source_name}' "
        f"(id {src['source_id']})")


def build(args) -> int:
    catalog = load_source_catalog(args.sources)
    sampling_version = args.sampling_version
    sampling_method = {
        1: "first_8_in_time_order",
        2: "deterministic_hash_top_k",
    }[sampling_version]

    out_dir = Path(args.out) / args.dataset_id
    if out_dir.exists():
        print(
            f"FAIL CLOSED: {out_dir} already exists; refusing to overwrite "
            f"(staging + atomic publish never clobbers an existing dataset)"
        )
        return 3

    records: list[dict] = []
    per_source: dict[str, int] = {}
    games_parsed = 0
    reject_stats = {
        "terminal": 0, "in_check": 0, "no_legal_move": 0,
        "missing_king": 0, "illegal_board": 0,
    }
    candidates_seen = 0

    # Deterministic source order (sorted keys for reproducibility).
    for source_name in sorted(catalog):
        src = catalog[source_name]
        pgn_path = Path(args.sources[0]) / f"{source_name}.pgn"
        # Locate the PGN: aggregate dirs name files by source key; source-local
        # dirs name them by source_id.
        if not pgn_path.is_file():
            pgn_path = find_pgn(args.sources, source_name, src)
        actual_sha = sha256_bytes(pgn_path.read_bytes())
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

    # --- canonical representation hash (concatenation of sorted JSON lines) ---
    canonical = "".join(
        json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records
    )
    dataset_sha = sha256_text(canonical)
    n = len(records)
    shard_size = max(1, math.ceil(n / 4))

    per_family: dict[str, int] = {}
    try:
        for r in records:
            fam = family_of(r["source_id"], catalog, final_mode=args.final_mode)
            per_family[fam] = per_family.get(fam, 0) + 1
    except ValueError as exc:
        print(f"FAIL CLOSED: {exc}")
        return 5
    largest_family = max(per_family.values()) if per_family else 0
    largest_share = largest_family / n if n else 0.0
    family_count = len(per_family)
    if family_count < MIN_FAMILIES:
        not_final_reason = (
            f"only {family_count} source family/families present; a second "
            f"independent source family is required for FINAL")
    elif largest_share > 0.70:
        not_final_reason = (
            f"largest source family share {largest_share:.2%} exceeds the "
            f"70% contract")
    elif n < TARGET:
        not_final_reason = (
            f"records_total {n} below the FINAL target {TARGET}")
    else:
        not_final_reason = "FINAL gate not enabled (--final-mode)"
    stats = {
        "dataset_id": args.dataset_id,
        "final": False,
        "final_target": TARGET,
        "not_final_reason": not_final_reason,
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
        "shards": [],
        "shard_hashes": [],
        "dataset_sha256": dataset_sha,
        "sources": {k: v["sha256"] for k, v in catalog.items()},
        "source_families": per_family,
        "largest_family_share": largest_share,
    }
    if args.enforce_family_mix and largest_share > 0.70:
        print(
            f"FAIL CLOSED: largest source family share {largest_share:.2%} "
            f"exceeds the 70% contract (families: {per_family})"
        )
        return 4

    # --- FINAL mode: exact split + phase targeting (deterministic) ---
    if args.final_mode:
        err = _final_target_and_write(records, stats, args, out_dir,
                                      shard_size, catalog)
        return 0 if err is None else err

    # --- non-final: build in staging, publish atomically after the gate ---
    staging = Path(args.out) / f".staging-{args.dataset_id}"
    import shutil
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    _write_dataset(records, stats, staging, shard_size)
    import os
    os.replace(staging, out_dir)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


def _verify_staged(staging: Path) -> int:
    """Full integrity verification of a staged FINAL dataset. At this point
    the dataset has NO labels yet, so the teacher/label checks are explicitly
    skipped (allow_unlabeled=True); every other integrity check runs."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "verify_dataset", str(Path(__file__).parent / "verify_dataset.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.verify(argparse.Namespace(dataset=str(staging),
                                         allow_unlabeled=True))


def _final_target_and_write(records, stats, args, out_dir, shard_size,
                            catalog) -> int | None:
    """FINAL 300k: exact per-split phase targets, gates, staging + atomic move."""
    err = _final_gates_pre(stats, args)
    if err:
        print(f"FAIL CLOSED (final-mode): {err}")
        return 5

    # deterministic per-split x phase-bucket downsample to the FINAL targets
    pools: dict[tuple[str, str], list[dict]] = {}
    for r in records:
        pools.setdefault((r["split"], bucket_of(r["phase"])), []).append(r)
    selected: list[dict] = []
    shortfalls: list[str] = []
    for split, split_want in FINAL_SPLIT_TARGETS.items():
        share = split_want / TARGET
        for bucket, bucket_want in FINAL_PHASE_TARGETS.items():
            want = round(bucket_want * share)
            pool = sorted(pools.get((split, bucket), []),
                          key=lambda r: r["position_id"])
            if len(pool) < want:
                shortfalls.append(f"{split}/{bucket}: pool {len(pool)} < {want}")
                continue
            selected.extend(pool[:want])
    if shortfalls:
        for msg in shortfalls:
            print(f"FAIL CLOSED (final-mode): pool shortfall {msg}")
        return 5

    # rebuild stats for the selected records
    stats = _stats_for(selected, stats, shard_size, catalog)
    err = _final_gates_post(stats, args)
    if err:
        print(f"FAIL CLOSED (final-mode): {err}")
        return 5
    stats["final"] = True
    stats.pop("not_final_reason", None)

    if out_dir.exists():
        print(f"FAIL CLOSED: FINAL target {out_dir} already exists; "
              f"refusing to overwrite")
        return 5
    staging = Path(str(out_dir) + ".staging")
    if staging.exists():
        import shutil
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    _write_dataset(selected, stats, staging, shard_size)

    # full verify of the staged dataset before the atomic move
    # (allow_unlabeled=True: no labels exist yet at this point)
    rc = _verify_staged(staging)
    if rc != 0:
        print("FAIL CLOSED: staged dataset failed verification")
        return 5
    import os
    os.replace(staging, out_dir)
    print(f"FINAL dataset installed at {out_dir}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return None


def _write_dataset(records, stats, target_dir, shard_size):
    n = len(records)
    shard_hashes = []
    for i in range(0, n, shard_size):
        part = records[i:i + shard_size]
        lines = "".join(
            json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in part)
        shard_path = target_dir / f"part-{i // shard_size:04d}.jsonl"
        shard_path.write_text(lines, encoding="utf-8")
        shard_hashes.append(sha256_text(lines))
    stats["shards"] = [f"part-{i:04d}.jsonl"
                       for i in range(math.ceil(n / shard_size))]
    stats["shard_hashes"] = shard_hashes
    stats["dataset_sha256"] = sha256_text("".join(
        json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n"
        for r in records))
    (target_dir / "dataset_manifest.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")


def _stats_for(records, base, shard_size, source_manifest):
    n = len(records)
    per_family: dict[str, int] = {}
    for r in records:
        fam = family_of(r["source_id"], source_manifest)
        per_family[fam] = per_family.get(fam, 0) + 1
    largest_share = max(per_family.values()) / n if per_family else 0.0
    buckets = {name: 0 for name in PHASE_BUCKETS}
    for r in records:
        buckets[bucket_of(r["phase"])] += 1
    splits = {s: 0 for s in ("train", "validation", "holdout")}
    for r in records:
        splits[r["split"]] += 1
    stats = dict(base)
    stats.update({
        "records_total": n,
        "splits": splits,
        "phase_buckets": buckets,
        "source_families": per_family,
        "largest_family_share": largest_share,
        "shards": [],
        "shard_hashes": [],
    })
    return stats


def _final_gates_pre(stats, args):
    if args.sampling_version != 2:
        return "FINAL requires --sampling-version 2"
    if args.dataset_id != "s6-eval-v1-core-300k":
        return (f"FINAL requires --dataset-id s6-eval-v1-core-300k, "
                f"got {args.dataset_id}")


def _final_gates_post(stats, args):
    if stats["records_total"] != TARGET:
        return f"records_total {stats['records_total']} != {TARGET}"
    for split, want in FINAL_SPLIT_TARGETS.items():
        got = stats["splits"][split]
        if got != want:
            return f"split {split}={got} != {want}"
    for bucket, want in FINAL_PHASE_TARGETS.items():
        got = stats["phase_buckets"][bucket]
        if got != want:
            return f"phase bucket {bucket}={got} != {want}"
    if len(stats["source_families"]) < MIN_FAMILIES:
        return (f"only {len(stats['source_families'])} families "
                f"(need >= {MIN_FAMILIES})")
    if stats["largest_family_share"] > 0.70:
        return f"largest family share {stats['largest_family_share']:.2%} > 70%"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", required=True, nargs="+",
                        help="one or more source dirs; each holds either an "
                             "aggregate source_manifest.json (map: key -> "
                             "{source_id, source_family, sha256, ...}) or a "
                             "source-local source-manifest.json (Lichess "
                             "style) with pgn_sha256")
    parser.add_argument("--sampling-version", type=int, default=2,
                        choices=(1, 2),
                        help="1 = first-8-in-time-order (frozen shard01); "
                             "2 = deterministic hash top-K (FINAL contract)")
    parser.add_argument("--final-mode", action="store_true",
                        help="FINAL 300k contract: sampling v2, exact "
                             "splits/phase targets, >=2 families, family "
                             "share <=70%, staging + atomic move")
    parser.add_argument("--enforce-family-mix", action="store_true",
                        help="fail closed when the largest source family "
                             "exceeds 70% (FINAL builds)")
    parser.add_argument("--dataset-id", required=True,
                        help="dataset identity (e.g. s6-eval-v1-core-300k); "
                             "REQUIRED so a run can never silently write "
                             "into an unrelated/frozen dataset dir")
    parser.add_argument("--out", default="data/s6")
    return build(parser.parse_args(sys.argv[1:]))


if __name__ == "__main__":
    raise SystemExit(main())
