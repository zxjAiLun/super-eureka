#!/usr/bin/env python3
"""S15 expand: convert ADDITIONAL Fishtest tests to position shards.

Reuses the frozen S14 parsing semantics verbatim by importing
`s14_convert.convert_games` (single semantic source: same COMMENT_RE,
same Base-side fail-closed detection, same one-sample-per-game
deterministic hash). What changes is only the SCOPE:

  * tests already consumed by the S14 round are skipped (read from
    `--exclude-file`, produced by s15_extract_used_tests.py), so we pay
    the PGN parse cost only for genuinely new data;
  * the target is the number of NEW positions (default 16,000,000, which
    makes the S15 pool 4,220,410 + 16,000,000 ~= 20.2M);
  * output goes to a separate directory so the frozen S14 shards are
    never overwritten.

Everything else (which tests, in what order, which move per game) is
deterministic and identical to S14.

Usage:
  python tools/s15/s15_convert.py --target 16000000 --workers 16
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, r"tools\s14")
import s14_convert  # noqa: E402  (frozen parsing semantics)

SHARD_SIZE = 25_000


def extract_test_ids(shard_glob: str) -> tuple[set[str], list[str]]:
    """Test ids already present in existing shards (for restart safety).

    Returns (ids, unreadable). A shard can be unreadable when it was
    truncated by a kill mid-write; it is reported rather than crashed on,
    because a missing id only means that test is converted again -- and the
    encoder's exact-FEN dedup removes any resulting duplicates.
    """
    ids: set[str] = set()
    unreadable: list[str] = []
    for sp in sorted(glob.glob(shard_glob)):
        try:
            with gzip.open(sp, "rt", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        ids.add(json.loads(line)["test_id"])
        except (EOFError, OSError, gzip.BadGzipFile):
            unreadable.append(Path(sp).name)
    return ids, unreadable


def _worker(wid: int, tests: list, limit: int, out_dir: Path,
            shared, lock, shard_size: int, report_every: float = 120.0):
    stats = {"games_seen": 0, "accepted": 0, "parse_error_games": 0,
             "replay_error_games": 0, "games_base_unknown": 0,
             "games_no_eligible": 0}
    # The caller already handed this worker its round-robin slice.
    my_tests = tests
    # CRITICAL: shard names must never collide across runs. Restarting this
    # script previously reused indices from 0 and silently OVERWROTE shards
    # written by an earlier run (observed: on-disk records dropped). Start
    # after the highest index already present for this worker.
    existing = glob.glob(str(out_dir / f"shard-s15-w{wid}-*.jsonl.gz"))
    shard_idx = 0
    for e in existing:
        stem = Path(e).name[: -len(".jsonl.gz")]
        try:
            shard_idx = max(shard_idx, int(stem.rsplit("-", 1)[1]) + 1)
        except (IndexError, ValueError):
            continue
    if existing:
        print(f"[w{wid}] resuming shard numbering at {shard_idx} "
              f"({len(existing)} existing shards for this worker)", flush=True)
    buf: list[dict] = []
    local_accepted = 0
    tests_done = 0
    t0 = time.time()
    last_report = t0

    def flush():
        nonlocal shard_idx, buf
        if not buf:
            return
        path = out_dir / f"shard-s15-w{wid}-{shard_idx:05d}.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            for r in buf:
                fh.write(json.dumps(r, separators=(",", ":")) + "\n")
        shard_idx += 1
        buf = []

    for pgn_s, test_id, resolved_base in my_tests:
        with lock:
            if shared["accepted"] >= limit:
                break
        meta = {"test_id": test_id, "resolved_base": resolved_base}
        for rec in s14_convert.convert_games(Path(pgn_s), meta, stats):
            buf.append(rec)
            local_accepted += 1
            if len(buf) >= shard_size:
                flush()
            with lock:
                shared["accepted"] = shared["accepted"] + 1
                total = shared["accepted"]
                if total >= limit:
                    break
            # Periodic progress: never wait for a test boundary, since a
            # single Fishtest test can hold tens of thousands of games.
            if local_accepted % 500 == 0:
                now = time.time()
                if now - last_report >= report_every:
                    el = now - t0
                    print(f"[w{wid}] local={local_accepted} "
                          f"tests={tests_done} global={total}/{limit} "
                          f"elapsed={el:.0f}s "
                          f"rate={total/max(el,1):.0f}/s", flush=True)
                    last_report = now
        flush()
        tests_done += 1
        with lock:
            shared["tests_used"] = shared["tests_used"] + 1
            for k in ("games_seen", "parse_error_games",
                      "replay_error_games", "games_base_unknown",
                      "games_no_eligible"):
                shared[k] = shared[k] + stats[k]
            stats = {k2: 0 for k2 in stats}
            total = shared["accepted"]
        now = time.time()
        if now - last_report >= report_every / 2:
            el = now - t0
            print(f"[w{wid}] local={local_accepted} tests={tests_done} "
                  f"global={total}/{limit} elapsed={el:.0f}s "
                  f"rate={total/max(el,1):.0f}/s (test boundary)", flush=True)
            last_report = now
    flush()
    print(f"[w{wid}] DONE local={local_accepted} tests={tests_done}",
          flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", type=int, default=16_000_000)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", type=Path, default=Path(r"data\s15\shards"))
    ap.add_argument("--exclude-file", type=Path,
                    default=Path(r"data\s15\used-test-ids.json"))
    ap.add_argument("--shard-size", type=int, default=SHARD_SIZE)
    ap.add_argument("--also-exclude-shards", type=str, default=None,
                    help="glob of existing S15 shards whose test_ids must "
                         "not be re-converted (restart safety)")
    args = ap.parse_args()

    exclude: set[str] = set()
    if args.exclude_file.exists():
        exclude = set(json.loads(args.exclude_file.read_text(encoding="utf-8")))
    if args.also_exclude_shards:
        extra, unreadable = extract_test_ids(args.also_exclude_shards)
        print(f"[s15] existing shards add {len(extra - exclude)} test ids "
              f"(unreadable shards: {len(unreadable)})", flush=True)
        if unreadable:
            print(f"[s15] unreadable: {unreadable[:5]}", flush=True)
        exclude |= extra
    print(f"[s15] excluding {len(exclude)} already-used tests", flush=True)

    tests = []
    skipped_used = skipped_nometa = 0
    for test_id, pgn, js, meta in s14_convert.iter_tests_newest_first():
        if meta is None:
            skipped_nometa += 1
            continue
        if meta["test_id"] in exclude:
            skipped_used += 1
            continue
        tests.append((str(pgn), meta["test_id"], meta["resolved_base"]))
    print(f"[s15] candidate new tests={len(tests)} "
          f"(skipped used={skipped_used}, no-meta={skipped_nometa})",
          flush=True)

    args.out.mkdir(parents=True, exist_ok=True)

    manager = mp.Manager()
    shared = manager.dict({
        "accepted": 0, "games_seen": 0, "parse_error_games": 0,
        "replay_error_games": 0, "games_base_unknown": 0,
        "games_no_eligible": 0, "tests_used": 0})
    lock = manager.Lock()

    t0 = time.time()
    procs = [mp.Process(target=_worker, args=(
        w, tests[w::args.workers], args.target, args.out, shared, lock,
        args.shard_size)) for w in range(args.workers)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()

    stats = dict(shared)
    stats["elapsed_s"] = round(time.time() - t0, 1)
    stats["throughput_games_per_s"] = round(
        stats["games_seen"] / max(stats["elapsed_s"], 0.1), 1)
    stats["target"] = args.target
    stats["workers"] = args.workers
    stats["excluded_tests"] = len(exclude)
    stats["candidate_new_tests"] = len(tests)
    print(json.dumps(stats, indent=1), flush=True)
    (args.out.parent / "s15-convert-stats.json").write_text(
        json.dumps(stats, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
