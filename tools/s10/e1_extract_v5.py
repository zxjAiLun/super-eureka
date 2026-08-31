"""S10-E1: production v5 extraction from the local 2026-07 archive.

Uses the header-gated fast selector (e1_fast_select_lib, equivalence
proven 300/300 vs the original loop at 64x) over the locally verified
July 2026 lichess standard-rated archive.

v5 contract (S10-E1 zero-phase expansion):
  * all selected games are LONG games (>= 100 plies) — long_fraction
    1.0, long_min_plies 100 — to maximize endgame (zero-phase) yield;
  * every game in the B1 sources v1/confirm/v2/v3/v4 is excluded by
    game fingerprint;
  * selection seed 20260830, accept_byte 0x05 (same hash gate as the
    original tool; the selection ORDER therefore matches what the
    original tool would have produced);
  * target 200,000 games (>= the ~67k zero-phase shortfall with a wide
    margin; extra non-zero positions are simply not used by the nested
    builder, which fills high/mid/low ONLY from the old pool);
  * output: data/s6/sources/lichess-standard-rated-v5/<id>.pgn +
    source-manifest.json in the same schema as v4's manifest.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tools.s6.lichess_select import (  # noqa: E402
    game_fingerprint,
    fingerprint_set_sha256,
)
from tools.s10.e1_fast_select_lib import (  # noqa: E402
    count_plies,
    full_parse,
    header_gate,
    parse_headers,
    stream_games_raw,
)

ZST = Path(r"E:\ubuntudownload\lichess_db_standard_rated_2026-07.pgn.zst")
ZST_SHA = (
    "68738b1c448f051dc8d42db645d5b01749988a3bc1c24981adfe44ea92060dc7"
)
SEED = 20260830
ACCEPT_BYTE = 0x1F
MIN_PLIES = 40
LONG_PLIES = 100
TARGET = 350_000
SOURCE_ID = "lichess-standard-rated-v5"
OUT = Path("data/s6/sources/lichess-standard-rated-v5")
EXCLUDE = [
    "data/s6/sources/lichess-standard-rated-v1/lichess-standard-rated-v1.pgn",
    "data/s6/sources/lichess-standard-rated-confirm-v1-g1400/"
    "lichess-standard-rated-confirm-v1-g1400.pgn",
    "data/s6/sources/lichess-standard-rated-v2/lichess-standard-rated-v2.pgn",
    "data/s6/sources/lichess-standard-rated-v3/lichess-standard-rated-v3.pgn",
    "data/s6/sources/lichess-standard-rated-v4/lichess-standard-rated-v4.pgn",
]


def main() -> int:
    # 0. verify the local archive's SHA (already done once; re-verify
    #    cheaply against the recorded value by full re-hash — 27GB,
    #    ~90s, worth it for provenance).
    print("hashing local archive...", flush=True)
    h = hashlib.sha256()
    with open(ZST, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    actual = h.hexdigest()
    if actual != ZST_SHA:
        print(f"FAIL CLOSED: archive SHA {actual} != official {ZST_SHA}")
        return 4
    print("archive SHA verified (official lichess 2026-07)", flush=True)

    # 1. exclusion fingerprints from the B1 sources
    import chess.pgn

    excluded: set[str] = set()
    exclude_sources = []
    for p in EXCLUDE:
        path = Path(p)
        games = 0
        with open(path, encoding="utf-8", errors="replace") as fh:
            while True:
                game = chess.pgn.read_game(fh)
                if game is None:
                    break
                excluded.add(game_fingerprint(game))
                games += 1
        print(f"exclude {path.name}: {games} games", flush=True)
        exclude_sources.append(
            {
                "path": p,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "games": games,
            }
        )

    # 2. selection over the local archive (fast path)
    staging = Path(str(OUT) + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    pgn_path = staging / f"{SOURCE_ID}.pgn"

    t0 = time.time()
    selected = 0
    seen = 0
    excluded_hits = 0
    dup_hits = 0
    selected_fingerprints: set[str] = set()
    with open(pgn_path, "w", encoding="utf-8") as fh:
        for header_block, movetext in stream_games_raw(ZST):
            seen += 1
            if seen % 1_000_000 == 0:
                print(
                    f"  {seen} games scanned, {selected} selected, "
                    f"{time.time() - t0:.0f}s",
                    flush=True,
                )
            headers = parse_headers(header_block)
            key = header_gate(headers, ACCEPT_BYTE, SEED)
            if key is None:
                continue
            # long-game requirement: >= LONG_PLIES plies
            if count_plies(movetext) < LONG_PLIES:
                continue
            game = full_parse(header_block, movetext)
            real_plies = len(list(game.mainline_moves()))
            if real_plies < LONG_PLIES:
                continue
            fp = game_fingerprint(game)
            if fp in excluded:
                excluded_hits += 1
                continue
            if fp in selected_fingerprints:
                dup_hits += 1
                continue
            selected_fingerprints.add(fp)
            exporter = chess.pgn.StringExporter(columns=None)
            fh.write(game.accept(exporter))
            fh.write("\n\n")
            selected += 1
            if selected >= TARGET:
                break

    dt = time.time() - t0
    print(
        f"selected {selected} games from {seen} scanned in "
        f"{dt / 60:.1f} min",
        flush=True,
    )
    if selected < TARGET:
        print(
            f"FAIL CLOSED: only {selected} < {TARGET} selected; staging "
            "NOT published",
            flush=True,
        )
        return 4

    # 3. manifest (v4 schema)
    import chess as _chess

    manifest = {
        "source_family": "lichess-standard-rated-v1",
        "source_id": SOURCE_ID,
        "license": "CC0",
        "upstream": [
            "https://database.lichess.org/standard/"
            "lichess_db_standard_rated_2026-07.pgn.zst"
        ],
        "official_sha256": {"2026-07": ZST_SHA},
        "local_archive": str(ZST),
        "local_archive_sha256_verified": actual,
        "selector": "e1_fast_select (header-gated; equivalence "
                    "300/300 vs lichess_select.py at 64x)",
        "selection_seed": SEED,
        "filters": {
            "database": "standard rated (all standard-chess time controls)",
            "event_filter": "none (TimeControl is the speed gate)",
            "elo_min": 1800,
            "no_bot_titles": True,
            "time_control_base_min_sec": 180,
            "mainline_plies_min": MIN_PLIES,
            "long_stratum_plies_min": LONG_PLIES,
            "long_fraction": 1.0,
            "accept_byte": ACCEPT_BYTE,
        },
        "games_selected": selected,
        "selection": (
            f"hash(GameURL, seed) first byte < 0x{ACCEPT_BYTE:02x}, "
            "ALL selected games are long (>= 100 plies)"
        ),
        "fingerprint": {
            "definition": "sha256(canonical JSON with sorted keys and no "
                          "whitespace)",
            "fields": ["initial_fen", "result", "moves"],
            "initial_fen_default": _chess.STARTING_FEN,
            "moves_encoding": "mainline UCI",
        },
        "exclude_sources": exclude_sources,
        "exclude_fingerprint_count": len(excluded),
        "exclude_fingerprints_sha256": fingerprint_set_sha256(excluded),
        "selected_fingerprint_count": len(selected_fingerprints),
        "selected_fingerprints_sha256": fingerprint_set_sha256(
            selected_fingerprints
        ),
        "excluded_candidates_skipped": excluded_hits,
        "duplicate_candidates_rejected": dup_hits,
        "pgn_sha256": hashlib.sha256(pgn_path.read_bytes()).hexdigest(),
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (staging / "source-manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    shutil.move(str(staging), str(OUT))
    print(json.dumps({k: manifest[k] for k in (
        "source_id", "games_selected", "pgn_sha256",
        "excluded_candidates_skipped")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
