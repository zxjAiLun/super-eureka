"""S10-I1-A0: sibling-ranking corpus builder.

Two-stage sibling labeling per parent:
  selector: SF18 MultiPV=8, nodes 4096  -> candidate 8 moves
  label:    SF18 nodes 32768, searchmoves <move> (one per candidate)

Parents: 50% ordinary train positions (one per game preferred),
50% Eureka search-site parents (>=2,500 distinct train roots, max 2
parents/root, 50k-node production searches, diagnostic capture build).

Siblings record: parent identity, source, root/game id, move, child
FEN, teacher cp (parent POV) / mate, actual nodes.

Usage:
    python tools/s10/i1_a0_corpus.py --phase smoke   # 64 parents
    python tools/s10/i1_a0_corpus.py --phase full    # 10,000 parents
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
import sys
import tempfile
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

SF = (r"C:\Users\81489\AppData\Local\Temp\opencode\sf18-win\stockfish"
      r"\stockfish-windows-x86-64-avx2.exe")
EUREKA_DIAG = None  # built per-run (needs the diagnostic capture build)
MODEL = (r"data\s10\e3\scale-1m-win\seed-20260820"
         r"\nnue-v2-q01-material-v3twin.bin")
DS = Path(r"data\s10\s10-eval-v2-1m01")
CACHE = Path(r"C:\Users\81489\AppData\Local\Temp\opencode\i1a-cache")

K_SIBLINGS = 8
SELECTOR_NODES = 4096
LABEL_NODES = 32768


def make_sf():
    p = subprocess.Popen([SF], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    def send(cmd):
        p.stdin.write(cmd + "\n"); p.stdin.flush()
    send("uci")
    while p.stdout.readline().strip() != "uciok":
        pass
    for name, value in (("Threads", "1"), ("Hash", "64")):
        send(f"setoption name {name} value {value}")
    return p, send


def select_moves(p, send, fen, k):
    """MultiPV-k selector at 4k nodes; returns k candidate moves."""
    send("ucinewgame")
    send(f"position fen {fen}")
    send(f"go nodes {SELECTOR_NODES}")
    moves = []
    while True:
        line = p.stdout.readline()
        if not line or line.startswith("bestmove"):
            break
        m = re.search(r" multipv (\d+) .* pv (\S+)", line)
        if m:
            moves.append((int(m.group(1)), m.group(2)))
    moves.sort()
    return [mv for _, mv in moves[:k]]


def label_move(p, send, fen, move):
    """Constrained 32k label for one move; score in PARENT POV."""
    send("ucinewgame")
    send(f"position fen {fen} moves {move}")
    send(f"go nodes {LABEL_NODES}")
    cp = mate = None
    while True:
        line = p.stdout.readline()
        if not line or line.startswith("bestmove"):
            break
        if not line.startswith("info") or " pv " not in line:
            continue
        m = re.search(r"score cp (-?\d+)|score mate (-?\d+)", line)
        if m:
            if m.group(1) is not None:
                cp = int(m.group(1))
            else:
                mate = int(m.group(2))
    # search runs AFTER the move: score is from the OPPONENT (child stm).
    # Parent POV = negate.
    if cp is not None:
        return {"cp": -cp, "mate": None}
    if mate is not None:
        return {"cp": None, "mate": -mate}
    return {"cp": None, "mate": None}


def load_exclusions():
    """Everything that must NOT leak into I1-A training."""
    h0c = Path(r"C:\Users\81489\AppData\Local\Temp\opencode\h0c-cache")
    excl_fens = set()
    excl_games = set()
    # H0-C validation roots + sites + E parents
    excl_fens |= set(json.load(open(h0c / "roots.json"))["root_fens"])
    excl_fens |= set(json.loads(l)["fen"] for l in
                     open(h0c / "labeled_rooted.jsonl", encoding="utf-8")
                     if l.strip())
    if (h0c / "e_parents.jsonl").exists():
        excl_fens |= set(json.loads(l)["fen"] for l in
                         open(h0c / "e_parents.jsonl", encoding="utf-8")
                         if l.strip())
    # H0-E sibling children (parent FENs already excluded; children too)
    if (h0c / "e_siblings.jsonl").exists():
        for l in open(h0c / "e_siblings.jsonl", encoding="utf-8"):
            if l.strip():
                p = json.loads(l)
                b = chess.Board(p["fen"])
                for sib in p["siblings"]:
                    b.push(chess.Move.from_uci(sib["uci"]))
                    excl_fens.add(b.fen())
                    b.pop()
    return excl_fens


def collect_search_parents(n_needed, seed, eureka_diag):
    """Search-site parents from many distinct train roots."""
    records = []
    for shard in sorted(DS.glob("part-*.jsonl")):
        for line in shard.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                if rec["split"] == "train":
                    records.append(rec)
    rng = random.Random(seed)
    pool = sorted(records, key=lambda r: r["position_id"])
    rng.shuffle(pool)
    excl = load_exclusions()
    parents = []
    per_root = defaultdict(int)
    t0 = time.time()
    for r in pool:
        if len(parents) >= n_needed:
            break
        if r["fen"] in excl:
            continue
        # capture sites from this root
        out = subprocess.run(
            [eureka_diag, "bench", "eval-site-capture", "--fen", r["fen"],
             "--nodes", "50000",
             "--profile", "current-final-nnue-v2q-material",
             "--nnue-model", MODEL, "--hash-mb", "32"],
            capture_output=True, text=True, timeout=1800, check=True)
        # candidate sites: prefer main_static; max 2 per root
        sites = []
        for line in out.stdout.splitlines():
            if line.startswith('{"fen"'):
                rec = json.loads(line)
                sites.append((rec["fen"], rec["site"]))
        rng.shuffle(sites)
        for fen, site in sites:
            if per_root[r["position_id"]] >= 2:
                break
            if fen in excl:
                continue
            b = chess.Board(fen)
            if len(list(b.legal_moves)) < 2:
                continue
            parents.append({"fen": fen, "source": "search_site",
                            "root_pid": r["position_id"],
                            "site_kind": site})
            per_root[r["position_id"]] += 1
            if len(parents) >= n_needed:
                break
        if (len(parents) % 1000) < 2:
            print(f"  search parents {len(parents)}/{n_needed} "
                  f"({time.time()-t0:.0f}s, "
                  f"{len(per_root)} roots)", flush=True)
    return parents, len(per_root)


def collect_ordinary_parents(n_needed, seed):
    """Ordinary train positions, one per game preferred."""
    records = []
    for shard in sorted(DS.glob("part-*.jsonl")):
        for line in shard.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                if rec["split"] == "train":
                    records.append(rec)
    rng = random.Random(seed)
    pool = sorted(records, key=lambda r: r["position_id"])
    rng.shuffle(pool)
    excl = load_exclusions()
    by_game = defaultdict(int)
    parents = []
    for r in pool:
        if len(parents) >= n_needed:
            break
        if r["fen"] in excl:
            continue
        if by_game[r["source_game_id"]] >= 1:
            continue
        b = chess.Board(r["fen"])
        if len(list(b.legal_moves)) < 2:
            continue
        parents.append({"fen": r["fen"], "source": "ordinary",
                        "game_id": r["source_game_id"]})
        by_game[r["source_game_id"]] += 1
    return parents


def label_parents(parents, n_workers=6):
    """Two-stage labeling with a thread-local SF process each."""
    local = threading.local()
    def get_sf():
        if getattr(local, "sf", None) is None:
            local.sf, local.send = make_sf()
        return local.sf, local.send

    def work(pi):
        p = parents[pi]
        sf, send = get_sf()
        cands = select_moves(sf, send, p["fen"], K_SIBLINGS)
        sibs = []
        b = chess.Board(p["fen"])
        for mv in cands:
            m = chess.Move.from_uci(mv)
            if m not in b.legal_moves:
                continue  # selector hallucination guard
            lab = label_move(sf, send, p["fen"], mv)
            b.push(m)
            sibs.append({"uci": mv, "child_fen": b.fen(), **lab})
            b.pop()
        p["siblings"] = sibs

    t0 = time.time()
    done = [0]
    def wrapped(pi):
        work(pi)
        done[0] += 1
        if done[0] % 64 == 0:
            rate = done[0] / (time.time() - t0)
            eta = (len(parents) - done[0]) / rate
            print(f"  labeled {done[0]}/{len(parents)} "
                  f"({rate:.1f} parents/s, ETA {eta/60:.0f}m)", flush=True)
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        list(ex.map(wrapped, range(len(parents))))
    return time.time() - t0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["smoke", "full"], required=True)
    args = parser.parse_args()
    CACHE.mkdir(exist_ok=True)

    if args.phase == "smoke":
        n = 64
        eureka = (r"C:\Users\81489\AppData\Local\Temp\opencode"
                  r"\eureka-diag.exe")
    else:
        n = 5000
        eureka = (r"C:\Users\81489\AppData\Local\Temp\opencode"
                  r"\eureka-diag.exe")

    print(f"=== I1-A0 {args.phase}: {n*2} parents ===", flush=True)
    ordinary = collect_ordinary_parents(n, seed=2026090701)
    print(f"ordinary parents: {len(ordinary)}", flush=True)
    search, n_roots = collect_search_parents(n, seed=2026090702,
                                              eureka_diag=eureka)
    print(f"search-site parents: {len(search)} from {n_roots} roots",
          flush=True)
    parents = ordinary + search

    wall = label_parents(parents)
    print(f"labeling wall: {wall:.0f}s "
          f"({len(parents)/wall:.1f} parents/s)", flush=True)

    # provenance
    ordered_sha = hashlib.sha256(
        "\n".join(sorted(p["fen"] for p in parents)).encode()).hexdigest()
    n_sibs = sum(len(p["siblings"]) for p in parents)
    print(f"total siblings: {n_sibs} | parent SHA {ordered_sha[:16]}")

    out = CACHE / (f"i1a_{'smoke' if args.phase=='smoke' else 'full'}.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for p in parents:
            f.write(json.dumps(p) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
