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


def make_sf(multipv=1):
    p = subprocess.Popen([SF], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    def send(cmd):
        p.stdin.write(cmd + "\n"); p.stdin.flush()
    send("uci")
    while p.stdout.readline().strip() != "uciok":
        pass
    for name, value in (("Threads", "1"), ("Hash", "64"),
                        ("MultiPV", str(multipv))):
        send(f"setoption name {name} value {value}")
    return p, send


def select_moves(p, send, fen, k):
    """MultiPV=k selector at 4k nodes. Accepts ONLY the final complete
    MultiPV frame (deepest depth with all ranks 1..k present exactly
    once); fail-closes on any incomplete/duplicate frame."""
    send("ucinewgame")
    send(f"position fen {fen}")
    send(f"go nodes {SELECTOR_NODES}")
    frames = {}  # depth -> {rank: move}
    while True:
        line = p.stdout.readline()
        if not line or line.startswith("bestmove"):
            break
        m = re.search(r"info depth (\d+) .*?multipv (\d+) .* pv (\S+)",
                      line)
        if m:
            depth, rank, move = (int(m.group(1)), int(m.group(2)),
                                 m.group(3))
            frames.setdefault(depth, {})[rank] = move
    # final frame = the deepest depth having ALL ranks 1..k exactly once;
    # PV heads may repeat across ranks (converging lines) — dedupe keeping
    # the highest-ranked occurrence.
    for depth in sorted(frames, reverse=True):
        fr = frames[depth]
        if sorted(fr.keys()) != list(range(1, k + 1)):
            continue
        moves = []
        for r in range(1, k + 1):
            mv = fr[r]
            if mv not in moves:
                moves.append(mv)
        if not moves:
            raise SystemExit(
                f"FAIL CLOSED: empty selector frame at depth {depth} "
                f"for {fen}")
        return moves
    raise SystemExit(
        f"FAIL CLOSED: no complete MultiPV-{k} frame for {fen} "
        f"(frames: {[(d, sorted(f.keys())) for d, f in sorted(frames.items())][-3:]})")


def label_move(p, send, fen, move):
    """Constrained 32k label: `go nodes 32768 searchmoves <move>` from the
    PARENT position (the frozen I1-A contract). The returned score is the
    parent-position search score with the move forced first — already in
    the PARENT's POV, no negation."""
    send("ucinewgame")
    send(f"position fen {fen}")
    send(f"go nodes {LABEL_NODES} searchmoves {move}")
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
    if cp is not None:
        return {"cp": cp, "mate": None}
    if mate is not None:
        return {"cp": None, "mate": mate}
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
    """Two-stage labeling with thread-local SF processes: one MultiPV=8
    selector process and one MultiPV=1 labeler per thread."""
    local = threading.local()
    def get_selector():
        if getattr(local, "sel", None) is None:
            local.sel, local.sel_send = make_sf(multipv=K_SIBLINGS)
        return local.sel, local.sel_send
    def get_labeler():
        if getattr(local, "lab", None) is None:
            local.lab, local.lab_send = make_sf(multipv=1)
        return local.lab, local.lab_send

    def work(pi):
        p = parents[pi]
        b = chess.Board(p["fen"])
        k = min(K_SIBLINGS, len(list(b.legal_moves)))
        sel_p, sel_send = get_selector()
        cands = select_moves(sel_p, sel_send, p["fen"], k)
        # fail-close: every selector move must be legal
        for mv in cands:
            if chess.Move.from_uci(mv) not in b.legal_moves:
                raise SystemExit(
                    f"FAIL CLOSED: selector returned illegal move {mv} "
                    f"for {p['fen']}")
        lab_p, lab_send = get_labeler()
        sibs = []
        for mv in cands:
            lab = label_move(lab_p, lab_send, p["fen"], mv)
            m = chess.Move.from_uci(mv)
            b.push(m)
            sibs.append({"uci": mv, "child_fen": b.fen(),
                         "selector_rank": cands.index(mv) + 1, **lab})
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

    # Repair 1: REUSE the exact original parent corpus (same parents,
    # same SHA) — only the sibling selection/labeling is redone.
    src = (r"C:\Users\81489\AppData\Local\Temp\opencode\i1a-cache"
           r"\i1a_full.jsonl")
    all_parents = [json.loads(l) for l in open(src, encoding="utf-8")
                   if l.strip()]
    if args.phase == "smoke":
        rng = random.Random(2026090801)
        parents = rng.sample(all_parents, 128)
        out_name = "i1a_smoke_r1"
    else:
        parents = all_parents
        out_name = "i1a_full_r1"
    n_roots = len({p.get("root_pid") for p in parents
                   if p["source"] == "search_site"})
    print(f"=== I1-A Repair1 {args.phase}: {len(parents)} parents "
          f"({n_roots} roots, REUSED) ===", flush=True)
    # drop old siblings
    for p in parents:
        p.pop("siblings", None)

    wall = label_parents(parents)
    print(f"labeling wall: {wall:.0f}s "
          f"({len(parents)/wall:.1f} parents/s)", flush=True)

    # provenance
    ordered_sha = hashlib.sha256(
        "\n".join(sorted(p["fen"] for p in parents)).encode()).hexdigest()
    n_sibs = sum(len(p["siblings"]) for p in parents)
    print(f"total siblings: {n_sibs} | parent SHA {ordered_sha[:16]}")

    out = CACHE / f"{out_name}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for p in parents:
            f.write(json.dumps(p) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
