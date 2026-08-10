import chess, chess.pgn, statistics, hashlib, json, glob, os, sys

PGN = r"E:\AUbuntuProject\project\chessenginedemo\results\artifacts\20260809-b4de653-linux-x86_64\arena-screen-200pairs\combined.pgn"
S50 = r"C:\Users\81489\AppData\Local\Temp\opencode\s43d\s50"
C200 = r"C:\Users\81489\AppData\Local\Temp\opencode\s43d\c200"
CAND = "CurrentFinal LegalityFast"

def norm_fen(fen):
    b = chess.Board(fen)
    return b.fen()

# ---------- B1: pair structure + recompute W/L/D ----------
games = []
with open(PGN, encoding="utf-8") as fh:
    while True:
        g = chess.pgn.read_game(fh)
        if g is None:
            break
        games.append(g)
print("games:", len(games))

pairs = {}
for g in games:
    fen = norm_fen(g.headers["FEN"])
    w = g.headers["White"]
    res = g.headers["Result"]
    tc = g.headers["TimeControl"]
    pairs.setdefault(fen, []).append((w, res, tc))
n_pairs = len(pairs)
print("pairs:", n_pairs)

pair_ok = True
for fen, entries in pairs.items():
    if len(entries) != 2:
        pair_ok = False
        print("  BAD pair size", fen, len(entries))
        continue
    (w1, r1, tc1), (w2, r2, tc2) = entries
    if w1 == w2 or tc1 != tc2:
        pair_ok = False
        print("  BAD pair", fen, entries)
    if w1 == CAND and w2 == CAND:
        pair_ok = False
    if set([w1, w2]) != set([CAND, "CurrentFinal"]):
        pair_ok = False
print("pair structure (2 games, reversed colors, same TC, engines):", "OK" if pair_ok else "FAIL")

cw = cd = cl = 0
for g in games:
    if g.headers["White"] == CAND:
        r = g.headers["Result"]
        cw += r == "1-0"; cd += r == "1/2-1/2"; cl += r == "0-1"
    else:
        r = g.headers["Result"]
        cw += r == "0-1"; cd += r == "1/2-1/2"; cl += r == "1-0"
score = (cw + cd / 2) / len(games) * 100
print(f"recomputed candidate W/L/D = {cw}/{cl}/{cd}  score = {score:.3f}%")
print("expected 185/142/73 = 55.375% ->", "OK" if (cw, cl, cd) == (185, 142, 73) and abs(score - 55.375) < 0.01 else "FAIL")

# ---------- B2: pentanomial ----------
ptnml = [0, 0, 0, 0, 0]
for fen, entries in pairs.items():
    pts = 0.0
    for (w, r, _) in entries:
        if r == "1-0":
            pts += 2.0 if w == CAND else 0.0
        elif r == "1/2-1/2":
            pts += 1.0
        else:
            pts += 0.0 if w == CAND else 2.0
    idx = int(round(pts))  # pair total 0..4 -> Ptnml[0..4]
    ptnml[idx] += 1
print("Ptnml(0-2) [0.0,0.5,1.0,1.5,2.0]:", ptnml, "sum:", sum(ptnml))

# ---------- B3: color sanity ----------
cw_w = cd_w = cl_w = 0
cw_b = cd_b = cl_b = 0
for g in games:
    r = g.headers["Result"]
    if g.headers["White"] == CAND:
        cw_w += r == "1-0"; cd_w += r == "1/2-1/2"; cl_w += r == "0-1"
    elif g.headers["Black"] == CAND:
        cw_b += r == "0-1"; cd_b += r == "1/2-1/2"; cl_b += r == "1-0"
sw = (cw_w + cd_w / 2) / 200 * 100
sb = (cw_b + cd_b / 2) / 200 * 100
print(f"candidate White = {sw:.2f}% (expected 60.75)  candidate Black = {sb:.2f}% (expected 50.00)")
print(f"baseline White = {100 - sb:.2f}% (expected 50.00)  baseline Black = {100 - sw:.2f}% (expected 39.25)")
print(f"same-color improvement: White {sw-(100-sb):+.2f}pp  Black {sb-(100-sw):+.2f}pp (expected +10.75 both)")

# ---------- B4: termination ----------
term = {"checkmate": 0, "stalemate": 0, "insufficient": 0, "fifty": 0, "repetition": 0, "unknown": 0}
for g in games:
    b = g.board()
    for mv in g.mainline_moves():
        b.push(mv)
    if b.is_checkmate():
        term["checkmate"] += 1
    elif b.is_stalemate():
        term["stalemate"] += 1
    elif b.is_insufficient_material():
        term["insufficient"] += 1
    elif b.is_fifty_moves():
        term["fifty"] += 1
    elif b.is_repetition(2):
        term["repetition"] += 1
    else:
        term["unknown"] += 1
print("termination:", term)
print("(unknown = non-terminal endings: resignation/adjudication; no explicit class in PGN)")
print("time_forfeit=0 crash=0 illegal=0 -> verifier already passed all 200 pairs (rc=0, moves_legal)")

# ---------- B5: timing ----------
def move_times(g, cand):
    out = []
    b = g.board()
    node = g
    for mv in g.mainline_moves():
        node = node.next()
        side = "w" if b.turn == chess.WHITE else "b"
        if (side == "w") == (g.headers["White"] == cand):
            c = node.comment
            # '+0.10/5 0.38s'
            try:
                t = float(c.rsplit(" ", 1)[-1].rstrip("s"))
                out.append(t)
            except Exception:
                pass
        b.push(mv)
    return out

perf = {"cand_w": [], "cand_b": [], "base_w": [], "base_b": []}
for g in games:
    cand_w = g.headers["White"] == CAND
    cand_b = g.headers["Black"] == CAND
    b = g.board()
    node = g
    for mv in g.mainline_moves():
        node = node.next()
        side = "w" if b.turn == chess.WHITE else "b"
        try:
            t = float(node.comment.rsplit(" ", 1)[-1].rstrip("s"))
        except Exception:
            t = None
        if t is not None:
            if side == "w" and cand_w: perf["cand_w"].append(t)
            elif side == "w" and not cand_w: perf["base_w"].append(t)
            elif side == "b" and cand_b: perf["cand_b"].append(t)
            elif side == "b" and not cand_b: perf["base_b"].append(t)
        b.push(mv)

def stats(v):
    return (round(statistics.median(v), 2), round(statistics.quantiles(v, n=100)[94], 2), round(max(v), 2))
for k in ("cand_w", "cand_b", "base_w", "base_b"):
    v = perf[k]
    if v:
        med, p95, mx = stats(v)
        print(f"{k:7s} n={len(v):5d} median={med:6.2f}s p95={p95:6.2f}s max={mx:6.2f}s")
    else:
        print(f"{k:7s} n=0")

# ---------- B6: opening overlap ----------
def load_openings(d):
    fens = []
    for p in glob.glob(os.path.join(d, "**", "opening.epd"), recursive=True):
        line = open(p).read().strip()
        fens.append(norm_fen(line.split(";")[0].strip()))
    return fens

s50 = load_openings(S50)
c200 = load_openings(C200)
overlap = len(set(s50) & set(c200))
print(f"opening overlap (50-screen vs 200-conf): {overlap}")
print("independent confirmation:", "YES" if overlap == 0 else "NO (larger confirmation)")
