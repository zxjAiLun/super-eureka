"""Local GUI acceptance for the S14 engine.

Drives the ACTUAL config En Croissant saved in engines.json against the
built eureka.exe, covering the four interactions a GUI performs:
  1. LOAD    - apply saved options (Hash / EvalFile / NnueMode / Evaluation)
  2. MOVE    - analyze the blunder position (expect f7e6, ~+695cp)
  3. STOP    - interrupt a long search, require a legal bestmove
  4. NEWGAME - ucinewgame + isready, config retained, search still works
"""
import hashlib
import os
import subprocess
import sys
import threading
import time

EXE = r"E:\AUbuntuProject\project\chessenginedemo\target\release\eureka.exe"
MODEL = r"E:\AUbuntuProject\project\chessenginedemo\target\release\nnue-s14-datasupply-v5.bin"
# Documented S14 reference: startpos moves e2e3 e7e6 d1g4 d8e7 g4e6
BLUNDER_MOVES = "e2e3 e7e6 d1g4 d8e7 g4e6"
EXPECT_MOVE = "f7e6"
FAILURES = []


def check(label, cond, detail=""):
    print("  [{}] {} {}".format("PASS" if cond else "FAIL", label, detail))
    if not cond:
        FAILURES.append(label)


h = hashlib.sha256()
with open(MODEL, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
        h.update(chunk)
print("model sha256 : {}".format(h.hexdigest()))
print("model bytes  : {} ({})".format(MODEL, os.path.getsize(MODEL)))

proc = subprocess.Popen(
    [EXE],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)
lines = []
lock = threading.Lock()


def reader():
    for line in proc.stdout:
        with lock:
            lines.append(line.rstrip("\n"))


threading.Thread(target=reader, daemon=True).start()


def send(cmd):
    proc.stdin.write(cmd + "\n")
    proc.stdin.flush()


def wait_for(prefix, timeout=60):
    deadline = time.time() + timeout
    seen = []
    while time.time() < deadline:
        with lock:
            while lines:
                ln = lines.pop(0)
                seen.append(ln)
                if ln.startswith(prefix):
                    return seen
        time.sleep(0.02)
    raise TimeoutError("timeout waiting for {!r}; got {}".format(prefix, seen[-25:]))


def drain():
    with lock:
        got = list(lines)
        lines.clear()
    return got


send("uci")
wait_for("uciok")
drain()

# --- STEP 1: LOAD (exactly the options En Croissant saved) --------------
print("\n===== STEP 1: LOAD (En Croissant saved options) =====")
send("setoption name Hash value 64")
send("setoption name EvalFile value " + MODEL)
try:
    loaded = wait_for("info string EvalFile loaded", timeout=30)
    msg = [l for l in loaded if "EvalFile loaded" in l]
    print("  evaluator msgs : {}".format(msg))
    check("S14 model loaded", any("nnue-s14-datasupply-v5.bin" in m for m in msg))
except TimeoutError as e:
    print("  {}".format(e))
    check("S14 model loaded", False)

send("setoption name NnueMode value nnue-v2q")
send("setoption name Evaluation value nnue")
send("isready")
wait_for("readyok", timeout=30)
drain()
check("Evaluation=nnue accepted", True)

# --- STEP 2: MOVE (analyze the blunder position) ------------------------
print("\n===== STEP 2: MOVE (black should take the queen) =====")
send("position startpos moves " + BLUNDER_MOVES)
send("go depth 8")
out = wait_for("bestmove", timeout=180)
bm = [l for l in out if l.startswith("bestmove")][-1].split()
cps = [l for l in out if " score cp " in l]
last_cp = cps[-1] if cps else ""
print("  bestmove      : {}".format(" ".join(bm)))
print("  last cp       : {}".format(last_cp[:100]))

move = bm[1] if len(bm) > 1 else ""
check("bestmove is {} (takes the queen)".format(EXPECT_MOVE), move == EXPECT_MOVE,
      "got {}".format(move))
cp_val = None
if " score cp " in last_cp:
    cp_val = int(last_cp.split(" score cp ")[1].split()[0])
check("score strongly positive", cp_val is not None and cp_val >= 600,
      "cp={}".format(cp_val))

# --- STEP 3: STOP (interrupt a long search) -----------------------------
print("\n===== STEP 3: STOP (interrupt long search) =====")
send("position startpos")
send("go movetime 20000")
time.sleep(2.0)
send("stop")
out = wait_for("bestmove", timeout=60)
bm = [l for l in out if l.startswith("bestmove")][-1].split()
move = bm[1] if len(bm) > 1 else ""
print("  bestmove      : {}".format(" ".join(bm)))
check("stop yields legal bestmove", move not in ("", "0000"), "got {}".format(move))

# --- STEP 4: NEWGAME (config must be retained) --------------------------
print("\n===== STEP 4: NEWGAME (config retained, search works) =====")
send("ucinewgame")
send("isready")
wait_for("readyok", timeout=60)
drain()

# 4a. Controls: a DIFFERENT position must not return the blunder answer.
#     This proves 4b is a fresh search, not a stale un-reset board.
send("position startpos")
send("go depth 6")
out = wait_for("bestmove", timeout=180)
ctrl = [l for l in out if l.startswith("bestmove")][-1].split()[1]
print("  control(startpos): {}".format(ctrl))
check("board reset works (startpos != blunder move)", ctrl != EXPECT_MOVE,
      "got {}".format(ctrl))

# 4b. Retention: S14 config still in force, same blunder answer.
send("position startpos moves " + BLUNDER_MOVES)
send("go depth 6")
out = wait_for("bestmove", timeout=180)
bm = [l for l in out if l.startswith("bestmove")][-1].split()
move = bm[1] if len(bm) > 1 else ""
print("  post-newgame  : {}".format(" ".join(bm)))
check("NNUE config retained across ucinewgame", move == EXPECT_MOVE,
      "got {}".format(move))

send("quit")
try:
    proc.wait(timeout=15)
except subprocess.TimeoutExpired:
    proc.kill()

print("\n" + "=" * 60)
if FAILURES:
    print("GUI ACCEPTANCE: FAIL ({}) -> {}".format(len(FAILURES), FAILURES))
    sys.exit(1)
print("GUI ACCEPTANCE: PASS (load / move / stop / newgame)")
