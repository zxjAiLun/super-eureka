"""S10-E2-W0: run the corpus on Linux + Windows x2 and evaluate the gates.

Three runs, identical per-position contract:
    L  = Linux 6b087694... (one temporary WSL process; VM shut down after)
    W1 = Windows C86215FA... (independent process)
    W2 = Windows C86215FA... (second independent process)

Each: Threads=1 Hash=64 MultiPV=1 UCI_ShowWDL=true, ucinewgame per
position, go nodes 16384.

Gate 1 (Windows self-determinism, STRICT): W1 vs W2 must be
2048/2048 exact on bestmove / cp / mate / WDL / nodes.

Gate 2 (cross-platform semantics, NOT letter-exact):
    bestmove agreement          >= 99.5%
    mate/non-mate class         100%
    cp-vs-cp: median |d| = 0, p95 <= 1, p99 <= 3, max <= 10
    |mean signed d|             <= 0.25 cp
    sign flips at |cp| >= 50    0
    WDL exact triplet           >= 99%

Usage:
    python tools/s10/e2_w0_run.py --corpus results/s10/s10-e2-w0-corpus.jsonl \
        --out results/s10/s10-e2-w0-report.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import threading
import queue
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

LINUX_BIN = "/home/sparkle/sf18"
LINUX_SHA = ("6b087694916228c905a5e14db74cca8c7e5643602226af1fa5d4235"
             "3c455b9f9")
WIN_EXE = (r"C:\Users\81489\AppData\Local\Temp\opencode\sf18-win\stockfish"
           r"\stockfish-windows-x86-64-avx2.exe")
WIN_SHA = ("c86215fa1977d53b82ed854540a4c7b025be4cd042276c85ba3de53fb911"
           "8911")
TEACHER_NODES = 16384

FIELDS = ("teacher_cp_stm", "teacher_mate", "teacher_bestmove",
          "teacher_wdl_stm", "nodes")


class NativeTeacher:
    """Windows-native SF18 process (same UCI discipline as Teacher)."""

    def __init__(self, path: str, expected_sha: str):
        import hashlib
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        sha = h.hexdigest()
        if sha != expected_sha:
            raise RuntimeError(f"binary SHA {sha} != {expected_sha}")
        self.proc = subprocess.Popen(
            [path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1)
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._reader = threading.Thread(target=self._pump, daemon=True)
        self._reader.start()
        self.send("uci")
        lines = self._read_until("uciok")
        self.uci_id_name = next(
            (l[len("id name"):] for l in lines
             if l.startswith("id name")), "unknown").strip()
        self.send("isready")
        self._expect("readyok")
        for name, value in (("Threads", "1"), ("Hash", "64"),
                            ("MultiPV", "1"), ("UCI_ShowWDL", "true")):
            self.send(f"setoption name {name} value {value}")

    def _pump(self):
        try:
            while True:
                line = self.proc.stdout.readline()
                if not line:
                    self._queue.put(None)
                    return
                self._queue.put(line)
        except Exception:
            self._queue.put(None)

    def send(self, cmd: str):
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def _read_until(self, marker: str, timeout: float = 60.0) -> list[str]:
        lines = []
        deadline = time.time() + timeout
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise RuntimeError(f"timeout waiting {marker}")
            try:
                line = self._queue.get(timeout=remaining)
            except queue.Empty:
                raise RuntimeError(f"timeout waiting {marker}") from None
            if line is None:
                raise RuntimeError("teacher process ended")
            line = line.strip()
            lines.append(line)
            if line == marker or line.startswith(marker):
                return lines

    def _expect(self, marker: str):
        self._read_until(marker)

    def label(self, fen: str) -> dict:
        self.send("ucinewgame")
        self.send(f"position fen {fen}")
        self.send(f"go nodes {TEACHER_NODES}")
        lines = self._read_until("bestmove", timeout=300)
        final_info: dict = {}
        for line in lines:
            if not line.startswith("info") or " pv " not in line:
                continue
            m = re.search(r"score cp (-?\d+)|score mate (-?\d+)", line)
            wdl = re.search(r"wdl (\d+) (\d+) (\d+)", line)
            pv = re.search(r" pv (\S+)", line)
            if m:
                final_info["cp"] = (int(m.group(1))
                                    if m.group(1) is not None else None)
                final_info["mate"] = (int(m.group(2))
                                      if m.group(2) is not None else None)
            if wdl:
                final_info["wdl"] = tuple(int(wdl.group(i)) for i in (1, 2, 3))
            if pv:
                final_info["bestmove"] = pv.group(1)
        best = lines[-1].split("bestmove ")[1].split(" ")[0]
        return {
            "teacher_cp_stm": final_info.get("cp"),
            "teacher_mate": final_info.get("mate"),
            "teacher_bestmove": final_info.get("bestmove", best),
            "teacher_wdl_stm": list(final_info.get(
                "wdl", [None, None, None])),
            "nodes": TEACHER_NODES,
        }

    def close(self):
        try:
            self.send("quit")
        except Exception:
            pass
        try:
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()


class WslTeacher(NativeTeacher):
    """Linux teacher through a WSL bridge (temporary; VM shut down after)."""

    def __init__(self):
        sha_out = subprocess.run(
            ["wsl.exe", "-e", "bash", "-c",
             f"sha256sum {LINUX_BIN}"],
            capture_output=True, text=True, timeout=120).stdout
        sha = sha_out.split()[0]
        if sha != LINUX_SHA:
            raise RuntimeError(f"linux SHA {sha} != {LINUX_SHA}")
        self.proc = subprocess.Popen(
            ["wsl.exe", "-e", "bash", "-lc", LINUX_BIN],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1)
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._reader = threading.Thread(target=self._pump, daemon=True)
        self._reader.start()
        self.send("uci")
        lines = self._read_until("uciok")
        self.uci_id_name = next(
            (l[len("id name"):] for l in lines
             if l.startswith("id name")), "unknown").strip()
        self.send("isready")
        self._expect("readyok")
        for name, value in (("Threads", "1"), ("Hash", "64"),
                            ("MultiPV", "1"), ("UCI_ShowWDL", "true")):
            self.send(f"setoption name {name} value {value}")


def run_corpus(teacher, fens: list[str]) -> dict[str, dict]:
    out = {}
    for i, fen in enumerate(fens):
        out[fen] = teacher.label(fen)
        if (i + 1) % 512 == 0:
            print(f"  {i + 1}/{len(fens)}", flush=True)
    return out


def pct(sorted_vals, p):
    n = len(sorted_vals)
    return sorted_vals[min(n - 1, int(round(p / 100 * (n - 1))))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    entries = [json.loads(l) for l in
               args.corpus.read_text(encoding="utf-8").splitlines() if l.strip()]
    fens = [e["fen"] for e in entries]
    print(f"corpus: {len(fens)} positions")

    print("=== L (linux) ===", flush=True)
    lt = WslTeacher()
    L = run_corpus(lt, fens)
    lt.close()

    print("=== W1 (windows) ===", flush=True)
    w1 = NativeTeacher(WIN_EXE, WIN_SHA)
    W1 = run_corpus(w1, fens)
    w1.close()

    print("=== W2 (windows, independent) ===", flush=True)
    w2 = NativeTeacher(WIN_EXE, WIN_SHA)
    W2 = run_corpus(w2, fens)
    w2.close()

    # ---- Gate 1: Windows self-determinism (STRICT) ------------------------
    g1_mism = []
    for fen in fens:
        a, b = W1[fen], W2[fen]
        if any(a.get(f) != b.get(f) for f in FIELDS):
            g1_mism.append(fen)
    gate1 = {
        "checked": len(fens),
        "mismatches": len(g1_mism),
        "pass": len(g1_mism) == 0,
    }
    print(f"Gate1 windows self-determinism: {len(fens) - len(g1_mism)}"
          f"/{len(fens)} {'PASS' if gate1['pass'] else 'FAIL'}")

    # ---- Gate 2: cross-platform semantics ---------------------------------
    bm_agree = 0
    class_mism = []
    cp_diffs = []
    signed = []
    sign_flips = []
    wdl_agree = 0
    tuple_exact = 0
    for fen in fens:
        l, w = L[fen], W1[fen]
        if l["teacher_bestmove"] == w["teacher_bestmove"]:
            bm_agree += 1
        l_mate = l["teacher_mate"] is not None
        w_mate = w["teacher_mate"] is not None
        if l_mate != w_mate:
            class_mism.append(fen)
        if not l_mate and not w_mate:
            d = w["teacher_cp_stm"] - l["teacher_cp_stm"]
            cp_diffs.append(abs(d))
            signed.append(d)
            if abs(l["teacher_cp_stm"]) >= 50 and \
                    (l["teacher_cp_stm"] > 0) != (w["teacher_cp_stm"] > 0):
                sign_flips.append(fen)
        if l["teacher_wdl_stm"] == w["teacher_wdl_stm"]:
            wdl_agree += 1
        if (l["teacher_cp_stm"] == w["teacher_cp_stm"]
                and l["teacher_mate"] == w["teacher_mate"]
                and l["teacher_bestmove"] == w["teacher_bestmove"]
                and l["teacher_wdl_stm"] == w["teacher_wdl_stm"]):
            tuple_exact += 1

    cp_sorted = sorted(cp_diffs)
    n_cp = len(cp_sorted)
    gate2 = {
        "bestmove_agreement": round(bm_agree / len(fens), 4),
        "mate_class_mismatches": len(class_mism),
        "cp_positions": n_cp,
        "cp_median_abs": cp_sorted[n_cp // 2] if n_cp else None,
        "cp_p95_abs": pct(cp_sorted, 95) if n_cp else None,
        "cp_p99_abs": pct(cp_sorted, 99) if n_cp else None,
        "cp_max_abs": cp_sorted[-1] if n_cp else None,
        "cp_mean_signed": (round(sum(signed) / len(signed), 4)
                           if signed else None),
        "sign_flips_ge50": len(sign_flips),
        "wdl_exact": round(wdl_agree / len(fens), 4),
        "tuple_exact": round(tuple_exact / len(fens), 4),
    }
    g2_pass = (
        gate2["bestmove_agreement"] >= 0.995
        and len(class_mism) == 0
        and n_cp > 0
        and gate2["cp_median_abs"] == 0
        and gate2["cp_p95_abs"] <= 1
        and gate2["cp_p99_abs"] <= 3
        and gate2["cp_max_abs"] <= 10
        and abs(gate2["cp_mean_signed"]) <= 0.25
        and len(sign_flips) == 0
        and gate2["wdl_exact"] >= 0.99
    )
    gate2["pass"] = g2_pass
    print(json.dumps(gate2, indent=1))

    report = {
        "schema_version": 1,
        "corpus_id": "s10-e2-w0-crossplatform-2048",
        "corpus_sha256": str(args.corpus),
        "n": len(fens),
        "linux_sha": LINUX_SHA,
        "windows_exe_sha": WIN_SHA,
        "windows_zip_sha": ("6f6c272ebd6ea594377715235c8a7326f75940ef4f4"
                            "f856f45106028fe6ae900"),
        "gate1_windows_self_determinism": gate1,
        "gate2_cross_platform": gate2,
        "overall_pass": gate1["pass"] and g2_pass,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    print(f"wrote {args.out}")
    print("W0 OVERALL:", "PASS" if report["overall_pass"] else "FAIL")
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
