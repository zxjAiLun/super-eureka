#!/usr/bin/env python3
"""S6.0 Teacher labeling: label a dataset with a FROZEN Stockfish artifact.

Contract (S6.0 spec):
- persistent UCI process, Threads=1, Hash=64 MB, MultiPV=1,
  UCI_ShowWDL=true, Syzygy disabled (default, recorded);
- per position: `ucinewgame` + `position fen <fen>` + `go nodes 16384`;
- parse the FINAL `info ... pv` line: score cp/mate + wdl + pv first move;
- score/wdl are side-to-move perspective (UCI convention);
- determinism audit: label the first N positions twice and require exact
  bestmove/score/WDL/mate equality.

Outputs (companion to the immutable dataset, keyed by position_id):
  labels.jsonl          one JSON per position
  teacher_manifest.json frozen teacher identity + audit result
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

TEACHER_BIN = "~/sf18"
TEACHER_NODES = 16384
TEACHER_OPTIONS = {
    "Threads": "1",
    "Hash": "64",
    "MultiPV": "1",
    "UCI_ShowWDL": "true",
}
AUDIT_N = 1000


class Teacher:
    def __init__(self, wsl: bool):
        cmd = ["wsl.exe", "-e", "bash", "-lc", TEACHER_BIN] if wsl else [TEACHER_BIN]
        # NOTE: stderr must be merged into stdout (wsl.exe console bridge
        # deadlocks `go` output when stderr is DEVNULL), and the UCI handshake
        # must be sent EXACTLY ONCE (a second `uci` poisons subsequent `go`
        # output under the wsl.exe bridge).
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        # NOTE (wsl.exe bridge quirks, empirically verified): the UCI
        # handshake must be requested with exactly ONE `uci` command (a
        # second one poisons subsequent `go` output), stderr must be merged
        # into stdout (DEVNULL deadlocks `go`), and `isready` must be
        # answered BEFORE any `setoption`.
        self.send("uci")
        handshake = self._read_until("uciok")
        self.uci_options: dict[str, str] = {
            m.group(1): line.strip()
            for line in handshake
            if (m := re.match(r"option name (\S+)", line))
        }
        self.send("isready")
        self._expect("readyok")
        for name, value in TEACHER_OPTIONS.items():
            self.send(f"setoption name {name} value {value}")

    def _collect_options(self):
        raise RuntimeError("unused: a second `uci` handshake poisons `go` output")

    def send(self, cmd: str):
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def _read_until(self, marker: str, timeout: float = 30.0) -> list[str]:
        lines: list[str] = []
        import time
        deadline = time.time() + timeout
        while True:
            if time.time() > deadline:
                raise RuntimeError(
                    f"teacher timeout waiting for {marker!r}; got {len(lines)} lines")
            line = self.proc.stdout.readline()
            if not line:
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
        lines = self._read_until("bestmove")
        final_info: dict = {}
        for line in lines:
            if not line.startswith("info"):
                continue
            if " pv " not in line:
                continue
            m = re.search(r"score cp (-?\d+)|score mate (-?\d+)", line)
            wdl = re.search(r"wdl (\d+) (\d+) (\d+)", line)
            pv = re.search(r" pv (\S+)", line)
            if m:
                final_info["cp"] = int(m.group(1)) if m.group(1) is not None else None
                final_info["mate"] = int(m.group(2)) if m.group(2) is not None else None
            if wdl:
                final_info["wdl"] = tuple(int(wdl.group(i)) for i in (1, 2, 3))
            if pv:
                final_info["bestmove"] = pv.group(1)
        best = lines[-1].split("bestmove ")[1].split(" ")[0]
        return {
            "teacher_cp_stm": final_info.get("cp"),
            "teacher_mate": final_info.get("mate"),
            "teacher_bestmove": final_info.get("bestmove", best),
            "teacher_wdl_stm": list(final_info.get("wdl", [None, None, None])),
        }

    def close(self):
        try:
            self.send("quit")
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="dataset dir (shards + manifest)")
    parser.add_argument("--audit-n", type=int, default=AUDIT_N)
    parser.add_argument("--no-audit", action="store_true")
    parser.add_argument("--wsl", action="store_true", default=True)
    args = parser.parse_args(sys.argv[1:])

    dataset_dir = Path(args.dataset)
    manifest = json.loads(
        (dataset_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    records: list[dict] = []
    for shard in sorted(dataset_dir.glob("part-*.jsonl")):
        for line in shard.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    print(f"labeling {len(records)} positions")

    teacher = Teacher(args.wsl)
    binary = teacher.uci_options.get("id name", "unknown")

    labels: dict[str, dict] = {}
    for i, rec in enumerate(records):
        if i % 500 == 0:
            print(f"  {i}/{len(records)}", flush=True)
        try:
            labels[rec["position_id"]] = teacher.label(rec["fen"])
        except RuntimeError as exc:
            print(f"teacher error at {i}: {exc}")
            return 2

    # determinism audit
    audit = {"ok": True, "checked": 0, "mismatches": []}
    if not args.no_audit:
        audit_n = min(args.audit_n, len(records))
        for i in range(audit_n):
            rec = records[i]
            first = labels[rec["position_id"]]
            second = teacher.label(rec["fen"])
            audit["checked"] += 1
            for field in ("teacher_cp_stm", "teacher_mate",
                          "teacher_bestmove", "teacher_wdl_stm"):
                if first[field] != second[field]:
                    audit["ok"] = False
                    audit["mismatches"].append({
                        "position_id": rec["position_id"], "field": field,
                        "first": first[field], "second": second[field],
                    })
                    if len(audit["mismatches"]) >= 5:
                        break
            if not audit["ok"]:
                break
        print(f"audit: {'PASS' if audit['ok'] else 'FAIL'} "
              f"({audit['checked']} replayed)")
    teacher.close()

    labels_path = dataset_dir / "labels.jsonl"
    labels_path.write_text(
        "".join(json.dumps(
            {"position_id": pid, **lbl}, ensure_ascii=False, sort_keys=True) + "\n"
            for pid, lbl in sorted(labels.items())),
        encoding="utf-8",
    )
    teacher_manifest = {
        "engine": binary,
        "binary_path": TEACHER_BIN,
        "binary_sha256": None,  # filled by caller after verification
        "nodes": TEACHER_NODES,
        "options": dict(TEACHER_OPTIONS),
        "uci_options_seen": list(teacher.uci_options.keys()),
        "syzygy": "disabled (default)",
        "labeling_mode": "go nodes",
        "audit": audit,
        "labels_sha256": sha256_file(labels_path),
        "labeled_positions": len(labels),
    }
    (dataset_dir / "teacher_manifest.json").write_text(
        json.dumps(teacher_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("teacher_manifest:", json.dumps(teacher_manifest, indent=2))
    return 0 if audit.get("ok") else 3


if __name__ == "__main__":
    raise SystemExit(main())
