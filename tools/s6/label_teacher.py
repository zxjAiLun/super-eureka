#!/usr/bin/env python3
"""S6.0 Teacher labeling: label a dataset with a FROZEN Stockfish artifact.

Contract (S6.0 spec):
- persistent UCI process, Threads=1, Hash=64 MB, MultiPV=1,
  UCI_ShowWDL=true, Syzygy disabled (default, recorded);
- per position: `ucinewgame` + `position fen <fen>` + `go nodes 16384`;
- parse the FINAL `info ... pv` line: score cp/mate + wdl + pv first move;
- score/wdl are side-to-move perspective (UCI convention);
- determinism / cross-driver audit: re-label the first N positions with the
  (hardened) driver and require EXACT bestmove/score/WDL/mate equality
  against the STORED labels (proves driver hardening did not change teacher
  semantics);
- teacher death handling: respawn and retry the SAME position once; a second
  failure fails closed (abort, no silent drop).

Outputs (companion to the immutable dataset, keyed by position_id):
  labels.jsonl          one JSON per position
  teacher_manifest.json frozen teacher identity + audit result
"""

from __future__ import annotations

import argparse
import hashlib
import json
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

TEACHER_BIN = "~/sf18"
TEACHER_NODES = 16384
TEACHER_OPTIONS = {
    "Threads": "1",
    "Hash": "64",
    "MultiPV": "1",
    "UCI_ShowWDL": "true",
}
DEFAULT_BINARY_SHA256 = "6b087694916228c905a5e14db74cca8c7e5643602226af1fa5d42353c455b9f9"
AUDIT_N = 1000


class Teacher:
    """Persistent UCI teacher process with a REAL read timeout (reader thread
    + queue: blocking readline never wedges the caller) and one-shot retry
    helpers."""

    def __init__(self, wsl: bool = True, binary: str | Path | None = None,
                 expected_binary_sha256: str | None = None):
        if wsl:
            cmd = ["wsl.exe", "-e", "bash", "-lc", binary or TEACHER_BIN]
        else:
            # native: "~/" must be expanded before exec
            path = Path(binary or TEACHER_BIN).expanduser()
            cmd = [str(path)]
        # FAIL-CLOSED executable verification: hash the ACTUAL teacher
        # binary and require it to match the expected frozen identity before
        # any labeling happens.
        actual_sha = _verify_binary_sha256(wsl, binary or TEACHER_BIN,
                                           expected_binary_sha256)
        # wsl.exe bridge quirks (empirically verified): stderr must be merged
        # into stdout (DEVNULL deadlocks `go`); exactly ONE `uci` handshake;
        # `isready` must be answered before any `setoption`.
        self.verified_binary_sha256 = actual_sha
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._reader = threading.Thread(target=self._pump, daemon=True)
        self._reader.start()

        self.send("uci")
        handshake = self._read_until("uciok")
        self.uci_id_name = self._id_field(handshake, "id name")
        self.uci_id_author = self._id_field(handshake, "id author")
        self.uci_options: dict[str, str] = {
            m.group(1): line.strip()
            for line in handshake
            if (m := re.match(r"option name (\S+)", line))
        }
        self.send("isready")
        self._expect("readyok")
        for name, value in TEACHER_OPTIONS.items():
            self.send(f"setoption name {name} value {value}")

    @staticmethod
    def _id_field(lines: list[str], prefix: str) -> str:
        for line in lines:
            if line.startswith(prefix):
                return line[len(prefix):].strip()
        return "unknown"

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
        if self.proc.poll() is not None:
            raise RuntimeError("teacher process not running")
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def _read_until(self, marker: str, timeout: float = 30.0) -> list[str]:
        lines: list[str] = []
        deadline = time.time() + timeout
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise RuntimeError(
                    f"teacher timeout waiting for {marker!r}; "
                    f"got {len(lines)} lines")
            try:
                line = self._queue.get(timeout=remaining)
            except queue.Empty:
                raise RuntimeError(
                    f"teacher timeout waiting for {marker!r}; "
                    f"got {len(lines)} lines") from None
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
            if self.proc.poll() is None:
                self.send("quit")
                self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


def _verify_binary_sha256(wsl: bool, binary: str | Path,
                          expected: str | None) -> str:
    """Hash the ACTUAL teacher executable and fail closed on mismatch."""
    if wsl:
        out = subprocess.run(
            ["wsl.exe", "-e", "bash", "-lc", f"sha256sum {binary}"],
            capture_output=True, text=True, timeout=60)
        if out.returncode != 0 or not out.stdout.strip():
            raise RuntimeError(f"cannot hash teacher binary via wsl: {out.stderr}")
        actual = out.stdout.split()[0]
    else:
        path = Path(binary).expanduser()
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected and actual != expected:
        raise RuntimeError(
            f"teacher binary SHA mismatch: expected {expected}, "
            f"actual {actual} - FAIL CLOSED")
    return actual


def respawn(teacher: Teacher, wsl: bool) -> Teacher:
    try:
        teacher.close()
    except Exception:
        pass
    return Teacher(wsl)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_records(dataset_dir: Path) -> list[dict]:
    records: list[dict] = []
    for shard in sorted(dataset_dir.glob("part-*.jsonl")):
        for line in shard.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    return records


def load_stored_labels(dataset_dir: Path) -> dict[str, dict]:
    labels: dict[str, dict] = {}
    path = dataset_dir / "labels.jsonl"
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                labels[rec["position_id"]] = rec
    return labels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="dataset dir (shards + manifest)")
    parser.add_argument("--audit-n", type=int, default=AUDIT_N)
    parser.add_argument("--no-audit", action="store_true")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--wsl", dest="wsl", action="store_true",
                       help="drive the teacher through wsl.exe (default)")
    group.add_argument("--native", dest="wsl", action="store_false",
                       help="run the teacher binary natively (expanduser applied)")
    parser.set_defaults(wsl=True)
    parser.add_argument("--teacher-binary", default=TEACHER_BIN,
                        help="actual teacher executable path")
    parser.add_argument("--expected-binary-sha256", default=DEFAULT_BINARY_SHA256,
                        help="frozen teacher binary SHA-256; the driver hashes "
                             "the ACTUAL executable and fails closed on mismatch")
    args = parser.parse_args(sys.argv[1:])

    dataset_dir = Path(args.dataset)
    records = load_records(dataset_dir)
    stored = load_stored_labels(dataset_dir)
    print(f"labeling {len(records)} positions (wsl={args.wsl})")

    teacher = Teacher(wsl=args.wsl, binary=args.teacher_binary,
                      expected_binary_sha256=args.expected_binary_sha256)
    labels: dict[str, dict] = {}
    for i, rec in enumerate(records):
        if i % 500 == 0:
            print(f"  {i}/{len(records)}", flush=True)
        pid = rec["position_id"]
        try:
            labels[pid] = teacher.label(rec["fen"])
        except RuntimeError:
            # respawn and retry the SAME position once; second failure FAILS
            # CLOSED (no silent drop, no skip)
            print(f"  teacher died at {i}; respawning and retrying", flush=True)
            teacher = respawn(teacher, args.wsl)
            try:
                labels[pid] = teacher.label(rec["fen"])
            except RuntimeError as exc:
                print(f"FATAL: label failed twice at record {i} {pid}: {exc}")
                teacher.close()
                return 2

    # cross-driver / determinism audit: hardened driver vs STORED labels
    audit = {"ok": True, "checked": 0, "mismatches": [], "mode": "vs-stored-labels"}
    if not args.no_audit:
        audit_n = min(args.audit_n, len(records))
        for i in range(audit_n):
            rec = records[i]
            first = labels[rec["position_id"]]
            second = stored.get(rec["position_id"])
            audit["checked"] += 1
            if second is None:
                audit["ok"] = False
                audit["mismatches"].append({
                    "position_id": rec["position_id"], "field": "missing",
                })
                break
            for field in ("teacher_cp_stm", "teacher_mate",
                          "teacher_bestmove", "teacher_wdl_stm"):
                if first[field] != second[field]:
                    audit["ok"] = False
                    audit["mismatches"].append({
                        "position_id": rec["position_id"], "field": field,
                        "hardened": first[field], "stored": second[field],
                    })
                    if len(audit["mismatches"]) >= 5:
                        break
            if not audit["ok"]:
                break
        print(f"audit: {'PASS' if audit['ok'] else 'FAIL'} "
              f"({audit['checked']} positions, hardened vs stored)")
    teacher.close()

    # write labels only if they changed or the file is missing (the hardened
    # driver must produce IDENTICAL labels; keep the file byte-stable)
    labels_path = dataset_dir / "labels.jsonl"
    new_labels_text = "".join(
        json.dumps({"position_id": pid, **lbl}, ensure_ascii=False,
                   sort_keys=True) + "\n"
        for pid, lbl in sorted(labels.items()))
    if not labels_path.is_file() or labels_path.read_text(
            encoding="utf-8") != new_labels_text:
        if not args.no_audit and not audit["ok"]:
            print("NOT writing labels: audit failed", flush=True)
            return 3
        labels_path.write_text(new_labels_text, encoding="utf-8")
        print("labels.jsonl written (changed)", flush=True)
    else:
        print("labels.jsonl byte-identical (unchanged)", flush=True)

    teacher_manifest = {
        "engine": teacher.uci_id_name,
        "verified_binary_sha256": teacher.verified_binary_sha256,
        "uci_id_name": teacher.uci_id_name,
        "uci_id_author": teacher.uci_id_author,
        "binary_path": TEACHER_BIN,
        "binary_sha256": args.expected_binary_sha256,
        "nodes": TEACHER_NODES,
        "options": dict(TEACHER_OPTIONS),
        "uci_options_seen": list(teacher.uci_options.keys()),
        "syzygy": "disabled (default)",
        "labeling_mode": "go nodes",
        "audit": audit,
        "labels_sha256": sha256_text(new_labels_text),
        "labeled_positions": len(labels),
    }
    (dataset_dir / "teacher_manifest.json").write_text(
        json.dumps(teacher_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("teacher_manifest written")
    print(json.dumps(teacher_manifest, ensure_ascii=False, indent=2))
    return 0 if audit.get("ok") else 3


if __name__ == "__main__":
    raise SystemExit(main())
