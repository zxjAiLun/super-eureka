#!/usr/bin/env python3
"""S6-N3B Teacher labeling (hardened audit, fail-closed publish).

Contract (S6.0 spec):
- persistent UCI process, Threads=1, Hash=64 MB, MultiPV=1,
  UCI_ShowWDL=true, Syzygy disabled (default, recorded);
- per position: `ucinewgame` + `position fen <fen>` + `go nodes 16384`;
- parse the FINAL `info ... pv` line: score cp/mate + wdl + pv first move;
- score/wdl are side-to-move perspective (UCI convention);
- teacher death handling: respawn and retry the SAME position once; a second
  failure fails closed (abort, no silent drop).

Audit (ALWAYS on; there is NO --no-audit bypass):
- fresh-second-pass: for a fresh dataset (no stored labels), after ALL
  records are labeled the Teacher process is DESTROYED and a NEW Stockfish
  process is created, which independently re-labels the deterministic first
  `audit_n` records; every field must match exactly.
- vs-stored: when labels.jsonl already exists, the first `audit_n` records
  are re-labeled and compared field-by-field against the stored labels.

Publish (fail-closed): labels.jsonl and teacher_manifest.json are written to
temporary files first; only after the full labeling AND a passing audit are
they published, with the manifest renamed LAST as the commit point. An
exception, audit mismatch, or interruption leaves no dataset that
verify_dataset can accept as labeled.

teacher_manifest.json records: audit_mode, sample_count, sample position-id
SHA-256, checked, mismatches, Stockfish identity/options, and labels SHA.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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

LABEL_FIELDS = ("teacher_cp_stm", "teacher_mate", "teacher_bestmove",
                "teacher_wdl_stm", "nodes")


class Teacher:
    """Persistent UCI teacher process with a REAL read timeout (reader thread
    + queue: blocking readline never wedges the caller) and one-shot retry
    helpers."""

    def __init__(self, wsl: bool = True, binary: str | Path | None = None,
                 expected_binary_sha256: str | None = None):
        # Identity contract: held for the whole teacher lifetime AND every
        # respawn (Repair 2: a respawn must never fall back to defaults).
        self.launch_mode_wsl = wsl
        self.binary_arg = binary or TEACHER_BIN
        self.expected_binary_sha256 = expected_binary_sha256
        if wsl:
            cmd = ["wsl.exe", "-e", "bash", "-lc", self.binary_arg]
        else:
            path = Path(self.binary_arg).expanduser()
            cmd = [str(path)]
        actual_sha = _verify_binary_sha256(wsl, self.binary_arg,
                                           expected_binary_sha256)
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
            "nodes": TEACHER_NODES,
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
        import shlex
        out = subprocess.run(
            ["wsl.exe", "-e", "bash", "-lc",
             f"sha256sum {shlex.quote(str(binary))}"],
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


def respawn(teacher: Teacher) -> Teacher:
    """Respawn with the EXACT identity contract of the original instance."""
    try:
        teacher.close()
    except Exception:
        pass
    return Teacher(
        wsl=teacher.launch_mode_wsl,
        binary=teacher.binary_arg,
        expected_binary_sha256=teacher.expected_binary_sha256,
    )


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
    """Load stored labels; REJECTS duplicate position_id (never dict-cover)."""
    labels: dict[str, dict] = {}
    path = dataset_dir / "labels.jsonl"
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                pid = rec["position_id"]
                if pid in labels:
                    raise RuntimeError(
                        f"duplicate position_id in stored labels {pid[:16]}")
                labels[pid] = rec
    return labels


def audit_compare(first: dict, second: dict) -> list[str]:
    """Exact per-field comparison; returns the mismatching field names."""
    mismatches = []
    for field in LABEL_FIELDS:
        if first.get(field) != second.get(field):
            mismatches.append(field)
    return mismatches


def run_audit(labels: dict[str, dict], records: list[dict], audit_n: int,
              stored: dict[str, dict], teacher_factory,
              teacher_kwargs: dict) -> dict:
    """Audit the first `audit_n` records.

    fresh-second-pass: destroy the labeling process and create a NEW Teacher
    that independently re-labels the sample. vs-stored: compare against the
    stored labels (the labeling process is already done)."""
    mode = "vs-stored" if stored else "fresh-second-pass"
    sample = records[:audit_n]
    sampled_pids = [r["position_id"] for r in sample]
    audit = {
        "ok": True,
        "checked": 0,
        "mismatches": [],
        "mode": mode,
        "sample_count": len(sample),
        "sample_position_id_sha256": sha256_text("\n".join(sampled_pids)),
    }
    if mode == "fresh-second-pass":
        second_teacher = teacher_factory(**teacher_kwargs)
        try:
            second = {}
            for rec in sample:
                second[rec["position_id"]] = second_teacher.label(rec["fen"])
        finally:
            second_teacher.close()
    else:
        second = stored
    for rec in sample:
        first = labels[rec["position_id"]]
        second_lbl = second.get(rec["position_id"])
        audit["checked"] += 1
        if second_lbl is None:
            audit["ok"] = False
            audit["mismatches"].append({
                "position_id": rec["position_id"], "field": "missing",
            })
            break
        bad = audit_compare(first, second_lbl)
        if bad:
            audit["ok"] = False
            audit["mismatches"].append({
                "position_id": rec["position_id"], "fields": bad,
                "first": {f: first.get(f) for f in bad},
                "second": {f: second_lbl.get(f) for f in bad},
            })
            if len(audit["mismatches"]) >= 5:
                break
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True,
                        help="dataset dir (shards + manifest)")
    parser.add_argument("--audit-n", type=int, default=AUDIT_N)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--wsl", dest="wsl", action="store_true",
                       help="drive the teacher through wsl.exe (default)")
    group.add_argument("--native", dest="wsl", action="store_false",
                       help="run the teacher binary natively")
    parser.set_defaults(wsl=True)
    parser.add_argument("--teacher-binary", default=TEACHER_BIN,
                        help="actual teacher executable path")
    parser.add_argument("--expected-binary-sha256",
                        default=DEFAULT_BINARY_SHA256,
                        help="frozen teacher binary SHA-256; fail closed on "
                             "mismatch")
    args = parser.parse_args(sys.argv[1:])

    if not re.fullmatch(r"[0-9a-f]{64}", args.expected_binary_sha256 or ""):
        print(f"FATAL: --expected-binary-sha256 must be 64 lowercase hex, "
              f"got {args.expected_binary_sha256!r}", flush=True)
        return 4
    dataset_dir = Path(args.dataset)
    records = load_records(dataset_dir)
    stored = load_stored_labels(dataset_dir)
    audit_n = min(args.audit_n, len(records))
    print(f"labeling {len(records)} positions (wsl={args.wsl}, "
          f"audit_mode={'vs-stored' if stored else 'fresh-second-pass'})",
          flush=True)

    teacher = Teacher(wsl=args.wsl, binary=args.teacher_binary,
                      expected_binary_sha256=args.expected_binary_sha256)
    labels: dict[str, dict] = {}
    try:
        for i, rec in enumerate(records):
            if i % 500 == 0:
                print(f"  {i}/{len(records)}", flush=True)
            pid = rec["position_id"]
            try:
                labels[pid] = teacher.label(rec["fen"])
            except RuntimeError:
                print(f"  teacher died at {i}; respawning and retrying",
                      flush=True)
                teacher = respawn(teacher)
                try:
                    labels[pid] = teacher.label(rec["fen"])
                except RuntimeError as exc:
                    print(f"FATAL: label failed twice at record {i} "
                          f"{pid}: {exc}", flush=True)
                    return 2
        if len(labels) != len(records):
            print(f"FATAL: labeled {len(labels)} != {len(records)} records",
                  flush=True)
            return 2
        teacher.close()

        audit = run_audit(labels, records, audit_n, stored, Teacher,
                          dict(wsl=args.wsl, binary=args.teacher_binary,
                               expected_binary_sha256=
                               args.expected_binary_sha256))
        print(f"audit [{audit['mode']}]: "
              f"{'PASS' if audit['ok'] else 'FAIL'} "
              f"({audit['checked']} checked, "
              f"{len(audit['mismatches'])} mismatches)", flush=True)
        if not audit["ok"]:
            print("NOT publishing: audit failed", flush=True)
            return 3

        labels_text = "".join(
            json.dumps({"position_id": pid, **lbl}, ensure_ascii=False,
                       sort_keys=True) + "\n"
            for pid, lbl in sorted(labels.items()))
        teacher_manifest = {
            "engine": teacher.uci_id_name,
            "verified_binary_sha256": teacher.verified_binary_sha256,
            "uci_id_name": teacher.uci_id_name,
            "uci_id_author": teacher.uci_id_author,
            "binary_path": args.teacher_binary,
            "binary_sha256": args.expected_binary_sha256,
            "nodes": TEACHER_NODES,
            "options": dict(TEACHER_OPTIONS),
            "uci_options_seen": list(teacher.uci_options.keys()),
            "syzygy": "disabled (default)",
            "labeling_mode": "go nodes",
            "audit": audit,
            "labels_sha256": sha256_text(labels_text),
            "labeled_positions": len(labels),
        }
        # temp files first; publish only after full labeling + audit pass,
        # manifest renamed LAST as the commit point.
        labels_tmp = dataset_dir / "labels.jsonl.tmp"
        manifest_tmp = dataset_dir / "teacher_manifest.json.tmp"
        labels_tmp.write_text(labels_text, encoding="utf-8")
        manifest_tmp.write_text(
            json.dumps(teacher_manifest, ensure_ascii=False, indent=2),
            encoding="utf-8")
        os.replace(labels_tmp, dataset_dir / "labels.jsonl")
        os.replace(manifest_tmp, dataset_dir / "teacher_manifest.json")
        print("labels.jsonl + teacher_manifest.json published", flush=True)
        print(json.dumps(teacher_manifest, ensure_ascii=False, indent=2))
        return 0
    finally:
        try:
            teacher.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
