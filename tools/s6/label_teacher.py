#!/usr/bin/env python3
"""S6-N3B Teacher labeling (hardened audit, fail-closed publish; S10-B2A
resumable checkpointing).

Contract (S6.0 spec - FROZEN, do not change):
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

S10-B2A resumable checkpointing:
- labels.partial.jsonl  : append-only journal, DATASET RECORD ORDER (not
  sorted by position_id); only the prefix described by teacher-progress.json
  is committed, any tail beyond it is an uncommitted crash window.
- teacher-progress.json : the CHECKPOINT COMMIT RECORD. Written atomically
  every --checkpoint-interval (default 500) completed positions. Binds the
  run to the dataset identity (dataset_id, dataset_sha256, record_count,
  ordered_position_id_sha256), the teacher identity (binary SHA-256, nodes,
  exact UCI options), and the committed prefix (completed_count,
  partial_size_bytes, partial_labels_sha256).
- resume: progress is authoritative. The committed prefix is re-hashed,
  re-parsed, and position-by-position validated against the dataset order
  (this rejects duplicate / unknown / reordered / non-prefix position ids
  for free, with explicit duplicate detection). An uncommitted tail beyond
  partial_size_bytes is truncated. A partial shorter than the committed
  size, a SHA mismatch, or ANY identity mismatch fails closed.
- interrupted runs never publish final artifacts; a dataset that only has
  partial/progress files remains unlabelled for verify_dataset.

teacher_manifest.json records: audit_mode, sample_count, sample position-id
SHA-256, checked, mismatches, Stockfish identity/options, labels SHA, and
the resume telemetry (checkpoint_interval, ordered_position_id_sha256,
resume_count, checkpoint_schema_version).
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
CHECKPOINT_INTERVAL = 500
CHECKPOINT_SCHEMA_VERSION = 1

LABEL_FIELDS = ("teacher_cp_stm", "teacher_mate", "teacher_bestmove",
                "teacher_wdl_stm", "nodes")

PARTIAL_NAME = "labels.partial.jsonl"
PROGRESS_NAME = "teacher-progress.json"

_HEX64_RE = re.compile(r"[0-9a-f]{64}")


class ResumeError(RuntimeError):
    """FAIL-CLOSED resume validation error (provenance/consistency)."""


def canonical_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ordered_position_id_sha256(records: list[dict]) -> str:
    """sha256 of the joined dataset-order position ids, NO trailing newline.

    Frozen serialization; proves the resume workload is the same dataset in
    the same deterministic order, not merely the same dataset SHA.
    """
    return sha256_text("\n".join(r["position_id"] for r in records))


def load_records(dataset_dir: Path) -> list[dict]:
    records: list[dict] = []
    for shard in sorted(dataset_dir.glob("part-*.jsonl")):
        for line in shard.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    return records


def load_dataset_manifest(dataset_dir: Path) -> dict:
    path = dataset_dir / "dataset_manifest.json"
    if not path.is_file():
        raise ResumeError(f"FAIL CLOSED: missing dataset_manifest.json "
                          f"({path})")
    return json.loads(path.read_text(encoding="utf-8"))


def build_resume_contract(dataset_manifest: dict,
                          records: list[dict]) -> dict:
    """Identity the partial run is bound to (dataset + record order)."""
    return {
        "dataset_id": dataset_manifest.get("dataset_id"),
        "dataset_sha256": dataset_manifest.get("dataset_sha256"),
        "record_count": len(records),
        "ordered_position_id_sha256": ordered_position_id_sha256(records),
    }


def build_teacher_contract(teacher_binary_sha256: str) -> dict:
    """Identity the partial run is bound to (teacher binary + settings)."""
    return {
        "teacher_binary_sha256": teacher_binary_sha256,
        "teacher_nodes": TEACHER_NODES,
        "teacher_options": dict(TEACHER_OPTIONS),
    }


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


# --------------------------------------------------------------------------
# S10-B2A: checkpoint / resume machinery
# --------------------------------------------------------------------------

def label_line(pid: str, lbl: dict) -> str:
    """One deterministic partial-journal line (dataset record order)."""
    return json.dumps({"position_id": pid, **lbl}, ensure_ascii=False,
                      sort_keys=True) + "\n"


def atomic_write_json_fsync(path: Path, obj) -> None:
    """Write JSON to tmp -> flush -> fsync -> os.replace (atomic publish)."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False, indent=2))
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def load_progress(dataset_dir: Path) -> dict | None:
    path = dataset_dir / PROGRESS_NAME
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise ResumeError(
            f"FAIL CLOSED: cannot parse {PROGRESS_NAME}: {exc}") from exc


def _require(progress: dict, field: str, ty, where: str):
    if field not in progress:
        raise ResumeError(f"FAIL CLOSED: {where} missing field {field!r}")
    v = progress[field]
    if isinstance(v, bool) and ty is int:
        raise ResumeError(f"FAIL CLOSED: {where} field {field!r} is a bool")
    if not isinstance(v, ty):
        raise ResumeError(f"FAIL CLOSED: {where} field {field!r} has "
                          f"wrong type {type(v).__name__}, expected "
                          f"{ty.__name__}")
    return v


def validate_progress_schema(progress: dict) -> None:
    where = PROGRESS_NAME
    _require(progress, "schema_version", int, where)
    if progress["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise ResumeError(
            f"FAIL CLOSED: {where} schema_version "
            f"{progress['schema_version']} != {CHECKPOINT_SCHEMA_VERSION}")
    for field in ("dataset_id", "teacher_binary_sha256"):
        _require(progress, field, str, where)
    for field in ("dataset_sha256", "ordered_position_id_sha256",
                  "partial_labels_sha256"):
        v = _require(progress, field, str, where)
        if not _HEX64_RE.fullmatch(v):
            raise ResumeError(
                f"FAIL CLOSED: {where} field {field!r} is not 64 lowercase "
                f"hex")
    _require(progress, "record_count", int, where)
    _require(progress, "teacher_nodes", int, where)
    _require(progress, "completed_count", int, where)
    _require(progress, "partial_size_bytes", int, where)
    _require(progress, "checkpoint_interval", int, where)
    opts = _require(progress, "teacher_options", dict, where)
    if set(opts) != set(TEACHER_OPTIONS):
        raise ResumeError(
            f"FAIL CLOSED: {where} teacher_options keys {sorted(opts)} != "
            f"{sorted(TEACHER_OPTIONS)}")
    for k, v in opts.items():
        if not isinstance(v, str):
            raise ResumeError(
                f"FAIL CLOSED: {where} teacher_options[{k!r}] not a string")
    if progress["completed_count"] < 0:
        raise ResumeError("FAIL CLOSED: negative completed_count")
    if progress["partial_size_bytes"] < 0:
        raise ResumeError("FAIL CLOSED: negative partial_size_bytes")


def _check_contract(progress: dict, contract: dict, label: str) -> None:
    for key, expected in contract.items():
        actual = progress.get(key)
        if actual != expected:
            raise ResumeError(
                f"FAIL CLOSED: {label} mismatch on {key!r}: "
                f"progress has {actual!r}, dataset/teacher has "
                f"{expected!r}")


def validate_committed_partial(dataset_dir: Path, records: list[dict],
                               progress: dict) -> tuple[dict[str, dict],
                                                         "hashlib._Hash"]:
    """Validate + load the committed prefix described by progress.

    Returns (labels_by_pid, hasher) where hasher is the incremental SHA-256
    state over the committed bytes (ready for continued labeling).
    Truncates any uncommitted crash tail beyond partial_size_bytes.
    """
    partial_path = dataset_dir / PARTIAL_NAME
    if not partial_path.is_file():
        raise ResumeError(
            f"FAIL CLOSED: {PROGRESS_NAME} exists but {PARTIAL_NAME} "
            f"does not (Case 3: committed journal lost)")
    size = partial_path.stat().st_size
    committed = progress["partial_size_bytes"]
    if size < committed:
        raise ResumeError(
            f"FAIL CLOSED: {PARTIAL_NAME} is {size} bytes, shorter than "
            f"the committed {committed} bytes (Case 6: committed data "
            f"lost/corrupted)")
    with open(partial_path, "rb") as fh:
        committed_bytes = fh.read(committed)
    actual_sha = hashlib.sha256(committed_bytes).hexdigest()
    if actual_sha != progress["partial_labels_sha256"]:
        raise ResumeError(
            f"FAIL CLOSED: committed prefix SHA mismatch "
            f"({actual_sha[:16]}... != {progress['partial_labels_sha256'][:16]}...)")

    completed = progress["completed_count"]
    if completed > len(records):
        raise ResumeError(
            f"FAIL CLOSED: completed_count {completed} > record count "
            f"{len(records)}")

    hasher = hashlib.sha256()
    hasher.update(committed_bytes)

    labels: dict[str, dict] = {}
    for i, line in enumerate(
            committed_bytes.decode("utf-8").splitlines()):
        if not line.strip():
            raise ResumeError(
                f"FAIL CLOSED: blank line {i} in committed partial")
        try:
            rec = json.loads(line)
        except ValueError as exc:
            raise ResumeError(
                f"FAIL CLOSED: committed partial line {i} is not JSON: "
                f"{exc}") from exc
        pid = rec.get("position_id")
        if pid in labels:
            raise ResumeError(
                f"FAIL CLOSED: duplicate position_id {str(pid)[:16]} in "
                f"committed partial line {i}")
        if i >= len(records):
            raise ResumeError(
                f"FAIL CLOSED: committed partial has more lines than the "
                f"dataset has records")
        expected_pid = records[i]["position_id"]
        if pid != expected_pid:
            raise ResumeError(
                f"FAIL CLOSED: committed partial line {i} position_id "
                f"{str(pid)[:16]} != dataset record {i} position_id "
                f"{expected_pid[:16]} (non-prefix/reordered/unknown pid)")
        for field in LABEL_FIELDS:
            if field not in rec:
                raise ResumeError(
                    f"FAIL CLOSED: committed partial line {i} missing "
                    f"label field {field!r}")
        if rec.get("nodes") != TEACHER_NODES:
            raise ResumeError(
                f"FAIL CLOSED: committed partial line {i} nodes "
                f"{rec.get('nodes')!r} != {TEACHER_NODES}")
        labels[pid] = {f: rec[f] for f in LABEL_FIELDS}
    if len(labels) != completed:
        raise ResumeError(
            f"FAIL CLOSED: committed partial has {len(labels)} parsed "
            f"lines but progress claims completed_count "
            f"{completed}")

    # Case 5: uncommitted crash tail - truncate to the committed boundary.
    if size > committed:
        with open(partial_path, "r+b") as fh:
            fh.truncate(committed)
            fh.flush()
            os.fsync(fh.fileno())
        print(f"resume: discarded {size - committed} uncommitted tail "
              f"bytes from {PARTIAL_NAME}", flush=True)
    return labels, hasher


def validate_and_load_resume(dataset_dir: Path, records: list[dict],
                             dataset_contract: dict,
                             teacher_contract: dict) -> tuple[dict[str, dict],
                                                              "hashlib._Hash",
                                                              dict]:
    """Full resume validation. Returns (labels, hasher, progress).

    Fails closed (ResumeError) on ANY identity mismatch or partial
    corruption. Cases 1/2/3/4/5/6 of the crash-consistency design.
    """
    progress = load_progress(dataset_dir)
    partial_path = dataset_dir / PARTIAL_NAME
    has_partial = partial_path.is_file()

    if progress is None and not has_partial:
        return {}, hashlib.sha256(), {}  # Case 1: fresh run
    if progress is None and has_partial:
        raise ResumeError(
            f"FAIL CLOSED: {PARTIAL_NAME} exists without {PROGRESS_NAME} "
            f"(Case 4: orphan journal - which lines are committed is "
            f"unknowable)")
    if progress is not None and not has_partial:
        raise ResumeError(
            f"FAIL CLOSED: {PROGRESS_NAME} exists without {PARTIAL_NAME} "
            f"(Case 3: committed journal lost)")

    validate_progress_schema(progress)
    _check_contract(progress, dataset_contract, "dataset identity")
    _check_contract(progress, teacher_contract, "teacher identity")
    labels, hasher = validate_committed_partial(dataset_dir, records,
                                                progress)
    return labels, hasher, progress


def write_checkpoint(progress_base: dict, completed_count: int,
                     partial_size_bytes: int, partial_sha: str,
                     checkpoint_interval: int, dataset_dir: Path) -> None:
    progress = dict(progress_base)
    progress.update({
        "completed_count": completed_count,
        "partial_size_bytes": partial_size_bytes,
        "partial_labels_sha256": partial_sha,
        "checkpoint_interval": checkpoint_interval,
    })
    atomic_write_json_fsync(dataset_dir / PROGRESS_NAME, progress)


def finalize_labels_from_partial(labels: dict[str, dict],
                                 records: list[dict]) -> tuple[str, str]:
    """Deterministic final labels.jsonl text + SHA (pid-sorted, existing
    contract). Returns (labels_text, labels_sha256)."""
    if len(labels) != len(records):
        raise ResumeError(
            f"FAIL CLOSED: {len(labels)} labels != {len(records)} records "
            f"at finalization")
    if len({r["position_id"] for r in records}) != len(records):
        raise ResumeError("FAIL CLOSED: duplicate position_id in dataset")
    labels_text = "".join(
        json.dumps({"position_id": pid, **lbl}, ensure_ascii=False,
                   sort_keys=True) + "\n"
        for pid, lbl in sorted(labels.items()))
    return labels_text, sha256_text(labels_text)


def label_dataset(dataset_dir: Path, records: list[dict],
                  teacher_factory=Teacher,
                  teacher_kwargs: dict | None = None,
                  audit_n: int = AUDIT_N,
                  checkpoint_interval: int = CHECKPOINT_INTERVAL,
                  stored: dict[str, dict] | None = None,
                  resume_count: int = 0) -> int:
    """Full labeling pipeline with checkpointing, audit, and publication.

    Returns the process exit code. Raises KeyboardInterrupt through (the
    durable state stays at the last successful checkpoint).
    """
    teacher_kwargs = teacher_kwargs or {}
    partial_path = dataset_dir / PARTIAL_NAME

    dataset_manifest = load_dataset_manifest(dataset_dir)
    dataset_contract = build_resume_contract(dataset_manifest, records)

    labels: dict[str, dict] = {}
    hasher = hashlib.sha256()
    start_index = 0
    if stored is None:
        labels, hasher, progress = validate_and_load_resume(
            dataset_dir, records, dataset_contract,
            build_teacher_contract(
                teacher_kwargs.get("expected_binary_sha256",
                                    DEFAULT_BINARY_SHA256)))
        if progress:
            start_index = progress["completed_count"]
            resume_count += 1
            print(f"resume: continuing from {start_index}/{len(records)} "
                  f"committed positions", flush=True)

    # A first Teacher instance is created to obtain the verified identity
    # (binary SHA) even for resume validation order stability; for a pure
    # resume the teacher is still required for the remaining work.
    teacher = teacher_factory(**teacher_kwargs)
    try:
        partial_fh = open(partial_path, "a", encoding="utf-8")
    except OSError as exc:
        teacher.close()
        raise ResumeError(f"FAIL CLOSED: cannot open {PARTIAL_NAME}: {exc}")

    progress_base = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        **dataset_contract,
        **build_teacher_contract(teacher.verified_binary_sha256),
    }

    last_checkpoint_count = start_index

    def checkpoint(count: int) -> None:
        nonlocal last_checkpoint_count
        partial_fh.flush()
        os.fsync(partial_fh.fileno())
        size = partial_fh.tell()
        write_checkpoint(progress_base, count, size, hasher.hexdigest(),
                         checkpoint_interval, dataset_dir)
        last_checkpoint_count = count

    try:
        for i in range(start_index, len(records)):
            rec = records[i]
            pid = rec["position_id"]
            try:
                lbl = teacher.label(rec["fen"])
            except RuntimeError:
                print(f"  teacher died at {i}; respawning and retrying",
                      flush=True)
                # Respawn with the EXACT same factory + identity kwargs
                # (a respawn must never fall back to defaults).
                try:
                    teacher.close()
                except Exception:
                    pass
                teacher = teacher_factory(**teacher_kwargs)
                try:
                    lbl = teacher.label(rec["fen"])
                except RuntimeError as exc:
                    print(f"FATAL: label failed twice at record {i} "
                          f"{pid}: {exc}", flush=True)
                    return 2
            labels[pid] = lbl
            line = label_line(pid, lbl)
            partial_fh.write(line)
            hasher.update(line.encode("utf-8"))
            if (i + 1 - start_index) % checkpoint_interval == 0:
                checkpoint(i + 1)
                print(f"  checkpoint {i + 1}/{len(records)}", flush=True)

        # Final checkpoint for the tail when the interval is not a divisor
        # (progress.completed_count == len(records) is required pre-audit).
        if last_checkpoint_count != len(records):
            checkpoint(len(records))

        if len(labels) != len(records):
            print(f"FATAL: labeled {len(labels)} != {len(records)} records",
                  flush=True)
            return 2
        teacher.close()

        audit = run_audit(labels, records, min(audit_n, len(records)),
                          stored or {}, teacher_factory, teacher_kwargs)
        print(f"audit [{audit['mode']}]: "
              f"{'PASS' if audit['ok'] else 'FAIL'} "
              f"({audit['checked']} checked, "
              f"{len(audit['mismatches'])} mismatches)", flush=True)
        if not audit["ok"]:
            print("NOT publishing: audit failed; partial/progress retained "
                  "for investigation", flush=True)
            return 3

        labels_text, labels_sha = finalize_labels_from_partial(
            labels, records)
        teacher_manifest = {
            "engine": teacher.uci_id_name,
            "verified_binary_sha256": teacher.verified_binary_sha256,
            "uci_id_name": teacher.uci_id_name,
            "uci_id_author": teacher.uci_id_author,
            "binary_path": teacher_kwargs.get("binary", TEACHER_BIN),
            "binary_sha256": teacher.verified_binary_sha256,
            "nodes": TEACHER_NODES,
            "options": dict(TEACHER_OPTIONS),
            "uci_options_seen": list(teacher.uci_options.keys()),
            "syzygy": "disabled (default)",
            "labeling_mode": "go nodes",
            "audit": audit,
            "labels_sha256": labels_sha,
            "labeled_positions": len(labels),
            "resume": {
                "enabled": True,
                "checkpoint_interval": checkpoint_interval,
                "ordered_position_id_sha256":
                    dataset_contract["ordered_position_id_sha256"],
                "resume_count": resume_count,
                "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            },
        }
        # temp files first; publish only after full labeling + audit pass,
        # manifest renamed LAST as the commit point.
        labels_tmp = dataset_dir / "labels.jsonl.tmp"
        manifest_tmp = dataset_dir / "teacher_manifest.json.tmp"
        with open(labels_tmp, "w", encoding="utf-8") as fh:
            fh.write(labels_text)
            fh.flush()
            os.fsync(fh.fileno())
        with open(manifest_tmp, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(teacher_manifest, ensure_ascii=False,
                                indent=2))
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(labels_tmp, dataset_dir / "labels.jsonl")
        os.replace(manifest_tmp, dataset_dir / "teacher_manifest.json")
        print("labels.jsonl + teacher_manifest.json published", flush=True)
        print(json.dumps(teacher_manifest, ensure_ascii=False, indent=2))

        # Partial/progress are no longer needed after a successful publish;
        # failure to delete them is a warning only (final artifacts are the
        # sole identity).
        partial_fh.close()
        for path in (partial_path, dataset_dir / PROGRESS_NAME):
            try:
                path.unlink()
            except OSError:
                print(f"warning: could not remove {path.name}", flush=True)
        return 0
    finally:
        try:
            teacher.close()
        except Exception:
            pass
        try:
            partial_fh.close()
        except Exception:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True,
                        help="dataset dir (shards + manifest)")
    parser.add_argument("--audit-n", type=int, default=AUDIT_N)
    parser.add_argument("--checkpoint-interval", type=int,
                        default=CHECKPOINT_INTERVAL,
                        help="commit a checkpoint every N completed "
                             "positions (default 500; production value)")
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
    if args.checkpoint_interval < 1:
        print("FATAL: --checkpoint-interval must be >= 1", flush=True)
        return 4
    dataset_dir = Path(args.dataset)
    records = load_records(dataset_dir)

    # Final artifacts already present: existing vs-stored behavior. A stale
    # partial/progress alongside them fails closed (never resume into a
    # published dataset).
    stored = load_stored_labels(dataset_dir)
    if stored:
        if (dataset_dir / PARTIAL_NAME).is_file() or \
                (dataset_dir / PROGRESS_NAME).is_file():
            print(f"FATAL: dataset already has final labels AND "
                  f"{PARTIAL_NAME}/{PROGRESS_NAME}; remove the stale "
                  f"partial files manually (never auto-resume into a "
                  f"published dataset)", flush=True)
            return 4
        audit_n = min(args.audit_n, len(records))
        print(f"labeling {len(records)} positions (wsl={args.wsl}, "
              f"audit_mode=vs-stored)", flush=True)
        return label_dataset(dataset_dir, records,
                             teacher_factory=Teacher,
                             teacher_kwargs=dict(
                                 wsl=args.wsl,
                                 binary=args.teacher_binary,
                                 expected_binary_sha256=
                                 args.expected_binary_sha256),
                             audit_n=audit_n,
                             checkpoint_interval=args.checkpoint_interval,
                             stored=stored)

    audit_n = min(args.audit_n, len(records))
    print(f"labeling {len(records)} positions (wsl={args.wsl}, "
          f"audit_mode=fresh-second-pass, "
          f"checkpoint_interval={args.checkpoint_interval})", flush=True)
    try:
        return label_dataset(dataset_dir, records,
                             teacher_factory=Teacher,
                             teacher_kwargs=dict(
                                 wsl=args.wsl,
                                 binary=args.teacher_binary,
                                 expected_binary_sha256=
                                 args.expected_binary_sha256),
                             audit_n=audit_n,
                             checkpoint_interval=args.checkpoint_interval,
                             stored=None)
    except ResumeError as exc:
        print(f"FATAL: {exc}", flush=True)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
