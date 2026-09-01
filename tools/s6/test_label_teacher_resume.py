#!/usr/bin/env python3
"""S10-B2A resumable teacher-labeling tests.

Proves:
- T1  fresh run (17 positions, interval 5) publishes and cleans up
- T2  interrupt + resume final labels are byte-identical to uninterrupted
- T3  crash tail (uncommitted bytes beyond partial_size_bytes) is truncated
- T4  partial shorter than committed size fails closed
- T5  committed-prefix SHA corruption fails closed
- T6  dataset SHA mismatch fails closed
- T7  ordered position-id hash mismatch fails closed
- T8  teacher binary SHA mismatch fails closed
- T9  teacher nodes / Threads / Hash option mismatch fails closed
- T10 duplicate position_id in committed partial fails closed
- T11 unknown position_id in committed partial fails closed
- T12 non-prefix / reordered position_id fails closed
- T13 orphan partial (no progress) fails closed
- T14 orphan progress (no partial) fails closed
- T15 interrupted run never publishes final artifacts
- T16 audit failure retains partial/progress and publishes nothing
- T17 partial-only dataset stays unlabelled for verify_dataset
- T18 final labels serialization stays pid-sorted (existing contract)
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent))

import label_teacher as lt  # noqa: E402

FAKE_SHA = "f" * 64
N = 17
INTERVAL = 5

PHASE_WEIGHTS = {chess.KNIGHT: 1, chess.BISHOP: 1, chess.ROOK: 2,
                 chess.QUEEN: 4}


def phase_of(board: chess.Board) -> int:
    total = sum(PHASE_WEIGHTS.get(p.piece_type, 0)
                for p in board.piece_map().values())
    return min(24, total)


def random_midgame(seed: int) -> str | None:
    rng = random.Random(seed)
    board = chess.Board()
    for _ in range(8):
        legal = list(board.legal_moves)
        if not legal:
            return None
        board.push(rng.choice(legal))
    if board.is_game_over(claim_draw=False) or board.is_check() \
            or not any(board.legal_moves) or not board.is_valid():
        return None
    return board.fen()


def make_dataset(tmp: Path, n: int = N) -> Path:
    d = tmp / "ds"
    d.mkdir()
    records = []
    seed = 0
    while len(records) < n:
        fen = random_midgame(seed)
        seed += 1
        if fen is None:
            continue
        board = chess.Board(fen)
        fen4 = " ".join(fen.split(" ")[:4])
        records.append({
            "position_id": hashlib.sha256(fen4.encode()).hexdigest(),
            "fen": fen,
            "canonical_fen4": fen4,
            "phase": phase_of(board),
            "source_game_id": f"g{len(records)}",
            "split": "train",
        })
    (d / "part-0000.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n"
                for r in records), encoding="utf-8")
    canonical = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True)
                        + "\n" for r in records)
    buckets = {"high": 0, "mid": 0, "low": 0, "zero": 0}
    for r in records:
        ph = r["phase"]
        for name, (lo, hi) in lt.PHASE_BUCKETS.items() if hasattr(lt, "PHASE_BUCKETS") else []:
            pass
        buckets["mid" if 8 <= ph <= 17 else ("high" if ph >= 18 else
                                             ("zero" if ph == 0 else "low"))] += 1
    (d / "dataset_manifest.json").write_text(json.dumps({
        "dataset_id": "syn-b2a",
        "records_total": len(records),
        "dataset_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "phase_buckets": buckets,
    }))
    return d


class FakeTeacher:
    """Deterministic fake teacher: labels derive purely from sha256(fen).

    fail_at: when set to an index, the label() call for that 0-based
    dataset-record index raises RuntimeError ONCE per instance (simulates a
    teacher crash), letting the retry path succeed after respawn.

    audit_mismatch: when True, the SECOND instance (the fresh audit pass)
    returns a corrupted teacher_cp_stm for every position.
    """

    instances = 0
    fail_at: int | None = None
    audit_mismatch = False

    def __init__(self, wsl: bool = True, binary=None,
                 expected_binary_sha256: str | None = None):
        type(self).instances += 1
        self.seq = type(self).instances
        self.uci_id_name = "FakeStockfish"
        self.uci_id_author = "fake"
        self.uci_options = {"Threads": "1"}
        self.verified_binary_sha256 = expected_binary_sha256 or FAKE_SHA
        self.closed = False
        self._call_index = 0
        self._failed_once = False

    def label(self, fen: str) -> dict:
        # fail_at: the FIRST instance dies once at that 0-based call index
        # (teacher crash + respawn retry recovers).
        if type(self).fail_at is not None and self.seq == 1 \
                and not self._failed_once \
                and self._call_index == type(self).fail_at:
            self._failed_once = True
            raise RuntimeError("simulated teacher death")
        self._call_index += 1
        h = hashlib.sha256(fen.encode()).digest()
        offset = 999 if (type(self).audit_mismatch and self.seq == 2) else 0
        return {
            "teacher_cp_stm": (h[0] << 8 | h[1]) - 65536 // 2 + offset,
            "teacher_mate": None,
            "teacher_bestmove": f"{h[2]:02x}{h[3]:02x}{h[4]:02x}{h[5]:02x}",
            "teacher_wdl_stm": [h[6], h[7], h[8]],
            "nodes": lt.TEACHER_NODES,
        }

    def close(self):
        self.closed = True


def _sub_dataset(parent: Path, name: str) -> Path:
    """make_dataset into a NAMED subdirectory (it hardcodes 'ds')."""
    sub = parent / name
    sub.mkdir(parents=True, exist_ok=True)
    return make_dataset(sub)


def run_label(dataset: Path, checkpoint_interval: int = INTERVAL,
              audit_n: int = N) -> int:
    FakeTeacher.instances = 0
    FakeTeacher.fail_at = None
    FakeTeacher.audit_mismatch = False
    argv = ["label_teacher.py", "--dataset", str(dataset), "--native",
            "--checkpoint-interval", str(checkpoint_interval),
            "--audit-n", str(audit_n),
            "--expected-binary-sha256", FAKE_SHA]
    with mock.patch.object(sys, "argv", argv), \
         mock.patch.object(lt, "Teacher", FakeTeacher):
        return lt.main()


def make_committed_state(dataset: Path, n_commit: int,
                         checkpoint_interval: int = INTERVAL) -> None:
    """Produce a real committed checkpoint for the first n_commit records by
    running the pipeline with a fake teacher that raises KeyboardInterrupt
    when it reaches the n_commit-th record (durable state stays at the last
    successful checkpoint, e.g. 5 for n_commit=8, interval=5)."""
    counter = {"i": 0}

    class Interrupting(FakeTeacher):
        def label(self, fen: str):
            if counter["i"] >= n_commit:
                raise KeyboardInterrupt
            counter["i"] += 1
            return FakeTeacher.label(self, fen)

    argv = ["label_teacher.py", "--dataset", str(dataset), "--native",
            "--checkpoint-interval", str(checkpoint_interval),
            "--expected-binary-sha256", FAKE_SHA]
    with mock.patch.object(sys, "argv", argv), \
         mock.patch.object(lt, "Teacher", Interrupting):
        try:
            lt.main()
        except KeyboardInterrupt:
            pass


class ResumeTests(unittest.TestCase):
    def setUp(self):
        FakeTeacher.instances = 0
        FakeTeacher.fail_at = None
        FakeTeacher.audit_mismatch = False

    # T1 -----------------------------------------------------------------
    def test_t1_fresh_run(self):
        with tempfile.TemporaryDirectory(prefix="s10-b2a-") as tmp:
            d = make_dataset(Path(tmp))
            rc = run_label(d)
            self.assertEqual(rc, 0)
            labels = [json.loads(l) for l in
                      (d / "labels.jsonl").read_text().splitlines() if l]
            self.assertEqual(len(labels), N)
            self.assertTrue((d / "teacher_manifest.json").is_file())
            # partial/progress cleaned up after successful publish
            self.assertFalse((d / lt.PARTIAL_NAME).exists())
            self.assertFalse((d / lt.PROGRESS_NAME).exists())
            # resume telemetry present
            tm = json.loads((d / "teacher_manifest.json").read_text())
            self.assertEqual(tm["resume"]["checkpoint_interval"], INTERVAL)
            self.assertEqual(tm["resume"]["resume_count"], 0)
            self.assertEqual(
                tm["resume"]["ordered_position_id_sha256"],
                lt.ordered_position_id_sha256(lt.load_records(d)))

    # T2 -----------------------------------------------------------------
    def test_t2_interrupt_resume_byte_identical(self):
        with tempfile.TemporaryDirectory(prefix="s10-b2a-") as tmp_a, \
             tempfile.TemporaryDirectory(prefix="s10-b2a-") as tmp_b:
            # Run A: uninterrupted
            a = make_dataset(Path(tmp_a))
            rc = run_label(a)
            self.assertEqual(rc, 0)
            text_a = (a / "labels.jsonl").read_bytes()
            # Run B: interrupt at 8 (past checkpoint 5), then resume
            b = make_dataset(Path(tmp_b))
            records = lt.load_records(b)
            self.assertEqual(
                (a / "dataset_manifest.json").read_text(),
                (b / "dataset_manifest.json").read_text())
            make_committed_state(b, n_commit=8)
            prog = json.loads((b / lt.PROGRESS_NAME).read_text())
            self.assertEqual(prog["completed_count"], 5)
            rc = run_label(b)
            self.assertEqual(rc, 0)
            text_b = (b / "labels.jsonl").read_bytes()
            self.assertEqual(text_a, text_b)
            tm = json.loads((b / "teacher_manifest.json").read_text())
            self.assertEqual(tm["resume"]["resume_count"], 1)

    # T3 -----------------------------------------------------------------
    def test_t3_crash_tail_truncated_and_resumed(self):
        with tempfile.TemporaryDirectory(prefix="s10-b2a-") as tmp:
            d = make_dataset(Path(tmp))
            make_committed_state(d, n_commit=8)
            prog = json.loads((d / lt.PROGRESS_NAME).read_text())
            committed = prog["partial_size_bytes"]
            # Simulate a crash window: 2 complete uncommitted lines appended.
            records = lt.load_records(d)
            extra = ""
            for rec in records[prog["completed_count"]:
                               prog["completed_count"] + 2]:
                lbl = FakeTeacher().label(rec["fen"])
                extra += lt.label_line(rec["position_id"], lbl)
            with open(d / lt.PARTIAL_NAME, "a", encoding="utf-8") as fh:
                fh.write(extra)
                fh.flush()
            size_before = (d / lt.PARTIAL_NAME).stat().st_size
            self.assertGreater(size_before, committed)
            rc = run_label(d)
            self.assertEqual(rc, 0)
            # final labels equal the uninterrupted reference
            ref = make_dataset(Path(tempfile.mkdtemp(prefix="s10-b2a-")))
            run_label(ref)
            self.assertEqual((d / "labels.jsonl").read_bytes(),
                             (ref / "labels.jsonl").read_bytes())

    # T4 -----------------------------------------------------------------
    def test_t4_truncated_committed_partial_fails_closed(self):
        with tempfile.TemporaryDirectory(prefix="s10-b2a-") as tmp:
            d = make_dataset(Path(tmp))
            make_committed_state(d, n_commit=8)
            prog = json.loads((d / lt.PROGRESS_NAME).read_text())
            committed = prog["partial_size_bytes"]
            # Lose committed bytes.
            data = (d / lt.PARTIAL_NAME).read_bytes()
            (d / lt.PARTIAL_NAME).write_bytes(data[:committed - 10])
            rc = run_label(d)
            self.assertEqual(rc, 4, "resume validation must FAIL CLOSED")
            self.assertFalse((d / "labels.jsonl").exists())

    # T5 -----------------------------------------------------------------
    def test_t5_committed_sha_corruption_fails_closed(self):
        with tempfile.TemporaryDirectory(prefix="s10-b2a-") as tmp:
            d = make_dataset(Path(tmp))
            make_committed_state(d, n_commit=8)
            prog = json.loads((d / lt.PROGRESS_NAME).read_text())
            committed = prog["partial_size_bytes"]
            data = bytearray((d / lt.PARTIAL_NAME).read_bytes())
            data[committed - 5] ^= 0xFF  # flip a committed byte
            (d / lt.PARTIAL_NAME).write_bytes(bytes(data))
            rc = run_label(d)
            self.assertEqual(rc, 4, "resume validation must FAIL CLOSED")

    # T6 -----------------------------------------------------------------
    def test_t6_dataset_sha_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory(prefix="s10-b2a-") as tmp:
            d = make_dataset(Path(tmp))
            make_committed_state(d, n_commit=8)
            prog = json.loads((d / lt.PROGRESS_NAME).read_text())
            prog["dataset_sha256"] = "0" * 64
            (d / lt.PROGRESS_NAME).write_text(json.dumps(prog))
            rc = run_label(d)
            self.assertEqual(rc, 4, "resume validation must FAIL CLOSED")

    # T7 -----------------------------------------------------------------
    def test_t7_ordered_pid_hash_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory(prefix="s10-b2a-") as tmp:
            d = make_dataset(Path(tmp))
            make_committed_state(d, n_commit=8)
            # Same record count, same dataset SHA, different ORDER ->
            # ordered_position_id_sha256 must catch it.
            records = lt.load_records(d)
            shuffled = list(records)
            shuffled[0], shuffled[1] = shuffled[1], shuffled[0]
            prog = json.loads((d / lt.PROGRESS_NAME).read_text())
            prog["ordered_position_id_sha256"] = \
                lt.ordered_position_id_sha256(shuffled)
            (d / lt.PROGRESS_NAME).write_text(json.dumps(prog))
            rc = run_label(d)
            self.assertEqual(rc, 4, "resume validation must FAIL CLOSED")

    # T8 -----------------------------------------------------------------
    def test_t8_teacher_binary_sha_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory(prefix="s10-b2a-") as tmp:
            d = make_dataset(Path(tmp))
            make_committed_state(d, n_commit=8)
            prog = json.loads((d / lt.PROGRESS_NAME).read_text())
            prog["teacher_binary_sha256"] = "a" * 64
            (d / lt.PROGRESS_NAME).write_text(json.dumps(prog))
            rc = run_label(d)
            self.assertEqual(rc, 4, "resume validation must FAIL CLOSED")

    # T9 -----------------------------------------------------------------
    def test_t9_teacher_option_and_node_mismatches_fail_closed(self):
        for field, value in (
                ("teacher_nodes", 8192),
                ("teacher_options", {"Threads": "2", "Hash": "64",
                                     "MultiPV": "1", "UCI_ShowWDL": "true"}),
                ("teacher_options", {"Threads": "1", "Hash": "128",
                                     "MultiPV": "1", "UCI_ShowWDL": "true"}),
        ):
            with tempfile.TemporaryDirectory(prefix="s10-b2a-") as tmp:
                d = make_dataset(Path(tmp))
                make_committed_state(d, n_commit=8)
                prog = json.loads((d / lt.PROGRESS_NAME).read_text())
                prog[field] = value
                (d / lt.PROGRESS_NAME).write_text(json.dumps(prog))
                rc = run_label(d)
                self.assertEqual(
                    rc, 4,
                    f"{field}={value!r} mismatch must FAIL CLOSED")

    # T10 ----------------------------------------------------------------
    def test_t10_duplicate_pid_fails_closed(self):
        with tempfile.TemporaryDirectory(prefix="s10-b2a-") as tmp:
            d = make_dataset(Path(tmp))
            make_committed_state(d, n_commit=8)
            records = lt.load_records(d)
            prog = json.loads((d / lt.PROGRESS_NAME).read_text())
            lines = (d / lt.PARTIAL_NAME).read_text().splitlines()
            # replace line 1's pid with line 0's pid -> duplicate
            rec = json.loads(lines[1])
            rec["position_id"] = json.loads(lines[0])["position_id"]
            lines[1] = json.dumps(rec, sort_keys=True)
            text = "\n".join(lines) + "\n"
            (d / lt.PARTIAL_NAME).write_text(text)
            prog["partial_size_bytes"] = len(text.encode())
            prog["partial_labels_sha256"] = hashlib.sha256(
                text.encode()).hexdigest()
            prog["completed_count"] = len(lines)
            (d / lt.PROGRESS_NAME).write_text(json.dumps(prog))
            rc = run_label(d)
            self.assertEqual(rc, 4, "resume validation must FAIL CLOSED")

    # T11 ----------------------------------------------------------------
    def test_t11_unknown_pid_fails_closed(self):
        with tempfile.TemporaryDirectory(prefix="s10-b2a-") as tmp:
            d = make_dataset(Path(tmp))
            make_committed_state(d, n_commit=8)
            lines = (d / lt.PARTIAL_NAME).read_text().splitlines()
            rec = json.loads(lines[0])
            rec["position_id"] = "e" * 64  # not in dataset
            lines[0] = json.dumps(rec, sort_keys=True)
            text = "\n".join(lines) + "\n"
            (d / lt.PARTIAL_NAME).write_text(text)
            prog = json.loads((d / lt.PROGRESS_NAME).read_text())
            prog["partial_size_bytes"] = len(text.encode())
            prog["partial_labels_sha256"] = hashlib.sha256(
                text.encode()).hexdigest()
            (d / lt.PROGRESS_NAME).write_text(json.dumps(prog))
            rc = run_label(d)
            self.assertEqual(rc, 4, "resume validation must FAIL CLOSED")

    # T12 ----------------------------------------------------------------
    def test_t12_reordered_pid_fails_closed(self):
        with tempfile.TemporaryDirectory(prefix="s10-b2a-") as tmp:
            d = make_dataset(Path(tmp))
            make_committed_state(d, n_commit=8)
            lines = (d / lt.PARTIAL_NAME).read_text().splitlines()
            # dataset: A B C ...; swap lines 0 and 1 -> non-prefix order
            lines[0], lines[1] = lines[1], lines[0]
            text = "\n".join(lines) + "\n"
            (d / lt.PARTIAL_NAME).write_text(text)
            prog = json.loads((d / lt.PROGRESS_NAME).read_text())
            prog["partial_size_bytes"] = len(text.encode())
            prog["partial_labels_sha256"] = hashlib.sha256(
                text.encode()).hexdigest()
            (d / lt.PROGRESS_NAME).write_text(json.dumps(prog))
            rc = run_label(d)
            self.assertEqual(rc, 4, "resume validation must FAIL CLOSED")

    # T13 ----------------------------------------------------------------
    def test_t13_orphan_partial_fails_closed(self):
        with tempfile.TemporaryDirectory(prefix="s10-b2a-") as tmp:
            d = make_dataset(Path(tmp))
            make_committed_state(d, n_commit=8)
            (d / lt.PROGRESS_NAME).unlink()
            rc = run_label(d)
            self.assertEqual(rc, 4, "resume validation must FAIL CLOSED")

    # T14 ----------------------------------------------------------------
    def test_t14_orphan_progress_fails_closed(self):
        with tempfile.TemporaryDirectory(prefix="s10-b2a-") as tmp:
            d = make_dataset(Path(tmp))
            make_committed_state(d, n_commit=8)
            (d / lt.PARTIAL_NAME).unlink()
            rc = run_label(d)
            self.assertEqual(rc, 4, "resume validation must FAIL CLOSED")

    # T15 ----------------------------------------------------------------
    def test_t15_interrupted_run_never_publishes(self):
        with tempfile.TemporaryDirectory(prefix="s10-b2a-") as tmp:
            d = make_dataset(Path(tmp))
            make_committed_state(d, n_commit=8)
            self.assertTrue((d / lt.PARTIAL_NAME).is_file())
            self.assertTrue((d / lt.PROGRESS_NAME).is_file())
            self.assertFalse((d / "labels.jsonl").exists())
            self.assertFalse((d / "teacher_manifest.json").exists())

    # T16 ----------------------------------------------------------------
    def test_t16_audit_failure_retains_partial_and_publishes_nothing(self):
        with tempfile.TemporaryDirectory(prefix="s10-b2a-") as tmp:
            d = make_dataset(Path(tmp))
            FakeTeacher.audit_mismatch = True
            argv = ["label_teacher.py", "--dataset", str(d), "--native",
                    "--checkpoint-interval", str(INTERVAL),
                    "--expected-binary-sha256", FAKE_SHA]
            FakeTeacher.instances = 0
            with mock.patch.object(sys, "argv", argv), \
                 mock.patch.object(lt, "Teacher", FakeTeacher):
                rc = lt.main()
            self.assertEqual(rc, 3)
            self.assertFalse((d / "labels.jsonl").exists())
            self.assertFalse((d / "teacher_manifest.json").exists())
            # partial/progress retained for investigation (completed 17)
            prog = json.loads((d / lt.PROGRESS_NAME).read_text())
            self.assertEqual(prog["completed_count"], N)

    # T17 ----------------------------------------------------------------
    def test_t17_partial_only_dataset_stays_unlabelled_for_verify(self):
        import verify_dataset as vd
        import argparse
        with tempfile.TemporaryDirectory(prefix="s10-b2a-") as tmp:
            d = make_dataset(Path(tmp))
            make_committed_state(d, n_commit=8)
            # without --allow-unlabeled it must FAIL (labels missing)
            rc = vd.verify(argparse.Namespace(dataset=str(d),
                                              allow_unlabeled=False))
            self.assertEqual(rc, 1)
            # with --allow-unlabeled it passes (partial is not a label set)
            rc = vd.verify(argparse.Namespace(dataset=str(d),
                                              allow_unlabeled=True))
            self.assertEqual(rc, 0)

    # T18 ----------------------------------------------------------------
    def test_t18_final_labels_are_pid_sorted(self):
        with tempfile.TemporaryDirectory(prefix="s10-b2a-") as tmp:
            d = make_dataset(Path(tmp))
            run_label(d)
            pids = [json.loads(l)["position_id"] for l in
                    (d / "labels.jsonl").read_text().splitlines() if l]
            self.assertEqual(pids, sorted(pids))
            self.assertEqual(len(pids), N)

    # T19 (extra): ordered PID SHA serialization is frozen ----------------
    def test_t19_ordered_pid_sha_serialization(self):
        with tempfile.TemporaryDirectory(prefix="s10-b2a-") as tmp:
            d = make_dataset(Path(tmp))
            records = lt.load_records(d)
            expected = hashlib.sha256(
                "\n".join(r["position_id"] for r in records)
                .encode("utf-8")).hexdigest()
            self.assertEqual(lt.ordered_position_id_sha256(records),
                             expected)

    # T20 (extra): teacher death retry still works with checkpointing ----
    def test_t20_teacher_death_retry_recovers(self):
        with tempfile.TemporaryDirectory(prefix="s10-b2a-") as tmp:
            d = make_dataset(Path(tmp))
            FakeTeacher.fail_at = 3
            FakeTeacher.instances = 0
            argv = ["label_teacher.py", "--dataset", str(d), "--native",
                    "--checkpoint-interval", str(INTERVAL),
                    "--expected-binary-sha256", FAKE_SHA]
            with mock.patch.object(sys, "argv", argv), \
                 mock.patch.object(lt, "Teacher", FakeTeacher):
                rc = lt.main()
            self.assertEqual(rc, 0)
            ref = make_dataset(Path(tempfile.mkdtemp(prefix="s10-b2a-")))
            run_label(ref)
            self.assertEqual((d / "labels.jsonl").read_bytes(),
                             (ref / "labels.jsonl").read_bytes())


class PreflightTests(unittest.TestCase):
    """S10-B2A Repair 1: frozen-dataset preflight integrity gate.

    Every failure path must FAIL CLOSED with ZERO Teacher instances
    constructed and no partial/progress mutation."""

    def setUp(self):
        FakeTeacher.instances = 0
        FakeTeacher.fail_at = None
        FakeTeacher.audit_mismatch = False

    def _run(self, d: Path, checkpoint_interval: int = INTERVAL) -> int:
        FakeTeacher.instances = 0
        argv = ["label_teacher.py", "--dataset", str(d), "--native",
                "--checkpoint-interval", str(checkpoint_interval),
                "--expected-binary-sha256", FAKE_SHA]
        with mock.patch.object(sys, "argv", argv), \
             mock.patch.object(lt, "Teacher", FakeTeacher):
            try:
                return lt.main()
            except lt.ResumeError:
                return 4

    def _rewrite_manifest(self, d: Path, **changes) -> None:
        m = json.loads((d / "dataset_manifest.json").read_text())
        m.update(changes)
        (d / "dataset_manifest.json").write_text(json.dumps(m))

    # T21: manifest records_total mismatch -------------------------------
    def test_t21_records_total_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory(prefix="s10-b2a-r1-") as tmp:
            d = make_dataset(Path(tmp))
            self._rewrite_manifest(d, records_total=N + 1)
            rc = self._run(d)
            self.assertEqual(rc, 4)
            self.assertEqual(FakeTeacher.instances, 0,
                             "no Teacher may be constructed on preflight "
                             "failure")
            self.assertFalse((d / lt.PARTIAL_NAME).exists())
            self.assertFalse((d / "labels.jsonl").exists())

    # T22: mutate one dataset record, manifest unchanged ------------------
    def test_t22_mutated_record_sha_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory(prefix="s10-b2a-r1-") as tmp:
            d = make_dataset(Path(tmp))
            lines = (d / "part-0000.jsonl").read_text().splitlines()
            rec = json.loads(lines[0])
            rec["fen"] = rec["fen"].replace("w KQkq", "b KQkq") \
                if "w KQkq" in rec["fen"] else rec["fen"] + " "
            lines[0] = json.dumps(rec, ensure_ascii=False, sort_keys=True)
            (d / "part-0000.jsonl").write_text("\n".join(lines) + "\n")
            rc = self._run(d)
            self.assertEqual(rc, 4)
            self.assertEqual(FakeTeacher.instances, 0)
            self.assertFalse((d / lt.PARTIAL_NAME).exists())

    # T23: remove one record line (count + SHA both drift) ----------------
    def test_t23_removed_record_fails_closed(self):
        with tempfile.TemporaryDirectory(prefix="s10-b2a-r1-") as tmp:
            d = make_dataset(Path(tmp))
            lines = (d / "part-0000.jsonl").read_text().splitlines()
            (d / "part-0000.jsonl").write_text(
                "\n".join(lines[:-1]) + "\n")
            rc = self._run(d)
            self.assertEqual(rc, 4)
            self.assertEqual(FakeTeacher.instances, 0)
            self.assertFalse((d / lt.PARTIAL_NAME).exists())

    # T23b: mutated local shards also fail closed on RESUME ---------------
    def test_t23b_mutation_fails_closed_on_resume_before_partial_touch(self):
        with tempfile.TemporaryDirectory(prefix="s10-b2a-r1-") as tmp:
            d = make_dataset(Path(tmp))
            make_committed_state(d, n_commit=8)
            prog_before = (d / lt.PROGRESS_NAME).read_bytes()
            partial_before = (d / lt.PARTIAL_NAME).read_bytes()
            # Mutate the dataset after the checkpoint exists.
            lines = (d / "part-0000.jsonl").read_text().splitlines()
            rec = json.loads(lines[0])
            rec["phase"] = (rec["phase"] + 1) % 25
            lines[0] = json.dumps(rec, ensure_ascii=False, sort_keys=True)
            (d / "part-0000.jsonl").write_text("\n".join(lines) + "\n")
            rc = self._run(d)
            self.assertEqual(rc, 4)
            self.assertEqual(FakeTeacher.instances, 0)
            # partial/progress untouched by the failed resume
            self.assertEqual((d / lt.PROGRESS_NAME).read_bytes(),
                             prog_before)
            self.assertEqual((d / lt.PARTIAL_NAME).read_bytes(),
                             partial_before)
            self.assertFalse((d / "labels.jsonl").exists())

    # T24: duplicate position_id in dataset -------------------------------
    def test_t24_duplicate_dataset_pid_fails_closed(self):
        with tempfile.TemporaryDirectory(prefix="s10-b2a-r1-") as tmp:
            d = make_dataset(Path(tmp))
            lines = (d / "part-0000.jsonl").read_text().splitlines()
            rec0 = json.loads(lines[0])
            rec1 = json.loads(lines[1])
            rec1["position_id"] = rec0["position_id"]
            lines[1] = json.dumps(rec1, ensure_ascii=False, sort_keys=True)
            text = "\n".join(lines) + "\n"
            (d / "part-0000.jsonl").write_text(text)
            # keep the manifest consistent with the mutated records so the
            # SHA check passes and the PID-uniqueness gate is what fires
            self._rewrite_manifest(
                d, records_total=N,
                dataset_sha256=hashlib.sha256(text.encode()).hexdigest())
            rc = self._run(d)
            self.assertEqual(rc, 4)
            self.assertEqual(FakeTeacher.instances, 0)

    # T25: valid dataset passes preflight; full pipeline still works ------
    def test_t25_valid_dataset_preflight_passes(self):
        with tempfile.TemporaryDirectory(prefix="s10-b2a-r1-") as tmp:
            d = make_dataset(Path(tmp))
            rc = self._run(d)
            self.assertEqual(rc, 0)
            self.assertGreater(FakeTeacher.instances, 0)
            self.assertTrue((d / "labels.jsonl").is_file())

    # P2: checkpoint interval is locked across resume ----------------------
    def test_p2_checkpoint_interval_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory(prefix="s10-b2a-r1-") as tmp:
            d = make_dataset(Path(tmp))
            make_committed_state(d, n_commit=8)  # interval 5, completed 5
            # resume with a DIFFERENT interval must fail closed
            rc = self._run(d, checkpoint_interval=7)
            self.assertEqual(rc, 4)
            self.assertEqual(FakeTeacher.instances, 0)
            self.assertFalse((d / "labels.jsonl").exists())
            # the same interval still resumes fine
            rc = self._run(d, checkpoint_interval=INTERVAL)
            self.assertEqual(rc, 0)
            tm = json.loads((d / "teacher_manifest.json").read_text())
            self.assertEqual(tm["resume"]["resume_count"], 1)


class ParallelWorkersTests(unittest.TestCase):
    """S10-E2 Repair: N-worker coordinator equivalence + resume + reuse."""

    def _run(self, d: Path, workers: int = 3,
             checkpoint_interval: int = INTERVAL,
             parent_labels: Path | None = None) -> int:
        FakeTeacher.instances = 0
        FakeTeacher.fail_at = None
        FakeTeacher.audit_mismatch = False
        argv = ["label_teacher.py", "--dataset", str(d), "--native",
                "--checkpoint-interval", str(checkpoint_interval),
                "--audit-n", str(N),
                "--expected-binary-sha256", FAKE_SHA,
                "--workers", str(workers)]
        if parent_labels is not None:
            argv += ["--reuse-parent-labels", str(parent_labels)]
        with mock.patch.object(sys, "argv", argv), \
             mock.patch.object(lt, "Teacher", FakeTeacher):
            try:
                return lt.main()
            except lt.ResumeError:
                return 4

    def test_parallel_equals_serial_labels(self):
        with tempfile.TemporaryDirectory(prefix="s10-e2-par-") as tmp:
            serial_d = _sub_dataset(Path(tmp), "serial")
            rc = run_label(serial_d)
            self.assertEqual(rc, 0)
            par_d = _sub_dataset(Path(tmp), "parallel")
            rc = self._run(par_d, workers=3)
            self.assertEqual(rc, 0)
            self.assertEqual(
                (serial_d / "labels.jsonl").read_bytes(),
                (par_d / "labels.jsonl").read_bytes())
            tm = json.loads(
                (par_d / "teacher_manifest.json").read_text())
            self.assertEqual(tm["workers"]["count"], 3)

    def test_parallel_resume_after_interrupt(self):
        with tempfile.TemporaryDirectory(prefix="s10-e2-parres-") as tmp:
            d = make_dataset(Path(tmp))
            make_committed_state(d, n_commit=8)  # serial crash state
            # resume the SAME partial with the parallel coordinator
            rc = self._run(d, workers=3)
            self.assertEqual(rc, 0)
            ref = _sub_dataset(Path(tempfile.mkdtemp(prefix="s10-e2-ref-")), "ref")
            run_label(ref)
            self.assertEqual((d / "labels.jsonl").read_bytes(),
                             (ref / "labels.jsonl").read_bytes())

    def test_parent_reuse_skips_and_records(self):
        with tempfile.TemporaryDirectory(prefix="s10-e2-reuse-") as tmp:
            # build a "parent" dataset (first 10 records), label it serially
            parent_d = _sub_dataset(Path(tmp), "parent")
            rc = run_label(parent_d)
            self.assertEqual(rc, 0)
            parent_labels = parent_d / "labels.jsonl"
            # full dataset shares the first records' ids? make_dataset
            # generates unique ids per dir, so craft the overlap manually:
            # label the full dataset with reuse pointing at a labels file
            # covering a SUBSET of its ids.
            full_d = _sub_dataset(Path(tmp), "full")
            recs = lt.load_records(full_d)
            subset_ids = {r["position_id"] for r in recs[:6]}
            kept = [json.loads(l) for l in
                    parent_labels.read_text(encoding="utf-8").splitlines()
                    if l.strip()]
            fake_subset = [
                {"position_id": pid,
                 "teacher_cp_stm": 1, "teacher_mate": None,
                 "teacher_bestmove": "e2e4",
                 "teacher_wdl_stm": [None, None, None],
                 "nodes": lt.TEACHER_NODES}
                for pid in subset_ids]
            sub_path = Path(tmp) / "subset_labels.jsonl"
            sub_path.write_text("".join(
                json.dumps(r, sort_keys=True) + "\n"
                for r in fake_subset), encoding="utf-8")
            rc = self._run(full_d, workers=3, parent_labels=sub_path)
            self.assertEqual(rc, 0)
            tm = json.loads((full_d / "teacher_manifest.json").read_text())
            self.assertEqual(tm["reused_parent_labels"]["count"], 6)
            # reused positions carry the subset values verbatim
            out = {json.loads(l)["position_id"]: json.loads(l) for l in
                   (full_d / "labels.jsonl").read_text(
                       encoding="utf-8").splitlines() if l.strip()}
            for pid in subset_ids:
                self.assertEqual(out[pid]["teacher_cp_stm"], 1)
                self.assertEqual(out[pid]["teacher_bestmove"], "e2e4")
            # non-parent positions were really labeled
            other = [r for r in recs if r["position_id"] not in subset_ids]
            for r in other:
                self.assertNotEqual(
                    out[r["position_id"]]["teacher_cp_stm"], 1)

    def test_parent_reuse_rejects_unknown_positions(self):
        with tempfile.TemporaryDirectory(prefix="s10-e2-rej-") as tmp:
            d = make_dataset(Path(tmp))
            bad = Path(tmp) / "bad.jsonl"
            bad.write_text(json.dumps({
                "position_id": "f" * 64, "teacher_cp_stm": 0,
                "teacher_mate": None, "teacher_bestmove": "a1a1",
                "teacher_wdl_stm": [None, None, None],
                "nodes": lt.TEACHER_NODES}) + "\n", encoding="utf-8")
            rc = self._run(d, workers=2, parent_labels=bad)
            self.assertEqual(rc, 4)

    def test_parent_reuse_requires_workers(self):
        with tempfile.TemporaryDirectory(prefix="s10-e2-ser-") as tmp:
            d = make_dataset(Path(tmp))
            parent_d = _sub_dataset(Path(tmp), "p")
            run_label(parent_d)
            FakeTeacher.instances = 0
            argv = ["label_teacher.py", "--dataset", str(d), "--native",
                    "--checkpoint-interval", str(INTERVAL),
                    "--audit-n", str(N),
                    "--expected-binary-sha256", FAKE_SHA,
                    "--workers", "1",
                    "--reuse-parent-labels",
                    str(parent_d / "labels.jsonl")]
            with mock.patch.object(sys, "argv", argv), \
                 mock.patch.object(lt, "Teacher", FakeTeacher):
                rc = lt.main()
            self.assertEqual(rc, 4)


if __name__ == "__main__":
    unittest.main()
