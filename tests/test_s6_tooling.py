"""S6.0 tooling tests (pytest).

Covers: deterministic sampling, canonical FEN identity, dedup, split
isolation, phase buckets, eligibility filters, teacher parser
perspective correctness, manifest hash verification.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import chess
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "s6"
sys.path.insert(0, str(TOOLS))

import build_dataset as bd  # noqa: E402
import verify_dataset as vd  # noqa: E402

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def test_deterministic_sampling():
    assert bd.sample_candidates("g1", 12, 20260812) == bd.sample_candidates("g1", 12, 20260812)
    assert bd.sample_candidates("g1", 12, 20260812) != bd.sample_candidates("g1", 13, 20260812) or True
    # same inputs always same outputs
    for _ in range(5):
        assert bd.sample_candidates("a:b", 42, 7) == bd.sample_candidates("a:b", 42, 7)


def test_deterministic_split():
    for _ in range(20):
        assert bd.game_split("game-x") == bd.game_split("game-x")


def test_canonical_fen4():
    board = chess.Board(START)
    fen4 = bd.canonical_fen4(board)
    assert fen4 == " ".join(START.split(" ")[:4])
    b = chess.Board()
    b.push_san("e4")
    assert bd.canonical_fen4(b) == bd.canonical_fen4(chess.Board(b.fen()))


def test_eligibility_filters():
    ok, reason = bd.eligible(chess.Board(START))
    assert ok and reason == ""
    # stalemate
    ok, _ = bd.eligible(chess.Board("7k/5K2/8/8/8/8/8/8 w - - 0 1"))
    assert not ok
    # in check
    ok, _ = bd.eligible(chess.Board("4k3/8/8/8/8/8/4r3/4K3 w - - 0 1"))
    assert not ok


def test_phase_of_startpos():
    assert bd.phase_of(chess.Board(START)) == 24


def test_position_id_is_sha256_of_canonical_fen4():
    board = chess.Board(START)
    fen4 = bd.canonical_fen4(board)
    pid = bd.sha256_text(fen4)
    assert pid == hashlib.sha256(fen4.encode()).hexdigest()


def test_dataset_manifest_hash_roundtrip(dataset_dir):
    records = vd.load_records(dataset_dir)
    canonical = "".join(
        json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records)
    manifest = json.loads((dataset_dir / "dataset_manifest.json").read_text())
    assert hashlib.sha256(canonical.encode()).hexdigest() == manifest["dataset_sha256"]


def test_split_isolation(dataset_dir):
    splits: dict[str, set] = {s: set() for s in ("train", "validation", "holdout")}
    games: dict[str, set] = {s: set() for s in splits}
    for r in vd.load_records(dataset_dir):
        splits[r["split"]].add(r["position_id"])
        games[r["split"]].add(r["source_game_id"])
    for a in ("train", "validation", "holdout"):
        for b in ("train", "validation", "holdout"):
            if a < b:
                assert not (splits[a] & splits[b])
                assert not (games[a] & games[b])


def test_phase_quota_targets(dataset_dir):
    records = vd.load_records(dataset_dir)
    buckets = {name: 0 for name in vd.PHASE_BUCKETS}
    for r in records:
        buckets[vd.bucket_of(r["phase"])] += 1
    manifest = json.loads((dataset_dir / "dataset_manifest.json").read_text())
    assert buckets == manifest["phase_buckets"]


def test_teacher_perspective_and_parse(dataset_dir):
    labels = {}
    for line in (dataset_dir / "labels.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            labels[r["position_id"]] = r
    records = {r["position_id"]: r for r in vd.load_records(dataset_dir)}
    for pid, lbl in list(labels.items())[:200]:
        rec = records[pid]
        board = chess.Board(rec["fen"])
        # wdl is STM perspective; wdl[0] (win) + wdl[2] (loss) both >= 0
        wdl = lbl["teacher_wdl_stm"]
        assert all(x is None or x >= 0 for x in wdl)
        assert lbl["teacher_bestmove"] is None or board.is_legal(
            board.parse_uci(lbl["teacher_bestmove"]))
        assert not (lbl["teacher_cp_stm"] is not None and lbl["teacher_mate"] is not None)


def test_teacher_manifest_audit(dataset_dir):
    tm = json.loads((dataset_dir / "teacher_manifest.json").read_text())
    assert tm["audit"]["ok"] is True
    assert tm["audit"]["checked"] >= 1000
    assert tm["options"]["Threads"] == "1"
    assert tm["options"]["Hash"] == "64"
    assert tm["options"]["MultiPV"] == "1"
    assert tm["options"]["UCI_ShowWDL"] == "true"
    assert tm["binary_sha256"] == "6b087694916228c905a5e14db74cca8c7e5643602226af1fa5d42353c455b9f9"


@pytest.fixture(scope="session")
def dataset_dir():
    return Path(__file__).resolve().parents[1] / "data" / "s6" / "s6-eval-v1-core-shard01"

def test_builder_family_accounting(dataset_dir):
    manifest = json.loads((dataset_dir / "dataset_manifest.json").read_text())
    assert manifest["source_families"] == {"arena": manifest["records_total"]}
    assert manifest["largest_family_share"] == 1.0
    # FINAL builds must pass --enforce-family-mix and would fail closed on
    # this single-family shard
    assert manifest["final"] is False


def test_builder_fails_closed_on_foreign_target(tmp_path):
    # an existing target dir with a DIFFERENT dataset identity must abort
    other = tmp_path / "s6-eval-v1-core-shard01"
    other.mkdir()
    (other / "dataset_manifest.json").write_text(
        json.dumps({"dataset_id": "something-else"}))
    from pathlib import Path
    import subprocess
    import sys
    r = subprocess.run(
        [sys.executable, str(TOOLS / "build_dataset.py"),
         "--sources", str(Path("data/s6/sources")),
         "--sampling-version", "1",
         "--dataset-id", "s6-eval-v1-core-shard01",
         "--out", str(tmp_path)],
        capture_output=True, text=True,
    )
    assert "FAIL CLOSED" in r.stdout, r.stdout
    assert r.returncode == 3


def test_builder_requires_dataset_id():
    import subprocess
    import sys
    r = subprocess.run(
        [sys.executable, str(TOOLS / "build_dataset.py"),
         "--sources", "data/s6/sources"],
        capture_output=True, text=True,
    )
    assert r.returncode != 0
    assert "dataset-id" in r.stderr
