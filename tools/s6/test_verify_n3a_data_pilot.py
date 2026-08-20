#!/usr/bin/env python3
"""verify_n3a_data_pilot provenance verifier unit tests: SHA mismatches,
duplicate source ids, engine replacement, verify/coverage failure, and the
full-success provenance shape. Uses small temp files and mocked subprocesses;
never touches the 29GB archive."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_dataset as bd  # noqa: E402
import train_nnue_probe as probe  # noqa: E402
import verify_n3a_data_pilot as vp  # noqa: E402


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_arena_dir(tmp: Path, n_sources: int = 2,
                   duplicate_id: bool = False) -> Path:
    d = tmp / "arena"
    d.mkdir(exist_ok=True)
    manifest = {}
    for i in range(n_sources):
        key = f"arena-{i}"
        data = f"[Event 's']\n[Result '1-0']\n1. e4 e5 {i}".encode()
        (d / f"{key}.pgn").write_bytes(data)
        manifest[key] = {
            "source_id": f"arena-{i}" if not duplicate_id else "arena-0",
            "source_family": "arena",
            "sha256": sha(data),
        }
    (d / "source_manifest.json").write_text(json.dumps(manifest, indent=1))
    return d


def make_lichess_dir(tmp: Path, pgn_sha: str | None = None,
                     script_sha: str | None = None) -> Path:
    d = tmp / "lichess"
    d.mkdir(exist_ok=True)
    data = b"[Event 's']\n[Result '1-0']\n1. e4 e5 lichess"
    (d / "lichess-standard-rated-v1.pgn").write_bytes(data)
    actual_pgn = sha(data)
    manifest = {
        "source_family": "lichess-standard-rated-v1",
        "source_id": "lichess-standard-rated-v1",
        "pgn_sha256": pgn_sha if pgn_sha is not None else actual_pgn,
        "selection_seed": vp.SEED,
        "games_selected": vp.GAMES_PER_MONTH,
        "script_sha256": script_sha if script_sha is not None
        else sha((Path(vp.__file__).parent / "lichess_select.py")
                 .read_bytes()),
        "official_sha256": {vp.MONTH: vp.EXPECTED_ARCHIVE_SHA},
    }
    (d / "source-manifest.json").write_text(json.dumps(manifest, indent=1))
    return d


def make_dataset_dir(tmp: Path, name: str, canonical_sha: str | None = None) -> Path:
    d = tmp / name
    d.mkdir(exist_ok=True)
    records = [{"position_id": f"p{i:064d}", "marker": "x"}
               for i in range(8)]
    (d / "part-0000.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n"
                for r in records), encoding="utf-8")
    if canonical_sha is None:
        canonical_sha = probe.compute_dataset_sha(records)
    (d / "dataset_manifest.json").write_text(json.dumps({
        "dataset_id": name,
        "dataset_sha256": canonical_sha,
        "sampling_version": 2,
        "source_families": {"arena": 4, "lichess-standard-rated-v1": 4},
        "sources": {},
    }))
    return d


def make_artifacts(tmp: Path) -> dict:
    archive = tmp / "month.pgn.zst"
    archive.write_bytes(b"archive-bytes")
    engine = tmp / "eureka"
    engine.write_bytes(b"engine-bytes")
    arena = make_arena_dir(tmp)
    lichess = make_lichess_dir(tmp)
    # pilot + rebuild with identical records -> identical canonical sha
    pilot = make_dataset_dir(tmp, "pilot")
    rebuild = make_dataset_dir(tmp, "rebuild")
    canonical = json.loads((pilot / "dataset_manifest.json").read_text())[
        "dataset_sha256"]
    # bind dataset manifest sources to the artifacts
    arena_src = {f"arena-{i}": arena_src_sha(i) for i in range(2)}
    lichess_src = {"lichess-standard-rated-v1":
                   sha((tmp / "lichess/lichess-standard-rated-v1.pgn").read_bytes())}
    manifest = json.loads((pilot / "dataset_manifest.json").read_text())
    manifest["sources"] = {**arena_src, **lichess_src}
    (pilot / "dataset_manifest.json").write_text(json.dumps(manifest))
    return {"archive": archive, "engine": engine, "arena": arena,
            "lichess": lichess, "pilot": pilot, "rebuild": rebuild,
            "canonical": canonical}


def arena_src_sha(i: int) -> str:
    return sha(f"[Event 's']\n[Result '1-0']\n1. e4 e5 {i}".encode())


def base_paths(tmp: Path, arts: dict) -> dict:
    return {
        "archive": arts["archive"].resolve(),
        "engine": arts["engine"].resolve(),
        "pilot_dir": arts["pilot"].resolve(),
        "rebuild_dir": arts["rebuild"].resolve(),
        "arena_dir": arts["arena"].resolve(),
        "lichess_dir": arts["lichess"].resolve(),
        "expected_engine_sha": sha(b"engine-bytes"),
    }


class ProvenanceVerifierTests(unittest.TestCase):
    def _run(self, tmp: Path, arts: dict, **overrides):
        paths = base_paths(tmp, arts)
        paths.update(overrides)
        with mock.patch.object(vp, "worktree_clean", return_value=True), \
             mock.patch.object(vp, "git_head", return_value="h" * 40), \
             mock.patch.object(vp, "run_verify_dataset", return_value=0), \
             mock.patch.object(vp, "run_coverage",
                               return_value={"rc": 0, "status":
                                             "DATA_PILOT_PASS"}), \
             mock.patch.object(vp, "EXPECTED_DATASET_SHA",
                               arts["canonical"]), \
             mock.patch.object(vp, "EXPECTED_ARCHIVE_SHA",
                               sha(b"archive-bytes")):
            # Rewrite the lichess manifest so its official_sha256 binding
            # matches the patched (temp-archive) expected SHA.
            arts["lichess"] = make_lichess_dir(Path(tmp))
            paths["lichess_dir"] = arts["lichess"].resolve()
            return vp.build_provenance(paths)

    def test_success_has_all_provenance_fields(self):
        with tempfile.TemporaryDirectory(prefix="s6-n3a-") as tmp:
            arts = make_artifacts(Path(tmp))
            result = self._run(Path(tmp), arts)
            self.assertEqual(result["status"], "DATA_PILOT_PASS")
            self.assertEqual(result["failures"], [])
            prov = result["provenance"]
            for section in ("archive", "lichess_selection", "arena_sources",
                            "dataset", "encoder", "tools", "environment"):
                self.assertIn(section, prov)
            self.assertEqual(
                prov["archive"]["actual_sha256"], sha(b"archive-bytes"))
            self.assertEqual(prov["dataset"]["pilot"]["canonical_sha256"],
                             arts["canonical"])
            self.assertTrue(prov["dataset"]["rebuild_sha_identical"])

    def test_archive_sha_mismatch_fails(self):
        with tempfile.TemporaryDirectory(prefix="s6-n3a-") as tmp:
            arts = make_artifacts(Path(tmp))
            arts["archive"].write_bytes(b"tampered")
            result = self._run(Path(tmp), arts)
            self.assertEqual(result["status"], "DATA_PILOT_FAIL")
            self.assertTrue(any("archive_sha" in f for f in result["failures"]))

    def test_selected_pgn_sha_mismatch_fails(self):
        with tempfile.TemporaryDirectory(prefix="s6-n3a-") as tmp:
            arts = make_artifacts(Path(tmp))
            arts["lichess"] = make_lichess_dir(
                Path(tmp), pgn_sha="f" * 64)
            paths = base_paths(Path(tmp), arts)
            paths["lichess_dir"] = arts["lichess"].resolve()
            with mock.patch.object(vp, "worktree_clean", return_value=True), \
                 mock.patch.object(vp, "git_head", return_value="h" * 40), \
                 mock.patch.object(vp, "run_verify_dataset", return_value=0), \
                 mock.patch.object(vp, "run_coverage",
                                   return_value={"rc": 0, "status":
                                                 "DATA_PILOT_PASS"}), \
                 mock.patch.object(vp, "EXPECTED_DATASET_SHA",
                                   arts["canonical"]):
                result = vp.build_provenance(paths)
            self.assertEqual(result["status"], "DATA_PILOT_FAIL")
            self.assertTrue(any("lichess_pgn_sha" in f
                                for f in result["failures"]))

    def test_script_sha_mismatch_fails(self):
        with tempfile.TemporaryDirectory(prefix="s6-n3a-") as tmp:
            arts = make_artifacts(Path(tmp))
            arts["lichess"] = make_lichess_dir(
                Path(tmp), script_sha="e" * 64)
            paths = base_paths(Path(tmp), arts)
            paths["lichess_dir"] = arts["lichess"].resolve()
            with mock.patch.object(vp, "worktree_clean", return_value=True), \
                 mock.patch.object(vp, "git_head", return_value="h" * 40), \
                 mock.patch.object(vp, "run_verify_dataset", return_value=0), \
                 mock.patch.object(vp, "run_coverage",
                                   return_value={"rc": 0, "status":
                                                 "DATA_PILOT_PASS"}), \
                 mock.patch.object(vp, "EXPECTED_DATASET_SHA",
                                   arts["canonical"]):
                result = vp.build_provenance(paths)
            self.assertEqual(result["status"], "DATA_PILOT_FAIL")
            self.assertTrue(any("lichess_script_sha" in f
                                for f in result["failures"]))

    def test_duplicate_source_id_fails(self):
        with tempfile.TemporaryDirectory(prefix="s6-n3a-") as tmp:
            arts = make_artifacts(Path(tmp))
            arts["arena"] = make_arena_dir(Path(tmp), duplicate_id=True)
            paths = base_paths(Path(tmp), arts)
            paths["arena_dir"] = arts["arena"].resolve()
            with mock.patch.object(vp, "worktree_clean", return_value=True), \
                 mock.patch.object(vp, "git_head", return_value="h" * 40), \
                 mock.patch.object(vp, "run_verify_dataset", return_value=0), \
                 mock.patch.object(vp, "run_coverage",
                                   return_value={"rc": 0, "status":
                                                 "DATA_PILOT_PASS"}), \
                 mock.patch.object(vp, "EXPECTED_DATASET_SHA",
                                   arts["canonical"]):
                result = vp.build_provenance(paths)
            self.assertEqual(result["status"], "DATA_PILOT_FAIL")
            self.assertTrue(any("duplicate_source_id" in f
                                for f in result["failures"]))

    def test_dataset_rebuild_sha_mismatch_fails(self):
        with tempfile.TemporaryDirectory(prefix="s6-n3a-") as tmp:
            arts = make_artifacts(Path(tmp))
            other = probe.compute_dataset_sha([
                {"position_id": f"q{i:064d}", "marker": "x"}
                for i in range(8)])
            arts["rebuild"] = make_dataset_dir(Path(tmp), "rebuild", other)
            paths = base_paths(Path(tmp), arts)
            paths["rebuild_dir"] = arts["rebuild"].resolve()
            with mock.patch.object(vp, "worktree_clean", return_value=True), \
                 mock.patch.object(vp, "git_head", return_value="h" * 40), \
                 mock.patch.object(vp, "run_verify_dataset", return_value=0), \
                 mock.patch.object(vp, "run_coverage",
                                   return_value={"rc": 0, "status":
                                                 "DATA_PILOT_PASS"}), \
                 mock.patch.object(vp, "EXPECTED_DATASET_SHA",
                                   arts["canonical"]):
                result = vp.build_provenance(paths)
            self.assertEqual(result["status"], "DATA_PILOT_FAIL")
            self.assertTrue(any("dataset_rebuild_equal" in f
                                for f in result["failures"]))

    def test_engine_replaced_fails(self):
        with tempfile.TemporaryDirectory(prefix="s6-n3a-") as tmp:
            arts = make_artifacts(Path(tmp))
            paths = base_paths(Path(tmp), arts)
            paths["expected_engine_sha"] = "f" * 64
            with mock.patch.object(vp, "worktree_clean", return_value=True), \
                 mock.patch.object(vp, "git_head", return_value="h" * 40), \
                 mock.patch.object(vp, "run_verify_dataset", return_value=0), \
                 mock.patch.object(vp, "run_coverage",
                                   return_value={"rc": 0, "status":
                                                 "DATA_PILOT_PASS"}), \
                 mock.patch.object(vp, "EXPECTED_DATASET_SHA",
                                   arts["canonical"]):
                result = vp.build_provenance(paths)
            self.assertEqual(result["status"], "DATA_PILOT_FAIL")
            self.assertTrue(any("engine_sha" in f for f in result["failures"]))

    def test_verify_nonzero_fails(self):
        with tempfile.TemporaryDirectory(prefix="s6-n3a-") as tmp:
            arts = make_artifacts(Path(tmp))
            with mock.patch.object(vp, "worktree_clean", return_value=True), \
                 mock.patch.object(vp, "git_head", return_value="h" * 40), \
                 mock.patch.object(vp, "run_verify_dataset", return_value=1), \
                 mock.patch.object(vp, "run_coverage",
                                   return_value={"rc": 0, "status":
                                                 "DATA_PILOT_PASS"}), \
                 mock.patch.object(vp, "EXPECTED_DATASET_SHA",
                                   arts["canonical"]):
                result = vp.build_provenance(base_paths(Path(tmp), arts))
            self.assertEqual(result["status"], "DATA_PILOT_FAIL")
            self.assertTrue(any("verify_dataset_allow_unlabeled" in f
                                for f in result["failures"]))

    def test_coverage_failure_fails(self):
        with tempfile.TemporaryDirectory(prefix="s6-n3a-") as tmp:
            arts = make_artifacts(Path(tmp))
            with mock.patch.object(vp, "worktree_clean", return_value=True), \
                 mock.patch.object(vp, "git_head", return_value="h" * 40), \
                 mock.patch.object(vp, "run_verify_dataset", return_value=0), \
                 mock.patch.object(vp, "run_coverage",
                                   return_value={"rc": 2, "status":
                                                 "DATA_PILOT_FAIL"}), \
                 mock.patch.object(vp, "EXPECTED_DATASET_SHA",
                                   arts["canonical"]):
                result = vp.build_provenance(base_paths(Path(tmp), arts))
            self.assertEqual(result["status"], "DATA_PILOT_FAIL")
            self.assertTrue(any("coverage_analyzer" in f
                                for f in result["failures"]))


if __name__ == "__main__":
    unittest.main()
