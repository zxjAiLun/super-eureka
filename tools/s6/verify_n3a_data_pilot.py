#!/usr/bin/env python3
"""S6-N3A provenance closure verifier (read-only, fail-closed).

Re-verifies the data pilot from the EXISTING local artifacts WITHOUT
re-downloading, re-selecting, or re-building the dataset. Reuses the shared
verification functions from build_dataset / analyze_nnue_coverage and
re-invokes verify_dataset + the coverage analyzer internally (never trusts a
caller-supplied rc/SHA/gate boolean).

Any mismatch in SHA / counts / source binding / verify / coverage writes
DATA_PILOT_FAIL and returns 2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_dataset as bd  # noqa: E402
import train_nnue_probe as probe  # noqa: E402

MONTH = "2026-07"
ARCHIVE_URL = ("https://database.lichess.org/standard/"
               "lichess_db_standard_rated_2026-07.pgn.zst")
EXPECTED_ARCHIVE_SHA = "68738b1c448f051dc8d42db645d5b01749988a3bc1c24981adfe44ea92060dc7"
EXPECTED_DATASET_SHA = "5501240e9fd30414cde204038ea0b1e94d20f0029cbeb796d69885375a0683af"
SEED = 20260812
GAMES_PER_MONTH = 2000
SAMPLING_VERSION = 2
LICHESS_SOURCE_ID = "lichess-standard-rated-v1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head() -> str:
    proc = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True)
    return proc.stdout.strip()


def worktree_clean() -> bool:
    proc = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"],
                          capture_output=True, text=True, check=True)
    return proc.stdout.strip() == ""


def tool_shas() -> dict:
    names = ["run_n3a_data_pilot.py", "analyze_nnue_coverage.py",
             "build_dataset.py", "verify_dataset.py", "lichess_select.py"]
    return {n: sha256_file(Path(__file__).parent / n) for n in names}


def env_versions() -> dict:
    return {
        "python": sys.version.split()[0],
        "python_chess": __import__("chess").__version__,
        "zstandard": __import__("zstandard").__version__,
    }


def run_check(name: str, ok: bool, detail: str, failures: list[str]) -> None:
    if not ok:
        failures.append(f"{name}: {detail}")


def check_archive(archive: Path, failures: list[str]) -> dict:
    actual = sha256_file(archive)
    info = {
        "month": MONTH,
        "url": ARCHIVE_URL,
        "expected_sha256": EXPECTED_ARCHIVE_SHA,
        "actual_sha256": actual,
        "bytes": archive.stat().st_size,
    }
    run_check("archive_sha", actual == EXPECTED_ARCHIVE_SHA,
              f"{actual[:16]} != {EXPECTED_ARCHIVE_SHA[:16]}", failures)
    return info


def check_lichess(lichess_dir: Path, failures: list[str]) -> dict:
    manifest = json.loads(
        (lichess_dir / "source-manifest.json").read_text(encoding="utf-8"))
    pgn = lichess_dir / "lichess-standard-rated-v1.pgn"
    pgn_sha = sha256_file(pgn)
    run_check("lichess_pgn_sha", pgn_sha == manifest.get("pgn_sha256"),
              f"{pgn_sha[:16]} != manifest {manifest.get('pgn_sha256', '?')[:16]}",
              failures)
    run_check("lichess_seed", manifest.get("selection_seed") == SEED,
              f"seed {manifest.get('selection_seed')} != {SEED}", failures)
    run_check("lichess_games", manifest.get("games_selected") == GAMES_PER_MONTH,
              f"games {manifest.get('games_selected')} != {GAMES_PER_MONTH}",
              failures)
    expected_script = sha256_file(Path(__file__).parent / "lichess_select.py")
    run_check("lichess_script_sha",
              manifest.get("script_sha256") == expected_script,
              f"script_sha256 {manifest.get('script_sha256', '?')[:16]} != "
              f"{expected_script[:16]}", failures)
    off = manifest.get("official_sha256", {})
    run_check("lichess_archive_sha_binding",
              off.get(MONTH) == EXPECTED_ARCHIVE_SHA,
              f"manifest official_sha256[{MONTH}] "
              f"{off.get(MONTH, '?')[:16]} != {EXPECTED_ARCHIVE_SHA[:16]}",
              failures)
    return {
        "source_id": manifest.get("source_id"),
        "source_family": manifest.get("source_family"),
        "pgn_sha256": pgn_sha,
        "pgn_bytes": pgn.stat().st_size,
        "manifest_sha256": sha256_file(lichess_dir / "source-manifest.json"),
        "selection_seed": manifest.get("selection_seed"),
        "games_selected": manifest.get("games_selected"),
        "script_sha256": manifest.get("script_sha256"),
        "official_sha256": off,
    }


def check_arena_sources(arena_dir: Path, failures: list[str]) -> dict:
    agg = arena_dir / "source_manifest.json"
    manifest = json.loads(agg.read_text(encoding="utf-8"))
    seen_ids: set[str] = set()
    sources: list[dict] = []
    for key, entry in manifest.items():
        src_id = entry["source_id"]
        if src_id in seen_ids:
            run_check(f"duplicate_source_id_{src_id}", False,
                      "duplicate source_id", failures)
        seen_ids.add(src_id)
        path = arena_dir / f"{key}.pgn"
        sources.append({
            "source_key": key,
            "source_id": src_id,
            "source_family": entry.get("source_family"),
            "pgn_sha256": entry.get("sha256"),
            "actual_pgn_sha256": sha256_file(path) if path.is_file() else None,
            "pgn_bytes": path.stat().st_size if path.is_file() else None,
        })
    for s in sources:
        run_check(f"arena_pgn_sha_{s['source_key']}",
                  s["actual_pgn_sha256"] == s["pgn_sha256"],
                  f"{s['source_key']} sha mismatch", failures)
    return {
        "aggregate_manifest_sha256": sha256_file(agg),
        "source_count": len(sources),
        "sources": sources,
    }


def check_dataset(dataset_dir: Path, failures: list[str]) -> dict:
    manifest = json.loads(
        (dataset_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    canonical = manifest["dataset_sha256"]
    records = []
    for shard in sorted(dataset_dir.glob("part-*.jsonl")):
        for line in shard.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    recomputed = probe.compute_dataset_sha(records)
    run_check(f"canonical_sha_{dataset_dir.name}", recomputed == canonical,
              f"{recomputed[:16]} != manifest {canonical[:16]}", failures)
    return {
        "dataset_id": manifest["dataset_id"],
        "records": len(records),
        "canonical_sha256": canonical,
        "manifest_file_sha256": sha256_file(dataset_dir / "dataset_manifest.json"),
        "sampling_version": manifest.get("sampling_version"),
        "source_families": manifest.get("source_families"),
        "sources": manifest.get("sources"),
    }


def run_verify_dataset(dataset_dir: Path) -> int:
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).parent / "verify_dataset.py"),
         "--dataset", str(dataset_dir), "--allow-unlabeled"],
        capture_output=True, text=True, timeout=1800)
    return proc.returncode


def run_coverage(engine: Path, dataset_dir: Path, arena_dir: Path,
                 lichess_dir: Path, rebuilt_sha: str) -> dict:
    tmp_out = Path(os.environ.get("TMPDIR", "/tmp")) / "s6-n3a-provenance-coverage.json"
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).parent / "analyze_nnue_coverage.py"),
         "--engine", str(engine), "--dataset", str(dataset_dir),
         "--sources", str(arena_dir), str(lichess_dir), "--gate",
         "--rebuilt-sha", rebuilt_sha, "--verify-rc", "0",
         "--out", str(tmp_out)],
        capture_output=True, text=True, timeout=3600)
    status = None
    if tmp_out.exists():
        status = json.loads(tmp_out.read_text()).get("status")
        tmp_out.unlink()
    return {"rc": proc.returncode, "status": status}


def cargo_version() -> str:
    for line in (REPO / "Cargo.toml").read_text().splitlines():
        line = line.strip()
        if line.startswith("version ="):
            return line.split("=", 1)[1].strip().strip('"')
    return "unknown"


def build_provenance(paths: dict) -> dict:
    failures: list[str] = []
    archive = check_archive(paths["archive"], failures)
    lichess = check_lichess(paths["lichess_dir"], failures)
    arena = check_arena_sources(paths["arena_dir"], failures)
    pilot = check_dataset(paths["pilot_dir"], failures)
    rebuild = check_dataset(paths["rebuild_dir"], failures)

    run_check("dataset_rebuild_equal",
              pilot["canonical_sha256"] == rebuild["canonical_sha256"],
              f"pilot {pilot['canonical_sha256'][:16]} != "
              f"rebuild {rebuild['canonical_sha256'][:16]}", failures)
    run_check("dataset_frozen_sha",
              pilot["canonical_sha256"] == EXPECTED_DATASET_SHA,
              f"{pilot['canonical_sha256'][:16]} != "
              f"{EXPECTED_DATASET_SHA[:16]}", failures)

    # dataset manifest `sources` (key -> sha) must bind to the source artifacts
    lichess_src = {lichess["source_id"]: lichess["pgn_sha256"]}
    arena_src = {s["source_key"]: s["pgn_sha256"] for s in arena["sources"]}
    for key, sha in pilot["sources"].items():
        bound = arena_src.get(key) or lichess_src.get(key)
        run_check(f"source_binding_{key}", bound == sha,
                  f"manifest {sha[:16]} vs artifact {bound[:16] if bound else '?'}",
                  failures)

    engine_sha = sha256_file(paths["engine"])
    git = git_head()
    run_check("worktree_clean", worktree_clean(), "worktree not clean", failures)
    verify_rc = run_verify_dataset(paths["pilot_dir"])
    run_check("verify_dataset_allow_unlabeled", verify_rc == 0,
              f"rc={verify_rc}", failures)
    coverage = run_coverage(paths["engine"], paths["pilot_dir"],
                            paths["arena_dir"], paths["lichess_dir"],
                            pilot["canonical_sha256"])
    run_check("coverage_analyzer",
              coverage["rc"] == 0 and coverage["status"] == "DATA_PILOT_PASS",
              f"rc={coverage['rc']} status={coverage['status']}", failures)
    tools = tool_shas()
    env = env_versions()
    if paths.get("expected_engine_sha"):
        run_check("engine_sha", engine_sha == paths["expected_engine_sha"],
                  f"{engine_sha[:16]} != expected "
                  f"{paths['expected_engine_sha'][:16]}", failures)

    return {
        "status": "DATA_PILOT_PASS" if not failures else "DATA_PILOT_FAIL",
        "provenance": {
            "archive": archive,
            "lichess_selection": lichess,
            "arena_sources": arena,
            "dataset": {
                "pilot": pilot,
                "rebuild": rebuild,
                "rebuild_sha_identical": pilot["canonical_sha256"]
                == rebuild["canonical_sha256"],
                "frozen_sha": EXPECTED_DATASET_SHA,
            },
            "encoder": {
                "engine_sha256": engine_sha,
                "git_head": git,
                "cargo_version": cargo_version(),
                "verify_rc": verify_rc,
                "coverage": coverage,
            },
            "tools": tools,
            "environment": env,
            "worktree_clean": worktree_clean() if not failures else False,
        },
        "failures": failures,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", type=Path, required=True)
    ap.add_argument("--engine", type=Path, default=REPO / "target/release/eureka")
    ap.add_argument("--pilot-dataset", type=Path,
                    default=REPO / "data/s6/s6-eval-v1-multisource-pilot01")
    ap.add_argument("--rebuild-dataset", type=Path,
                    default=REPO / "data/s6/rebuild-check/"
                    "s6-eval-v1-multisource-pilot01")
    ap.add_argument("--arena-sources", type=Path,
                    default=REPO / "data/s6/sources")
    ap.add_argument("--lichess-dir", type=Path,
                    default=REPO / "data/s6/sources/lichess-standard-rated-v1")
    ap.add_argument("--expected-engine-sha", default=None,
                    help="optional engine SHA to compare (engine-replacement "
                         "detection)")
    ap.add_argument("--out", type=Path,
                    default=REPO / "results/s6/s6-n3a-data-pilot.json")
    args = ap.parse_args()

    paths = {
        "archive": args.archive.resolve(),
        "engine": args.engine.resolve(),
        "pilot_dir": args.pilot_dataset.resolve(),
        "rebuild_dir": args.rebuild_dataset.resolve(),
        "arena_dir": args.arena_sources.resolve(),
        "lichess_dir": args.lichess_dir.resolve(),
        "expected_engine_sha": args.expected_engine_sha,
    }
    result = build_provenance(paths)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md = args.out.with_suffix(".md")
    if md.exists():
        text = md.read_text(encoding="utf-8")
        section = (
            "\n## Provenance Closure (re-verified by verify_n3a_data_pilot.py)\n\n"
            f"STATUS: **{result['status']}**\n\n"
            "```text\n" +
            json.dumps(result.get("provenance", {}), indent=1) +
            "\n```\n")
        if "Provenance Closure" not in text:
            md.write_text(text.rstrip() + "\n" + section, encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result["status"] == "DATA_PILOT_PASS" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"PIPELINE_FAILURE: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
