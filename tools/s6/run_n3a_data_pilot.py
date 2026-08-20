#!/usr/bin/env python3
"""S6-N3A data-pilot runner (committed, fail-closed).

Executes the full frozen pipeline in order:
  1. verify the monthly archive SHA-256 against the official checksum;
  2. lichess_select (2026-07, 2000 games, seed 20260812);
  3. pilot build A (s6-eval-v1-multisource-pilot01, sampling v2);
  4. verify_dataset --allow-unlabeled on A;
  5. pilot build B (independent out dir) and compare dataset SHA-256;
  6. analyze_nnue_coverage with the hard gates (rebuilt SHA and verify rc
     are computed INTERNALLY - never accepted from the command line);
  7. generate results/s6/s6-n3a-data-pilot.{json,md}.

Any subprocess failure or DATA_PILOT_FAIL returns nonzero. There is NO
auto-fallback: the month, seed, games-per-month, dataset id, and gate
thresholds are frozen; the script never retries or re-tunes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
VENV_PYTHON = REPO.parent / ".venv-super-eureka-s6-n1" / "bin" / "python"

ARCHIVE_SHA = "68738b1c448f051dc8d42db645d5b01749988a3bc1c24981adfe44ea92060dc7"
MONTH = "2026-07"
SEED = 20260812
GAMES_PER_MONTH = 2000
DATASET_ID = "s6-eval-v1-multisource-pilot01"
SOURCES = [str(REPO / "data/s6/sources"),
           str(REPO / "data/s6/sources/lichess-standard-rated-v1")]


def run(cmd: list[str]) -> int:
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    proc = subprocess.run(cmd)
    return proc.returncode


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", type=Path, required=True,
                    help="official 2026-07 .pgn.zst archive (SHA-verified)")
    ap.add_argument("--engine", type=Path, default=REPO / "target/release/eureka")
    ap.add_argument("--out", type=Path, default=REPO / "data/s6")
    args = ap.parse_args()
    python = VENV_PYTHON if VENV_PYTHON.exists() else sys.executable

    # 1. archive SHA
    actual = sha256_file(args.archive)
    if actual != ARCHIVE_SHA:
        print(f"FAIL: archive SHA {actual[:16]} != official {ARCHIVE_SHA[:16]}")
        return 3
    print(f"archive SHA verified: {actual}", flush=True)

    lichess_out = REPO / "data/s6/sources/lichess-standard-rated-v1"
    if lichess_out.exists():
        print(f"FAIL: lichess selection dir already exists {lichess_out}")
        return 4

    # 2. lichess_select
    rc = run([str(python), str(REPO / "tools/s6/lichess_select.py"),
              "--months", MONTH, "--games-per-month", str(GAMES_PER_MONTH),
              "--seed", str(SEED), "--out", str(lichess_out),
              "--local", str(args.archive)])
    if rc != 0:
        print(f"FAIL: lichess_select rc={rc}")
        return rc

    out_a = args.out / DATASET_ID
    if out_a.exists():
        print(f"FAIL: pilot dataset already exists {out_a}")
        return 4
    build_cmd = [str(python), str(REPO / "tools/s6/build_dataset.py"),
                 "--sources", *SOURCES, "--dataset-id", DATASET_ID,
                 "--sampling-version", "2", "--out", str(args.out)]
    # 3. build A
    rc = run(build_cmd)
    if rc != 0:
        print(f"FAIL: build A rc={rc}")
        return rc
    sha1 = json.loads((out_a / "dataset_manifest.json").read_text())["dataset_sha256"]

    # 4. verify --allow-unlabeled
    rc = run([str(python), str(REPO / "tools/s6/verify_dataset.py"),
              "--dataset", str(out_a), "--allow-unlabeled"])
    if rc != 0:
        print(f"FAIL: verify_dataset rc={rc}")
        return rc

    # 5. build B (independent out dir) + SHA comparison
    rebuild_out = args.out / "rebuild-check"
    if rebuild_out.exists():
        import shutil
        shutil.rmtree(rebuild_out)
    rc = run(build_cmd + ["--out", str(rebuild_out)])
    if rc != 0:
        print(f"FAIL: build B rc={rc}")
        return rc
    sha2 = json.loads((rebuild_out / DATASET_ID / "dataset_manifest.json")
                      .read_text())["dataset_sha256"]
    if sha1 != sha2:
        print(f"FAIL: rebuild SHA mismatch {sha1[:16]} != {sha2[:16]}")
        return 5
    print(f"rebuild SHA identical: {sha1}", flush=True)

    # 6. coverage + hard gates (rc computed internally)
    results = REPO / "results/s6/s6-n3a-data-pilot.json"
    rc = run([str(python), str(REPO / "tools/s6/analyze_nnue_coverage.py"),
              "--engine", str(args.engine), "--dataset", str(out_a),
              "--sources", *SOURCES, "--gate",
              "--rebuilt-sha", sha1, "--verify-rc", "0",
              "--out", str(results)])
    if rc != 0:
        print(f"FAIL: analyze/coverage rc={rc}")
        return rc

    # 7. render markdown from the JSON record
    data = json.loads(results.read_text())
    status = data["status"]
    a = data["analysis"]
    g = data["gate"]
    lines = [
        "# S6-N3A — Independent-Source Data Pilot",
        "",
        f"STATUS: **{status}**",
        "",
        "## Dataset",
        "",
        "```text",
        f"dataset_id:   {a['dataset_id']}",
        f"records:      {a['records_total']}",
        f"dataset SHA:  {a['dataset_sha256']}",
        f"rebuild SHA:  {g['facts']['rebuilt_sha256']}",
        f"verify --allow-unlabeled rc: {g['facts']['verify_rc']}",
        "```",
        "",
        "## Composition",
        "",
        "| family | records | share |",
        "|---|---:|---:|",
    ]
    for fam, c in sorted(a["per_family"].items()):
        lines.append(f"| {fam} | {c} | {c / a['records_total']:.1%} |")
    lines += ["", "| phase bucket | records |", "|---|---:|"]
    for p, c in sorted(a["per_phase"].items()):
        lines.append(f"| {p} | {c} |")
    lines += ["", "| split | records |", "|---|---:|"]
    for s, c in sorted(a["per_split"].items()):
        lines.append(f"| {s} | {c} |")
    lines += [
        "",
        "## Feature coverage (Rust nnue-features-batch)",
        "",
        "| split | positions | union unique | union/40960 | unseen act. | unseen rate | pos w/ unseen |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in ("train", "validation", "holdout"):
        c = a["coverage"][s]
        lines.append(
            f"| {s} | {c['positions']} | {c.get('union_unique', '-')} | "
            f"{c.get('union_fraction', '-')} | {c.get('unseen_activations', '-')} | "
            f"{c.get('unseen_rate', '-')} | {c.get('positions_with_unseen', '-')} |")
    lines += ["", "## Pilot hard gate", "", "| check | pass |", "|---|---|"]
    for k, v in g["checks"].items():
        lines.append(f"| {k} | {'PASS' if v else 'FAIL'} |")
    lines += ["", f"**{status}**", ""]
    md_path = REPO / "results/s6/s6-n3a-data-pilot.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {md_path}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
