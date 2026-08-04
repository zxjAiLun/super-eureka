#!/usr/bin/env python3
"""Prepare or launch the frozen S3-PROMOTION comparison.

The default action is fail-closed preparation.  The explicit ``--run`` flag
is required before cutechess can start.  This wrapper owns the frozen engine
and opening provenance, UCI identity probes, exact manager argv, and output
directory contract; the manager owns clocks, pairing, legality, and SPRT.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from typing import Any

import chess


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENGINE = REPO_ROOT / "target" / "release" / "chess-engine-demo.exe"
DEFAULT_MANAGER = REPO_ROOT / "tools" / ".cache" / "cutechess-1.5.1-win64" / "cutechess-cli.exe"
DEFAULT_SELECTION = REPO_ROOT / "tests" / "data" / "openings" / "s3-promotion-openings-v1.json"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "s3-promotion" / "run-001"

FROZEN_ENGINE_GIT_SHA = "91347775906f3f5d3730c9e9596037493429776d"
FROZEN_ENGINE_SHA256 = "b4bf0c3e73158bf3f5c072aa863aca671721275efc5b2e6d5354cf53a0fd0933"
MANAGER_VERSION = "1.5.1"
FROZEN_MANAGER_SHA256 = "8889f9582dc688c567704cf083f6025baf77f791cde903698c70b3420caf5d7e"
PROFILE_CANDIDATE = "current-final"
PROFILE_BASELINE = "current"
LABEL_CANDIDATE = "CurrentFinal"
LABEL_BASELINE = "Current"
PAIRS = 500
GAMES_MAX = 1000
TIME_CONTROL = "10+0.1"
HASH_MB = 16
THREADS = 1
CONCURRENCY = 1
SPRT = {"elo0": 20, "elo1": 60, "alpha": 0.05, "beta": 0.05}


class PromotionError(RuntimeError):
    """Raised for any fail-closed preflight or manager integrity failure."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PromotionError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise PromotionError(f"cannot read {path}: {exc}") from exc
    return digest.hexdigest()


def file_metadata(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    require(resolved.is_file(), f"required file does not exist: {resolved}")
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size": stat.st_size,
    }


def run_capture(argv: list[str], *, input_text: str | None = None, timeout: float = 30.0) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            argv,
            cwd=REPO_ROOT,
            input=input_text,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PromotionError(f"command failed: {argv!r}: {exc}") from exc


def git_output(*args: str) -> str:
    result = run_capture(["git", *args])
    require(result.returncode == 0, f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def assert_frozen_engine_source() -> str:
    current_sha = git_output("rev-parse", "HEAD")
    require(not git_output("status", "--porcelain"), "S3-PROMOTION requires a clean worktree")
    diff = run_capture(
        [
            "git",
            "diff",
            "--quiet",
            FROZEN_ENGINE_GIT_SHA,
            "--",
            "src",
            "Cargo.toml",
            "Cargo.lock",
        ]
    )
    require(
        diff.returncode == 0,
        f"engine source differs from frozen S3-FINAL commit {FROZEN_ENGINE_GIT_SHA}",
    )
    return current_sha


def probe_identity(engine: Path, profile: str) -> dict[str, Any]:
    argv = [str(engine), "--profile", profile]
    result = run_capture(argv, input_text="uci\nquit\n", timeout=15.0)
    lines = result.stdout.splitlines()
    reported = next(
        (
            line[len("info string search profile ") :].strip()
            for line in lines
            if line.startswith("info string search profile ")
        ),
        None,
    )
    identity = {
        "argv": argv,
        "id_name": next((line[8:].strip() for line in lines if line.startswith("id name ")), None),
        "id_author": next((line[10:].strip() for line in lines if line.startswith("id author ")), None),
        "reported_profile": reported,
        "uciok": "uciok" in lines,
        "return_code": result.returncode,
        "stdout_tail": result.stdout[-4096:],
        "stderr_tail": result.stderr[-4096:],
    }
    require(
        identity["return_code"] == 0
        and identity["uciok"]
        and identity["id_name"]
        and identity["id_author"]
        and identity["reported_profile"] == profile,
        f"profile identity mismatch for {profile}: {identity}",
    )
    return identity


def manager_metadata(manager: Path) -> dict[str, Any]:
    result = run_capture([str(manager), "-version"])
    require(result.returncode == 0, f"manager version probe failed: {result.stderr.strip()}")
    version = result.stdout.strip()
    require(f"cutechess-cli {MANAGER_VERSION}" in version, f"unsupported manager: {version}")
    metadata = {**file_metadata(manager), "version": version}
    require(
        metadata["sha256"] == FROZEN_MANAGER_SHA256,
        "manager SHA-256 is not the frozen cutechess-cli binary",
    )
    return metadata


def position_key(fen: str) -> str:
    fields = fen.split()
    require(len(fields) >= 4, f"invalid FEN/EPD line: {fen!r}")
    return " ".join(fields[:4])


def load_selection(selection_path: Path) -> tuple[dict[str, Any], Path, list[str]]:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    require(selection.get("suite_id") == "s3-promotion-openings-v1", "unexpected selection suite")
    spec = selection.get("selection")
    require(isinstance(spec, dict), "selection metadata is missing")
    source = (REPO_ROOT / spec["source_path"]).resolve()
    source_meta = file_metadata(source)
    require(source_meta["sha256"] == spec["source_sha256"], "opening source SHA-256 mismatch")
    start = int(spec["source_line_start"])
    end = int(spec["source_line_end"])
    lines = source.read_text(encoding="utf-8").splitlines()
    require(start >= 1000 and end >= start, "promotion selection must stay in the disjoint source range")
    selected = lines[start - 1 : end]
    require(len(selected) == int(spec["count"]) == PAIRS, "promotion selection count mismatch")
    serialized = "\n".join(selected) + "\n"
    require(
        hashlib.sha256(serialized.encode("utf-8")).hexdigest() == spec["slice_sha256"],
        "promotion opening slice SHA-256 mismatch",
    )
    keys: set[str] = set()
    for line_number, line in enumerate(selected, start=start):
        fields = line.split(";", 1)[0].split()
        fen = " ".join(fields[:6] if len(fields) >= 6 else fields[:4] + ["0", "1"])
        board = chess.Board(fen)
        require(any(board.legal_moves), f"promotion source line {line_number} is terminal")
        key = position_key(fen)
        require(key not in keys, f"duplicate promotion opening at source line {line_number}")
        keys.add(key)
    return selection, source, selected


def write_runtime_openings(output_dir: Path, selected: list[str]) -> tuple[Path, str]:
    runtime = output_dir / "openings.epd"
    with runtime.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(selected) + "\n")
    return runtime, sha256_file(runtime)


def command_text(command: list[str]) -> str:
    return subprocess.list2cmdline(command) if os.name == "nt" else shlex.join(command)


def build_command(manager: Path, engine: Path, openings: Path, output_dir: Path) -> list[str]:
    return [
        str(manager.resolve()),
        "-engine",
        f"name={LABEL_CANDIDATE}",
        f"cmd={engine.resolve()}",
        "proto=uci",
        "arg=--profile",
        f"arg={PROFILE_CANDIDATE}",
        "-engine",
        f"name={LABEL_BASELINE}",
        f"cmd={engine.resolve()}",
        "proto=uci",
        "arg=--profile",
        f"arg={PROFILE_BASELINE}",
        "-variant",
        "standard",
        "-openings",
        f"file={openings.resolve()}",
        "format=epd",
        "order=sequential",
        "policy=default",
        "-each",
        f"tc={TIME_CONTROL}",
        f"option.Hash={HASH_MB}",
        "-rounds",
        str(GAMES_MAX),
        "-repeat",
        "2",
        "-concurrency",
        str(CONCURRENCY),
        "-pgnout",
        str((output_dir / "match.pgn").resolve()),
        "-resultformat",
        "short",
        "-sprt",
        f"elo0={SPRT['elo0']}",
        f"elo1={SPRT['elo1']}",
        f"alpha={SPRT['alpha']}",
        f"beta={SPRT['beta']}",
    ]


def refuse_reused_output(output_dir: Path) -> None:
    require(not output_dir.exists() or not any(output_dir.iterdir()), f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare(
    args: argparse.Namespace,
    output_dir_override: Path | None = None,
) -> tuple[dict[str, Any], list[str], Path, Path, Path]:
    output_dir = (output_dir_override or args.output_dir).expanduser().resolve()
    refuse_reused_output(output_dir)
    current_git_sha = assert_frozen_engine_source()
    engine = args.engine.expanduser().resolve()
    engine_meta = file_metadata(engine)
    require(engine_meta["sha256"] == FROZEN_ENGINE_SHA256, "engine SHA-256 is not the frozen S3-FINAL binary")
    selection_path = args.selection.expanduser().resolve()
    selection, source, selected = load_selection(selection_path)
    manager = args.manager.expanduser().resolve()
    manager_meta = manager_metadata(manager)
    identities = {
        "candidate": probe_identity(engine, PROFILE_CANDIDATE),
        "baseline": probe_identity(engine, PROFILE_BASELINE),
    }
    runtime_openings, runtime_hash = write_runtime_openings(output_dir, selected)
    command = build_command(manager, engine, runtime_openings, output_dir)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "milestone": "S3-PROMOTION",
        "status": "PREPARED",
        "decision": "NOT_STARTED",
        "prepared_utc": utc_now(),
        "current_git_sha": current_git_sha,
        "frozen_engine_git_sha": FROZEN_ENGINE_GIT_SHA,
        "engine": {
            "binary": engine_meta,
            "candidate": {"label": LABEL_CANDIDATE, "profile": PROFILE_CANDIDATE, "identity": identities["candidate"]},
            "baseline": {"label": LABEL_BASELINE, "profile": PROFILE_BASELINE, "identity": identities["baseline"]},
        },
        "manager": manager_meta,
        "opening": {
            "selection_manifest": str(selection_path),
            "selection_manifest_sha256": sha256_file(selection_path),
            "source": str(source),
            "runtime_path": str(runtime_openings),
            "runtime_sha256": runtime_hash,
            "pairs": PAIRS,
            "strict_color_swap": True,
            "order": "sequential",
        },
        "match": {
            "protocol": "uci",
            "time_control": TIME_CONTROL,
            "hash_mb": HASH_MB,
            "threads": THREADS,
            "concurrency": CONCURRENCY,
            "games_max": GAMES_MAX,
            "candidate_first": True,
        },
        "sprt": {"subject": "CurrentFinal minus Current", **SPRT},
        "command": command,
        "command_text": command_text(command),
        "artifacts": {
            "manifest": "manifest.json",
            "command": "command.txt",
            "openings": "openings.epd",
            "stdout": "manager.stdout.log",
            "stderr": "manager.stderr.log",
            "pgn": "match.pgn",
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    (output_dir / "command.txt").write_text(command_text(command) + "\n", encoding="utf-8")
    return manifest, command, output_dir, engine, manager


def run_manager(manifest: dict[str, Any], command: list[str], output_dir: Path, engine: Path, manager: Path) -> dict[str, Any]:
    stdout_path = output_dir / "manager.stdout.log"
    stderr_path = output_dir / "manager.stderr.log"
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        try:
            process = subprocess.Popen(command, cwd=REPO_ROOT, stdout=stdout, stderr=stderr, text=True)
            return_code = process.wait()
        except OSError as exc:
            raise PromotionError(f"manager failed to start: {exc}") from exc
    require(return_code == 0, f"manager exited with code {return_code}")
    require((output_dir / "match.pgn").is_file(), "manager completed without producing match.pgn")
    require(sha256_file(engine) == FROZEN_ENGINE_SHA256, "engine changed during the match")
    require(sha256_file(manager) == manifest["manager"]["sha256"], "manager changed during the match")
    require(git_output("rev-parse", "HEAD") == manifest["current_git_sha"], "Git tip changed during the match")
    manifest["status"] = "MANAGER_COMPLETED"
    manifest["manager_return_code"] = return_code
    manifest["finished_utc"] = utc_now()
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    mode = command.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    command.add_argument("--engine", type=Path, default=DEFAULT_ENGINE)
    command.add_argument("--manager", type=Path, default=DEFAULT_MANAGER)
    command.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    command.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return command


def main() -> int:
    args = parser().parse_args()
    try:
        if args.dry_run:
            with tempfile.TemporaryDirectory(prefix="s3-promotion-preflight-") as temporary:
                manifest, _, _, _, _ = prepare(args, Path(temporary))
                manifest["status"] = "DRY_RUN"
                manifest["requested_output_dir"] = str(args.output_dir.expanduser().resolve())
                print(json.dumps(manifest, ensure_ascii=False, indent=2))
            return 0
        manifest, command, output_dir, engine, manager = prepare(args)
        result = run_manager(manifest, command, output_dir, engine, manager)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, PromotionError, ValueError, json.JSONDecodeError) as exc:
        print(f"INTEGRITY_FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
