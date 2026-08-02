#!/usr/bin/env python3
"""Launch the D1.14 Current-vs-CurrentLmr Cute Chess match.

This is a thin, fail-closed manager wrapper. Cutechess-cli owns clocks,
legal-move handling, pairing, and built-in SPRT; this script owns provenance,
profile identity, deterministic opening selection, exact command capture, and
artifact handling. It never starts a game if the binary, opening suite,
profile identity, or approved engine-code baseline is inconsistent.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
from typing import Any, Optional

from prepare_d114_openings import DEFAULT_METADATA, DEFAULT_OUTPUT, verify as verify_openings


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENGINE = REPO_ROOT / "target" / "release" / "chess-engine-demo.exe"
DEFAULT_CUTECHESS = REPO_ROOT / "tools" / ".cache" / "cutechess-1.5.1-win64" / "cutechess-cli.exe"
APPROVED_ENGINE_CODE_BASELINE = "dcb009813239549f4805e9a2467aadae4ccebc11"
MANAGER_VERSION = "1.5.1"
BASELINE_PROFILE = "current"
CANDIDATE_PROFILE = "current-lmr"
BASELINE_LABEL = "Current"
CANDIDATE_LABEL = "CurrentLmr"
FULL_OPENING_COUNT = 4_852
SMOKE_OPENING_COUNT = 20
FORMAL_GAMES = FULL_OPENING_COUNT * 2
SMOKE_GAMES = SMOKE_OPENING_COUNT * 2


class D114Error(RuntimeError):
    """Raised for a preflight or manager integrity failure."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise D114Error(f"cannot read {path}: {exc}") from exc
    return digest.hexdigest()


def file_metadata(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise D114Error(f"file does not exist: {resolved}")
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size": stat.st_size,
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    }


def run_capture(argv: list[str], *, timeout: float = 30.0) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise D114Error(f"command failed to run: {argv!r}: {exc}") from exc


def git_output(*args: str) -> str:
    result = run_capture(["git", *args])
    if result.returncode != 0:
        raise D114Error(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def cpu_metadata() -> dict[str, Any]:
    physical: Optional[int] = None
    logical = os.cpu_count()
    if platform.system() == "Windows":
        probe = run_capture(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "@{physical=((Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores -Sum).Sum); logical=((Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfLogicalProcessors -Sum).Sum)} | ConvertTo-Json -Compress",
            ],
            timeout=15,
        )
        if probe.returncode == 0:
            try:
                value = json.loads(probe.stdout)
                physical = int(value["physical"])
                logical = int(value["logical"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                physical = None
    return {
        "model": platform.processor() or platform.uname().machine,
        "physical_cores": physical,
        "logical_processors": logical,
    }


def probe_engine_identity(argv: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(
            argv,
            input="uci\nquit\n",
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise D114Error(f"engine identity probe failed for {argv!r}: {exc}") from exc
    lines = result.stdout.splitlines()
    name = next((line[8:].strip() for line in lines if line.startswith("id name ")), None)
    author = next((line[10:].strip() for line in lines if line.startswith("id author ")), None)
    profile = next(
        (
            line[len("info string search profile ") :].strip()
            for line in lines
            if line.startswith("info string search profile ")
        ),
        None,
    )
    return {
        "argv": argv,
        "id_name": name,
        "id_author": author,
        "reported_profile": profile,
        "uciok": "uciok" in lines,
        "return_code": result.returncode,
        "stdout_tail": result.stdout[-4096:],
        "stderr_tail": result.stderr[-4096:],
    }


def require_engine_identity(argv: list[str], expected_profile: str) -> dict[str, Any]:
    identity = probe_engine_identity(argv)
    if (
        identity["return_code"] != 0
        or not identity["uciok"]
        or not identity["id_name"]
        or not identity["id_author"]
        or identity["reported_profile"] != expected_profile
    ):
        raise D114Error(
            f"profile integrity failure for {expected_profile}: "
            f"reported={identity['reported_profile']!r}, identity={identity}"
        )
    return identity


def manager_metadata(path: Path) -> dict[str, Any]:
    result = run_capture([str(path), "-version"])
    if result.returncode != 0:
        raise D114Error(f"cutechess version probe failed: {result.stderr.strip()}")
    version_text = result.stdout.strip()
    if f"cutechess-cli {MANAGER_VERSION}" not in version_text:
        raise D114Error(f"unsupported cutechess version; expected {MANAGER_VERSION}: {version_text}")
    return {
        **file_metadata(path),
        "version": version_text,
        "source": "https://github.com/cutechess/cutechess/releases/tag/v1.5.1",
    }


def load_opening_contract(metadata_path: Path, output_path: Path) -> dict[str, Any]:
    result = verify_openings(
        argparse.Namespace(
            verify=True,
            source=None,
            output=output_path,
            metadata=metadata_path,
        )
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    lines = [line.strip() for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != FULL_OPENING_COUNT:
        raise D114Error(f"D1.14 opening suite must contain {FULL_OPENING_COUNT} positions")
    return {
        "verification": result,
        "metadata": metadata,
        "path": str(output_path.resolve()),
        "sha256": sha256_file(output_path),
        "count": len(lines),
    }


def write_smoke_openings(full_path: Path, output_dir: Path) -> tuple[Path, int, str]:
    lines = [line for line in full_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < SMOKE_OPENING_COUNT:
        raise D114Error("full D1.14 opening suite is shorter than the smoke selection")
    smoke_path = output_dir / "openings-smoke.epd"
    smoke_path.write_text("\n".join(lines[:SMOKE_OPENING_COUNT]) + "\n", encoding="utf-8")
    return smoke_path, SMOKE_OPENING_COUNT, sha256_file(smoke_path)


def build_command(
    manager: Path,
    engine: Path,
    opening_path: Path,
    output_dir: Path,
    mode: str,
    concurrency: int,
) -> list[str]:
    if mode == "Smoke":
        time_control = "5+0.05"
        games = SMOKE_GAMES
        sprt: list[str] = []
    elif mode == "Sprt":
        time_control = "10+0.1"
        games = FORMAL_GAMES
        sprt = ["-sprt", "elo0=0", "elo1=5", "alpha=0.05", "beta=0.05"]
    else:
        raise D114Error(f"unknown D1.14 mode: {mode}")
    engine_path = str(engine.resolve())
    return [
        str(manager.resolve()),
        "-engine",
        f"name={CANDIDATE_LABEL}",
        f"cmd={engine_path}",
        "proto=uci",
        "arg=--profile",
        f"arg={CANDIDATE_PROFILE}",
        "-engine",
        f"name={BASELINE_LABEL}",
        f"cmd={engine_path}",
        "proto=uci",
        "arg=--profile",
        f"arg={BASELINE_PROFILE}",
        "-variant",
        "standard",
        "-openings",
        f"file={opening_path.resolve()}",
        "format=epd",
        "order=sequential",
        "policy=encounter",
        "-each",
        f"tc={time_control}",
        "option.Hash=16",
        "-rounds",
        str(games),
        "-repeat",
        "2",
        "-concurrency",
        str(concurrency),
        "-pgnout",
        str((output_dir / "match.pgn").resolve()),
        "-event",
        "D1.14 Current vs CurrentLmr",
        "-site",
        "chessenginedemo",
        "-resultformat",
        "short",
        *sprt,
    ]


def command_text(command: list[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(command)
    return shlex.join(command)


def refuse_reused_output(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise D114Error(f"output directory is not empty; choose a new D1.14 run directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_opening_hash(path: Path, opening: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(
            {
                "suite": opening["metadata"],
                "runtime_path": opening["path"],
                "runtime_sha256": opening["sha256"],
                "runtime_count": opening["count"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def write_binary_hash(path: Path, engine: dict[str, Any], identities: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(
            {
                "binary": engine,
                "profiles": {
                    "baseline": identities["baseline"]["reported_profile"],
                    "candidate": identities["candidate"]["reported_profile"],
                },
                "engine_code_baseline_sha": APPROVED_ENGINE_CODE_BASELINE,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def gzip_file(source: Path, destination: Path) -> None:
    with source.open("rb") as input_file, gzip.open(destination, "wb", compresslevel=9) as output_file:
        shutil.copyfileobj(input_file, output_file)


def _drain(stream, destination: Path, echo: bool) -> None:
    with destination.open("w", encoding="utf-8", errors="replace") as output:
        for line in iter(stream.readline, ""):
            output.write(line)
            output.flush()
            if echo:
                print(line, end="", flush=True)


def run_manager(command: list[str], output_dir: Path) -> int:
    stdout_path = output_dir / "sprt-output.txt"
    stderr_path = output_dir / "manager.stderr.txt"
    try:
        process = subprocess.Popen(
            command,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
    except OSError as exc:
        raise D114Error(f"cutechess failed to start: {exc}") from exc
    assert process.stdout is not None and process.stderr is not None
    stdout_thread = threading.Thread(target=_drain, args=(process.stdout, stdout_path, True), daemon=True)
    stderr_thread = threading.Thread(target=_drain, args=(process.stderr, stderr_path, True), daemon=True)
    stdout_thread.start()
    stderr_thread.start()
    return_code = process.wait()
    stdout_thread.join(timeout=10)
    stderr_thread.join(timeout=10)
    if not (output_dir / "match.pgn").is_file():
        raise D114Error("cutechess exited without producing match.pgn")
    gzip_file(output_dir / "match.pgn", output_dir / "match.pgn.gz")
    with gzip.open(output_dir / "manager.log.gz", "wt", encoding="utf-8") as log:
        log.write("=== stdout ===\n")
        log.write(stdout_path.read_text(encoding="utf-8", errors="replace"))
        log.write("\n=== stderr ===\n")
        log.write(stderr_path.read_text(encoding="utf-8", errors="replace"))
    (output_dir / "manager.stderr.txt").unlink(missing_ok=True)
    (output_dir / "match.pgn").unlink(missing_ok=True)
    if return_code != 0:
        raise D114Error(f"cutechess exited with code {return_code}")
    return return_code


def build_manifest(
    args: argparse.Namespace,
    output_dir: Path,
    opening: dict[str, Any],
    runtime_opening: Path,
    runtime_opening_count: int,
    runtime_opening_sha256: str,
    manager: Optional[dict[str, Any]],
    engine: dict[str, Any],
    identities: dict[str, Any],
    command: list[str],
    git_sha: str,
) -> dict[str, Any]:
    mode = args.mode
    concurrency = 1 if mode == "Smoke" else 2
    games = SMOKE_GAMES if mode == "Smoke" else FORMAL_GAMES
    return {
        "schema_version": 1,
        "milestone": "D1.14",
        "mode": mode,
        "status": "PREPARED",
        "decision": "NOT_STARTED",
        "started_utc": utc_now(),
        "git_sha": git_sha,
        "engine_code_baseline_sha": args.engine_code_baseline_sha,
        "rust_version": run_capture(["rustc", "--version"]).stdout.strip(),
        "cargo_lock_sha256": sha256_file(REPO_ROOT / "Cargo.lock"),
        "build_command": "cargo build --release",
        "cpu": cpu_metadata(),
        "engine": {
            "binary": engine,
            "baseline": {
                "label": BASELINE_LABEL,
                "profile": BASELINE_PROFILE,
                "argv": [str(Path(engine["path"]).resolve()), "--profile", BASELINE_PROFILE],
                "identity": identities["baseline"],
            },
            "candidate": {
                "label": CANDIDATE_LABEL,
                "profile": CANDIDATE_PROFILE,
                "argv": [str(Path(engine["path"]).resolve()), "--profile", CANDIDATE_PROFILE],
                "identity": identities["candidate"],
            },
        },
        "manager": manager,
        "opening": {
            "suite_metadata": opening["metadata"],
            "runtime_path": str(runtime_opening.resolve()),
            "runtime_sha256": runtime_opening_sha256,
            "runtime_count": runtime_opening_count,
            "order": "sequential",
            "repeat": 2,
            "color_swap": True,
        },
        "match": {
            "protocol": "uci",
            "variant": "standard",
            "time_control": "5+0.05" if mode == "Smoke" else "10+0.1",
            "hash_mb": 16,
            "ponder": False,
            "threads": 1,
            "concurrency": concurrency,
            "games_max": games,
            "pairs_max": runtime_opening_count,
            "adjudication": {
                "draw": False,
                "resign": False,
                "tablebase": False,
            },
        },
        "sprt": {
            "subject": "current-lmr minus current",
            "player_1": "current-lmr",
            "player_2": "current",
            "elo0": 0,
            "elo1": 5,
            "alpha": 0.05,
            "beta": 0.05,
            "manager_builtin": mode == "Sprt",
        },
        "command": command,
        "command_text": command_text(command),
        "artifacts": {
            "command": "command.txt",
            "manifest": "manifest.json",
            "openings": "openings.sha256",
            "binary": "binary.sha256",
            "sprt_output": "sprt-output.txt",
            "pgn": "match.pgn.gz",
            "manager_log": "manager.log.gz",
            "summary": "summary.json",
        },
    }


def check_engine_code_baseline(expected: str) -> None:
    if expected != APPROVED_ENGINE_CODE_BASELINE:
        raise D114Error(
            f"D1.14 only permits the approved engine-code baseline {APPROVED_ENGINE_CODE_BASELINE}"
        )
    result = run_capture(
        ["git", "diff", "--quiet", expected, "--", "src/chess", "src/engine"]
    )
    if result.returncode != 0:
        raise D114Error("engine source differs from the approved D1.13 code baseline")


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--mode", choices=("Smoke", "Sprt"), required=True)
    command.add_argument("--engine", type=Path, default=DEFAULT_ENGINE)
    command.add_argument("--cutechess", type=Path, default=DEFAULT_CUTECHESS)
    command.add_argument("--opening-file", type=Path, default=DEFAULT_OUTPUT)
    command.add_argument("--opening-metadata", type=Path, default=DEFAULT_METADATA)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--engine-code-baseline", dest="engine_code_baseline_sha", default=APPROVED_ENGINE_CODE_BASELINE)
    command.add_argument("--dry-run", action="store_true")
    return command


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.expanduser().resolve()
    refuse_reused_output(output_dir)
    git_sha = git_output("rev-parse", "HEAD")
    if git_output("status", "--porcelain"):
        raise D114Error("D1.14 requires a clean worktree before the match")
    check_engine_code_baseline(args.engine_code_baseline_sha)
    opening_file = args.opening_file.expanduser().resolve()
    opening_metadata = args.opening_metadata.expanduser().resolve()
    opening = load_opening_contract(opening_metadata, opening_file)
    if args.mode == "Smoke":
        runtime_opening, runtime_count, runtime_hash = write_smoke_openings(opening_file, output_dir)
    else:
        runtime_opening, runtime_count, runtime_hash = opening_file, FULL_OPENING_COUNT, opening["sha256"]
    engine_path = args.engine.expanduser().resolve()
    engine = file_metadata(engine_path)
    baseline_argv = [str(engine_path), "--profile", BASELINE_PROFILE]
    candidate_argv = [str(engine_path), "--profile", CANDIDATE_PROFILE]
    identities = {
        "baseline": require_engine_identity(baseline_argv, BASELINE_PROFILE),
        "candidate": require_engine_identity(candidate_argv, CANDIDATE_PROFILE),
    }
    manager_path = args.cutechess.expanduser().resolve()
    manager = None if args.dry_run else manager_metadata(manager_path)
    if args.dry_run and manager_path.is_file():
        manager = manager_metadata(manager_path)
    command = build_command(manager_path, engine_path, runtime_opening, output_dir, args.mode, 1 if args.mode == "Smoke" else 2)
    manifest = build_manifest(
        args,
        output_dir,
        opening,
        runtime_opening,
        runtime_count,
        runtime_hash,
        manager,
        engine,
        identities,
        command,
        git_sha,
    )
    (output_dir / "command.txt").write_text(command_text(command) + "\n", encoding="utf-8")
    write_opening_hash(output_dir / "openings.sha256", opening)
    write_binary_hash(output_dir / "binary.sha256", engine, identities)
    write_json(output_dir / "manifest.json", manifest)
    if args.dry_run:
        manifest["status"] = "DRY_RUN"
        manifest["decision"] = "NOT_STARTED"
        write_json(output_dir / "manifest.json", manifest)
        return manifest
    try:
        return_code = run_manager(command, output_dir)
    except D114Error as exc:
        manifest["status"] = "INTEGRITY_FAIL"
        manifest["decision"] = "INCONCLUSIVE"
        manifest["failure"] = str(exc)
        write_json(output_dir / "manifest.json", manifest)
        raise
    if git_output("rev-parse", "HEAD") != git_sha:
        manifest["status"] = "INTEGRITY_FAIL"
        manifest["decision"] = "INCONCLUSIVE"
        manifest["failure"] = "Git tip changed during the match"
        write_json(output_dir / "manifest.json", manifest)
        raise D114Error(manifest["failure"])
    manifest["status"] = "MANAGER_COMPLETED"
    manifest["decision"] = "PENDING_VERIFICATION"
    manifest["manager_return_code"] = return_code
    manifest["finished_utc"] = utc_now()
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> int:
    args = parser().parse_args()
    try:
        result = run(args)
    except D114Error as exc:
        print(f"INTEGRITY_FAIL: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
