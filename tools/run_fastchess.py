#!/usr/bin/env python3
"""Prepare and run formal engine A/B matches through Fastchess.

This wrapper owns reproducibility metadata, pinned opening-book acquisition,
and command construction. Fastchess owns UCI processes, clocks, scheduling,
legality, recovery, pair statistics, and SPRT. The wrapper never starts an
engine directly and never reimplements game or statistical logic.
"""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Optional
from urllib.request import urlopen
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE_CONFIG = Path(__file__).with_name("fastchess_profiles.json")
DEFAULT_BOOK_MANIFEST = REPO_ROOT / "books" / "manifest.json"
DEFAULT_BOOK_CACHE = REPO_ROOT / "books" / "cache"
STARTUP_PROFILES = frozenset(
    {
        "current",
        "current-aspiration",
        "current-aspiration-lmr",
        "current-aspiration-lmr-futility",
        "current-aspiration-lmr-futility-see",
    }
)


class FastchessError(RuntimeError):
    """A configuration, integrity, or Fastchess execution failure."""


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FastchessError(f"cannot read JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FastchessError(f"JSON root must be an object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha384_sri(path: Path) -> str:
    digest = hashlib.sha384()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return base64.b64encode(digest.digest()).decode("ascii")


def file_metadata(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FastchessError(f"file does not exist: {resolved}")
    return {
        "resolved_path": str(resolved),
        "file_size": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def locate_fastchess(requested: Optional[Path]) -> Optional[Path]:
    if requested is not None:
        return requested.expanduser().resolve()
    discovered = shutil.which("fastchess") or shutil.which("fastchess.exe")
    return Path(discovered).resolve() if discovered else None


def fastchess_metadata(path: Path) -> dict[str, Any]:
    metadata = file_metadata(path)
    try:
        probe = subprocess.run(
            [str(path), "-version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise FastchessError(f"cannot probe Fastchess {path}: {exc}") from exc
    if probe.returncode != 0:
        raise FastchessError(
            f"Fastchess version probe failed with exit code {probe.returncode}"
        )
    metadata.update(
        {
            "version_stdout": probe.stdout.strip(),
            "version_stderr": probe.stderr.strip(),
            "version_return_code": probe.returncode,
        }
    )
    return metadata


def load_profile(config_path: Path, profile_name: str) -> dict[str, Any]:
    config = read_json(config_path)
    profiles = config.get("profiles")
    if not isinstance(profiles, dict) or profile_name not in profiles:
        raise FastchessError(f"unknown Fastchess profile: {profile_name}")
    profile = profiles[profile_name]
    if not isinstance(profile, dict):
        raise FastchessError(f"Fastchess profile is not an object: {profile_name}")
    for side in ("baseline", "candidate"):
        entry = profile.get(side)
        if not isinstance(entry, dict) or entry.get("search_profile") not in STARTUP_PROFILES:
            raise FastchessError(f"invalid {side} search profile in {profile_name}")
    return profile


def load_book_entry(manifest_path: Path, book_id: str) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    books = manifest.get("books")
    if not isinstance(books, dict) or book_id not in books:
        raise FastchessError(f"unknown opening book: {book_id}")
    entry = books[book_id]
    if not isinstance(entry, dict):
        raise FastchessError(f"opening book entry is not an object: {book_id}")
    for key in ("format", "content_filename", "archive_url", "content_sha384_base64"):
        if not entry.get(key):
            raise FastchessError(f"opening book {book_id} lacks {key}")
    if entry["format"] not in {"epd", "pgn"}:
        raise FastchessError(f"unsupported opening format: {entry['format']}")
    return entry


def _safe_zip_member(member: zipfile.ZipInfo, content_filename: str) -> bool:
    if member.is_dir() or Path(member.filename).name != content_filename:
        return False
    return not any(part in {"", ".", ".."} for part in Path(member.filename).parts)


def _download_book(entry: dict[str, Any], destination: Path) -> None:
    archive_name = entry.get("archive_filename") or Path(entry["archive_url"]).name
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fastchess-book-") as temp_dir:
        archive_path = Path(temp_dir) / archive_name
        try:
            with urlopen(entry["archive_url"], timeout=60) as response, archive_path.open("wb") as target:
                shutil.copyfileobj(response, target)
            with zipfile.ZipFile(archive_path) as archive:
                matches = [
                    member
                    for member in archive.infolist()
                    if _safe_zip_member(member, entry["content_filename"])
                ]
                if len(matches) != 1:
                    raise FastchessError(
                        f"book archive must contain one {entry['content_filename']} member"
                    )
                extracted = archive.read(matches[0])
            temporary = destination.with_suffix(destination.suffix + ".part")
            temporary.write_bytes(extracted)
            temporary.replace(destination)
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            raise FastchessError(f"cannot download opening book: {exc}") from exc


def ensure_book(
    manifest_path: Path,
    book_id: str,
    local_path: Optional[Path],
    cache_dir: Path,
    download: bool,
) -> dict[str, Any]:
    entry = load_book_entry(manifest_path, book_id)
    if local_path is None:
        book_path = cache_dir / entry["content_filename"]
    else:
        book_path = local_path.expanduser().resolve()
    if not book_path.is_file():
        if not download:
            raise FastchessError(
                f"opening book is missing: {book_path}; rerun with --download-book"
            )
        _download_book(entry, book_path)
    actual_sri = sha384_sri(book_path)
    expected_sri = entry["content_sha384_base64"]
    if actual_sri != expected_sri:
        raise FastchessError(
            f"opening book hash mismatch for {book_path}: "
            f"expected {expected_sri}, got {actual_sri}"
        )
    result = dict(entry)
    result.update(
        {
            "book_id": book_id,
            "resolved_path": str(book_path.resolve()),
            "actual_content_sha384_base64": actual_sri,
            "verified": True,
        }
    )
    return result


def _engine_arg(path: Path, profile: str) -> list[str]:
    return [str(path.expanduser().resolve()), "--profile", profile]


def build_fastchess_command(
    fastchess_path: Path,
    engine_a: Path,
    engine_b: Path,
    profile: dict[str, Any],
    book_path: Path,
    output_dir: Path,
    decision_mode: str,
    rounds: Optional[int] = None,
    concurrency: Optional[int] = None,
    seed: Optional[int] = None,
    hash_mb: Optional[int] = None,
    time_control: Optional[str] = None,
) -> list[str]:
    baseline = profile["baseline"]
    candidate = profile["candidate"]
    effective_rounds = rounds if rounds is not None else int(profile["rounds"])
    effective_concurrency = concurrency if concurrency is not None else int(profile["concurrency"])
    effective_seed = seed if seed is not None else int(profile["seed"])
    effective_hash = hash_mb if hash_mb is not None else int(profile["hash_mb"])
    effective_tc = time_control or str(profile["time_control"])
    pgn_path = output_dir / "games.pgn"
    command = [
        str(fastchess_path.resolve()),
        "-strict",
        "-engine",
        f"cmd={engine_a.expanduser().resolve()}",
        f"name={baseline['label']}",
        f"args=--profile {baseline['search_profile']}",
        "-engine",
        f"cmd={engine_b.expanduser().resolve()}",
        f"name={candidate['label']}",
        f"args=--profile {candidate['search_profile']}",
        "-openings",
        f"file={book_path.resolve()}",
        f"format={profile.get('book_format', 'pgn')}",
        f"order={profile.get('opening_order', 'random')}",
        f"plies={int(profile.get('opening_plies', 0))}",
        "-srand",
        str(effective_seed),
        "-each",
        f"tc={effective_tc}",
        f"option.Hash={effective_hash}",
        "proto=uci",
        "-rounds",
        str(effective_rounds),
        "-repeat",
        "-concurrency",
        str(effective_concurrency),
        "-recover",
        "-pgnout",
        f"file={pgn_path.resolve()}",
        "notation=san",
        "nodes=true",
        "seldepth=true",
        "nps=true",
        "hashfull=true",
        "pv=true",
        "-report",
        "penta=true",
    ]
    if decision_mode == "sprt":
        sprt = profile.get("sprt")
        if not isinstance(sprt, dict):
            raise FastchessError("selected profile has no SPRT configuration")
        command.extend(
            [
                "-sprt",
                f"elo0={sprt['elo0']}",
                f"elo1={sprt['elo1']}",
                f"alpha={sprt['alpha']}",
                f"beta={sprt['beta']}",
                f"model={sprt['model']}",
            ]
        )
    return command


def classify_result(stdout: str, stderr: str, return_code: int, decision_mode: str) -> tuple[str, str]:
    if return_code != 0:
        return "REJECTED", f"fastchess-exit-{return_code}"
    if decision_mode != "sprt":
        return "INCONCLUSIVE", "fixed-games-no-sprt-decision"
    text = f"{stdout}\n{stderr}".lower()
    accepted = (
        re.search(r"h1(?:\s+was)?\s+accepted", text)
        or re.search(r"h0(?:\s+was)?\s+rejected", text)
    )
    rejected = (
        re.search(r"h0(?:\s+was)?\s+accepted", text)
        or re.search(r"h1(?:\s+was)?\s+rejected", text)
    )
    if accepted and not rejected:
        return "PASS", "fastchess-sprt"
    if rejected and not accepted:
        return "REJECTED", "fastchess-sprt"
    return "INCONCLUSIVE", "fastchess-sprt-no-boundary"


def build_manifest(
    args: argparse.Namespace,
    profile_name: str,
    profile: dict[str, Any],
    command: list[str],
    book: dict[str, Any],
    fastchess: Optional[dict[str, Any]],
) -> dict[str, Any]:
    engine_a_profile = profile["baseline"]["search_profile"]
    engine_b_profile = profile["candidate"]["search_profile"]
    engine_a_path = args.engine_a.expanduser().resolve()
    engine_b_path = args.engine_b.expanduser().resolve()
    effective_rounds = args.rounds if args.rounds is not None else int(profile["rounds"])
    effective_concurrency = args.concurrency if args.concurrency is not None else int(profile["concurrency"])
    effective_seed = args.seed if args.seed is not None else int(profile["seed"])
    effective_hash = args.hash_mb if args.hash_mb is not None else int(profile["hash_mb"])
    effective_tc = args.time_control or str(profile["time_control"])
    return {
        "schema_version": 1,
        "status": "DRY_RUN" if args.dry_run else "PREPARED",
        "profile_name": profile_name,
        "decision_mode": args.decision_mode,
        "command": command,
        "engine_a_baseline": {
            **file_metadata(engine_a_path),
            "label": profile["baseline"]["label"],
            "search_profile": engine_a_profile,
            "git_sha": args.sha_a,
            "argv": _engine_arg(engine_a_path, engine_a_profile),
        },
        "engine_b_candidate": {
            **file_metadata(engine_b_path),
            "label": profile["candidate"]["label"],
            "search_profile": engine_b_profile,
            "git_sha": args.sha_b,
            "argv": _engine_arg(engine_b_path, engine_b_profile),
        },
        "fastchess": fastchess,
        "opening_book": book,
        "uci_options": {"Hash": effective_hash},
        "effective_threads": int(profile["threads"]),
        "time_control": effective_tc,
        "rounds": effective_rounds,
        "games_expected": effective_rounds * 2,
        "concurrency": effective_concurrency,
        "seed": effective_seed,
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    profile = load_profile(args.profile_config, args.profile_name)
    book_id = args.book_id or str(profile["book"])
    book = ensure_book(
        args.book_manifest,
        book_id,
        args.book_path,
        args.book_cache_dir,
        args.download_book,
    )
    fastchess_path = locate_fastchess(args.fastchess)
    if fastchess_path is None and not args.dry_run:
        raise FastchessError("Fastchess was not found; pass --fastchess or add it to PATH")
    fastchess_info = fastchess_metadata(fastchess_path) if fastchess_path else None
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    command = build_fastchess_command(
        fastchess_path or Path("fastchess"),
        args.engine_a,
        args.engine_b,
        profile,
        Path(book["resolved_path"]),
        output_dir,
        args.decision_mode,
        rounds=args.rounds,
        concurrency=args.concurrency,
        seed=args.seed,
        hash_mb=args.hash_mb,
        time_control=args.time_control,
    )
    manifest = build_manifest(
        args, args.profile_name, profile, command, book, fastchess_info
    )
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest)
    if args.dry_run:
        return manifest

    manifest["status"] = "RUNNING"
    write_json(manifest_path, manifest)
    stdout_path = output_dir / "fastchess.stdout.log"
    stderr_path = output_dir / "fastchess.stderr.log"
    try:
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr:
            process = subprocess.run(
                command,
                cwd=REPO_ROOT,
                stdout=stdout,
                stderr=stderr,
                text=True,
                check=False,
            )
    except OSError as exc:
        manifest.update(
            {
                "status": "REJECTED",
                "reason": f"fastchess-launch-error: {exc}",
                "ended_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        write_json(manifest_path, manifest)
        return manifest
    stdout_text = stdout_path.read_text(encoding="utf-8", errors="replace")
    stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
    status, reason = classify_result(
        stdout_text, stderr_text, process.returncode, args.decision_mode
    )
    manifest.update(
        {
            "status": status,
            "reason": reason,
            "return_code": process.returncode,
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
            "stdout_tail": stdout_text[-8192:],
            "stderr_tail": stderr_text[-8192:],
            "ended_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    write_json(manifest_path, manifest)
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--fastchess", type=Path, default=None)
    result.add_argument("--engine-a", type=Path, required=True)
    result.add_argument("--engine-b", type=Path, required=True)
    result.add_argument("--sha-a", required=True, help="baseline Git SHA")
    result.add_argument("--sha-b", required=True, help="candidate Git SHA")
    result.add_argument("--profile-name", default="s2-current-vs-aspiration")
    result.add_argument("--profile-config", type=Path, default=DEFAULT_PROFILE_CONFIG)
    result.add_argument("--book-id", default=None)
    result.add_argument("--book-manifest", type=Path, default=DEFAULT_BOOK_MANIFEST)
    result.add_argument("--book-path", type=Path, default=None)
    result.add_argument("--book-cache-dir", type=Path, default=DEFAULT_BOOK_CACHE)
    result.add_argument("--download-book", action="store_true")
    result.add_argument("--decision-mode", choices=("sprt", "fixed"), default="sprt")
    result.add_argument("--rounds", type=int, default=None)
    result.add_argument("--concurrency", type=int, default=None)
    result.add_argument("--seed", type=int, default=None)
    result.add_argument("--hash-mb", type=int, default=None)
    result.add_argument("--time-control", default=None)
    result.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "fastchess")
    result.add_argument("--dry-run", action="store_true")
    return result


def main(argv: Optional[list[str]] = None) -> int:
    args = parser().parse_args(argv)
    try:
        manifest = run(args)
    except FastchessError as exc:
        print(f"fastchess_error {exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["status"] in {"PASS", "INCONCLUSIVE", "DRY_RUN"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
