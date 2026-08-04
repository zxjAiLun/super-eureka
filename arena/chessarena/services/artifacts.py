"""Artifact paths, provenance and safe downloads (sections 13, 21).

Path handling rules:
- All run artifacts live under ``run_root/<tournament_id>``.
- Downloads resolve against the run root; any attempt to escape it is a 404.
- Every download is served with its SHA-256 via the artifact manifest.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..config import Settings

_run_root: Path | None = None
_build_root: Path | None = None
_opening_root: Path | None = None


def configure_artifact_service(settings: Settings) -> None:
    global _run_root, _build_root, _opening_root
    _run_root = settings.run_root
    _build_root = settings.build_root
    _opening_root = settings.opening_root


def get_run_root() -> Path:
    if _run_root is None:
        raise RuntimeError("artifact service not configured")
    return _run_root


def get_build_root() -> Path:
    if _build_root is None:
        raise RuntimeError("artifact service not configured")
    return _build_root


def get_opening_root() -> Path:
    if _opening_root is None:
        raise RuntimeError("artifact service not configured")
    return _opening_root


def tournament_run_dir(tournament_id: str) -> Path:
    return get_run_root() / tournament_id


def pair_run_dir(tournament_id: str, pair_index: int, attempt: int) -> Path:
    return (
        get_run_root()
        / tournament_id
        / "pairs"
        / f"{pair_index:06d}"
        / f"attempt-{attempt:02d}"
    )


def safe_resolve(base: Path, *parts: str) -> Path | None:
    """Resolve ``base/parts`` and reject any path escaping ``base``.

    Returns None when the path would escape the base directory.
    """
    try:
        candidate = base.joinpath(*parts).resolve()
    except (OSError, ValueError):
        return None
    resolved_base = base.resolve()
    if candidate != resolved_base and resolved_base not in candidate.parents:
        return None
    return candidate


def download_path(tournament_id: str, relative_path: str) -> Path | None:
    """Resolve a user-supplied artifact path under a tournament run dir."""
    parts = [p for p in relative_path.split("/") if p not in ("", ".")]
    if not parts or any(p == ".." for p in parts):
        return None
    return safe_resolve(tournament_run_dir(tournament_id), *parts)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(directory: Path, name: str, payload: Any) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Tournament-level artifacts (section 13)
# ---------------------------------------------------------------------------
def build_combined_pgn(tournament_id: str, game_pgn_paths: list[Path]) -> Path | None:
    """Concatenate all verified game PGNs into ``combined.pgn``.

    Returns None when there is nothing to combine.
    """
    if not game_pgn_paths:
        return None
    run_dir = tournament_run_dir(tournament_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    combined = run_dir / "combined.pgn"
    with open(combined, "w", encoding="utf-8", newline="\n") as out:
        for p in game_pgn_paths:
            if not Path(p).exists():
                continue
            text = Path(p).read_text(encoding="utf-8").strip()
            if text:
                out.write(text + "\n\n")
    if combined.stat().st_size == 0:
        return None
    return combined


def build_artifact_manifest(tournament_id: str, result_files: list[Path]) -> Path:
    """Write ``artifact-manifest.json`` with SHA-256 of every result file."""
    entries = {}
    for path in sorted(set(result_files)):
        if path.is_file():
            entries[path.name] = {
                "sha256": sha256_file(path),
                "size": path.stat().st_size,
            }
    run_dir = tournament_run_dir(tournament_id)
    payload = {
        "schema_version": 1,
        "tournament_id": tournament_id,
        "files": entries,
    }
    return write_json(run_dir, "artifact-manifest.json", payload)


def generate_tournament_artifacts(tournament) -> None:
    """Write combined.pgn / summary.json / artifact-manifest.json.

    Called once when a tournament reaches COMPLETED.  ``tournament`` is a
    Tournament ORM instance with its ``games`` relationship loaded.
    """
    games = tournament.games
    # A pair's two games share one match.pgn; deduplicate before combining.
    game_paths = list(dict.fromkeys(Path(g.pgn_path) for g in games if g.pgn_path))
    combined = build_combined_pgn(tournament.id, game_paths)

    wins = tournament.candidate_wins
    losses = tournament.candidate_losses
    draws = tournament.draws
    played = wins + losses + draws
    summary = write_json(
        tournament_run_dir(tournament.id),
        "summary.json",
        {
            "tournament_id": tournament.id,
            "name": tournament.name,
            "status": tournament.status,
            "time_control": tournament.time_control,
            "requested_pairs": tournament.requested_pairs,
            "completed_pairs": tournament.completed_pairs,
            "engine_a": {
                "build_id": tournament.engine_a_build_id,
                "profile": tournament.engine_a_profile,
            },
            "engine_b": {
                "build_id": tournament.engine_b_build_id,
                "profile": tournament.engine_b_profile,
            },
            "candidate_perspective": {
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "games": played,
                "score_percent": round((wins + 0.5 * draws) / played * 100, 2)
                if played
                else None,
            },
            "games": [
                {
                    "game_number": g.game_number,
                    "white": g.white_engine,
                    "black": g.black_engine,
                    "result": g.result,
                    "termination": g.termination,
                }
                for g in games
            ],
            "generated_utc": None,
        },
    )

    result_files: list[Path] = []
    for g in game_paths:
        if g.exists():
            result_files.append(g)
    if combined and combined.exists():
        result_files.append(combined)
    result_files.append(summary)
    build_artifact_manifest(tournament.id, result_files)
