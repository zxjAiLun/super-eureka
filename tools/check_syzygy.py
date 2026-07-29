"""Inspect optional local Syzygy files without enabling the engine feature."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def piece_count(path: Path) -> int | None:
    stem = path.stem.lower()
    if "v" not in stem:
        return None
    pieces = stem.replace("v", "")
    if not pieces or any(piece not in "kqrbnp" for piece in pieces):
        return None
    return len(pieces)


def inspect(manifest_path: Path, directories: list[Path]) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    configured = [Path(item).expanduser() for item in manifest.get("directories", [])]
    roots = [path.resolve() for path in [*configured, *directories]]
    files = []
    for root in roots:
        if not root.is_dir():
            continue
        files.extend(path for path in root.rglob("*") if path.suffix.lower() in {".rtbw", ".rtbz"})
    counts = {".rtbw": 0, ".rtbz": 0}
    piece_counts: dict[str, int] = {}
    for path in files:
        counts[path.suffix.lower()] += 1
        count = piece_count(path)
        if count is not None:
            key = str(count)
            piece_counts[key] = piece_counts.get(key, 0) + 1
    return {
        "schema_version": 1,
        "enabled": bool(manifest.get("enabled", False)),
        "piece_limit": int(manifest.get("piece_limit", 5)),
        "configured_directories": [str(root) for root in roots],
        "existing_directories": [str(root) for root in roots if root.is_dir()],
        "files": len(files),
        "by_extension": counts,
        "by_piece_count": piece_counts,
        "coverage_is_complete": False,
        "policy": manifest.get("policy"),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifest", type=Path, default=Path(__file__).with_name("syzygy_manifest.json"))
    result.add_argument("--directory", type=Path, action="append", default=[])
    result.add_argument("--require-files", action="store_true")
    result.add_argument("--json", action="store_true", dest="as_json")
    return result


def main() -> int:
    args = parser().parse_args()
    report = inspect(args.manifest, args.directory)
    if args.as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not args.require_files or report["files"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
