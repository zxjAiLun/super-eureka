"""Download and verify the selected official Stockfish book suites.

Archives are pinned to a repository commit and only the named member is
extracted.  The downloaded content lives under the ignored ``books/cache``
directory.  ``--update-manifest`` records the raw extracted-content hash and
the normalized upstream SRI side by side; this matters because the upstream
books repository normalizes line endings before publishing its SRI values.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "books" / "manifest.json"
DEFAULT_CACHE = REPO_ROOT / "books" / "cache"


class BookPreparationError(RuntimeError):
    pass


def sha384_sri(data: bytes) -> str:
    return base64.b64encode(hashlib.sha384(data).digest()).decode("ascii")


def read_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_member(member: zipfile.ZipInfo, filename: str) -> bool:
    path = Path(member.filename)
    return (
        not member.is_dir()
        and path.name == filename
        and not any(part in {"", ".", ".."} for part in path.parts)
    )


def download_entry(entry: dict, destination: Path) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    archive_name = entry.get("archive_filename") or Path(entry["archive_url"]).name
    archive_bytes = 0
    with tempfile.TemporaryDirectory(prefix="stockfish-books-") as temp_dir:
        archive_path = Path(temp_dir) / archive_name
        try:
            with urlopen(entry["archive_url"], timeout=120) as response, archive_path.open("wb") as target:
                shutil.copyfileobj(response, target)
            archive_bytes = archive_path.stat().st_size
            with zipfile.ZipFile(archive_path) as archive:
                matches = [
                    member for member in archive.infolist()
                    if safe_member(member, entry["content_filename"])
                ]
                if len(matches) != 1:
                    raise BookPreparationError(
                        f"{entry['content_filename']}: expected one archive member, got {len(matches)}"
                    )
                content = archive.read(matches[0])
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            raise BookPreparationError(f"cannot download {entry['content_filename']}: {exc}") from exc

    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.write_bytes(content)
    temporary.replace(destination)
    raw_sri = sha384_sri(content)
    normalized_sri = sha384_sri(content.replace(b"\r\n", b"\n"))
    expected = entry.get("content_sha384_base64")
    if expected and expected != raw_sri:
        raise BookPreparationError(
            f"{entry['content_filename']}: raw hash mismatch; expected {expected}, got {raw_sri}"
        )
    upstream = entry.get("upstream_normalized_sri")
    if upstream and upstream != normalized_sri:
        raise BookPreparationError(
            f"{entry['content_filename']}: normalized hash mismatch; expected {upstream}, got {normalized_sri}"
        )
    return {
        "resolved_path": str(destination.resolve()),
        "archive_bytes": archive_bytes,
        "content_bytes": len(content),
        "raw_content_sha384_base64": raw_sri,
        "normalized_content_sha384_base64": normalized_sri,
        "verified": True,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    result.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    result.add_argument("--book-id", action="append", dest="book_ids")
    result.add_argument("--update-manifest", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    manifest = read_manifest(args.manifest)
    books = manifest.get("books")
    if not isinstance(books, dict):
        raise BookPreparationError("manifest has no books object")
    selected = args.book_ids or [
        book_id for book_id, entry in books.items()
        if entry.get("selected_for_preflight")
    ]
    if not selected:
        raise BookPreparationError("no book selected; use --book-id or selected_for_preflight")

    report = []
    for book_id in selected:
        if book_id not in books:
            raise BookPreparationError(f"unknown book id: {book_id}")
        entry = books[book_id]
        destination = args.cache_dir / entry["content_filename"]
        result = download_entry(entry, destination)
        if args.update_manifest:
            entry["content_sha384_base64"] = result["raw_content_sha384_base64"]
            entry["raw_content_sha384_base64"] = result["raw_content_sha384_base64"]
            entry["normalized_content_sha384_base64"] = result[
                "normalized_content_sha384_base64"
            ]
        report.append({"book_id": book_id, **result})

    if args.update_manifest:
        args.manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8"
        )
    print(json.dumps({"manifest": str(args.manifest.resolve()), "books": report}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BookPreparationError as exc:
        raise SystemExit(f"book preparation failed: {exc}") from exc
