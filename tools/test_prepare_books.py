import tempfile
import unittest
import zipfile
from pathlib import Path

from prepare_books import (
    BookPreparationError,
    download_entry,
    normalize_line_endings,
    sha384_sri,
)


class BookPreparationTests(unittest.TestCase):
    def test_hash_failure_does_not_replace_existing_destination(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive_path = root / "book.zip"
            extracted = b"one\r\ntwo\rthree\n"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("dir/book.epd", extracted)
            destination = root / "cache" / "book.epd"
            destination.parent.mkdir()
            destination.write_bytes(b"known-good")
            entry = {
                "archive_url": archive_path.as_uri(),
                "archive_filename": archive_path.name,
                "content_filename": "book.epd",
                "content_sha384_base64": sha384_sri(b"wrong"),
            }
            with self.assertRaisesRegex(BookPreparationError, "raw hash mismatch"):
                download_entry(entry, destination)
            self.assertEqual(destination.read_bytes(), b"known-good")

    def test_normalization_handles_crlf_and_lone_cr(self):
        self.assertEqual(normalize_line_endings(b"a\r\nb\rc\n"), b"a\nb\nc\n")


if __name__ == "__main__":
    unittest.main()
