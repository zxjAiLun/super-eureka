import json
from pathlib import Path
import tempfile
import unittest

from prepare_d114_openings import OpeningError, prepare, parser, verify


class D114OpeningPreparationTests(unittest.TestCase):
    def _args(self, root: Path, source: Path, output: Path, metadata: Path, seed: int = 7):
        return parser().parse_args(
            [
                "--source",
                str(source),
                "--output",
                str(output),
                "--metadata",
                str(metadata),
                "--count",
                "3",
                "--smoke-count",
                "2",
                "--seed",
                str(seed),
            ]
        )

    def test_prepare_is_deterministic_and_verifiable(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.epd"
            source.write_text(
                "\n".join(
                    [
                        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
                        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            output = root / "selected.epd"
            metadata = root / "selected.json"
            args = self._args(root, source, output, metadata)
            first = prepare(args)
            self.assertEqual(first["output"]["position_count"], 3)
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 3)
            checked = verify(
                parser().parse_args(
                    [
                        "--verify",
                        "--source",
                        str(source),
                        "--output",
                        str(output),
                        "--metadata",
                        str(metadata),
                    ]
                )
            )
            self.assertEqual(checked["status"], "PASS")
            before = output.read_bytes()
            prepare(args)
            self.assertEqual(before, output.read_bytes())

    def test_prepare_rejects_terminal_positions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.epd"
            source.write_text("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1\n", encoding="utf-8")
            with self.assertRaises(OpeningError):
                prepare(self._args(root, source, root / "selected.epd", root / "selected.json"))

    def test_verify_rejects_output_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.epd"
            source.write_text(
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1\n"
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1\n"
                "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3\n",
                encoding="utf-8",
            )
            output = root / "selected.epd"
            metadata = root / "selected.json"
            prepare(self._args(root, source, output, metadata))
            output.write_text(output.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(OpeningError, "SHA-256 mismatch"):
                verify(
                    parser().parse_args(
                        ["--verify", "--output", str(output), "--metadata", str(metadata)]
                    )
                )


if __name__ == "__main__":
    unittest.main()
