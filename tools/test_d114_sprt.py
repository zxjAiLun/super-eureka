import sys
from pathlib import Path
import tempfile
import unittest

from run_d114_sprt import (
    BASELINE_LABEL,
    BASELINE_PROFILE,
    CANDIDATE_LABEL,
    CANDIDATE_PROFILE,
    D114Error,
    build_command,
    probe_engine_identity,
    refuse_reused_output,
)


class D114LauncherTests(unittest.TestCase):
    @staticmethod
    def command_block(command: list[str], index: int) -> list[str]:
        start = command.index("-engine", index)
        try:
            end = command.index("-engine", start + 1)
        except ValueError:
            end = len(command)
        return command[start:end]

    def test_smoke_command_is_candidate_first_and_not_sprt(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            command = build_command(
                root / "cutechess-cli.exe",
                root / "chess-engine-demo.exe",
                root / "openings.epd",
                root,
                "Smoke",
                1,
            )
        candidate = self.command_block(command, 0)
        baseline = self.command_block(command, command.index("-engine") + 1)
        self.assertIn(f"name={CANDIDATE_LABEL}", candidate)
        self.assertIn(f"arg={CANDIDATE_PROFILE}", candidate)
        self.assertIn(f"name={BASELINE_LABEL}", baseline)
        self.assertIn(f"arg={BASELINE_PROFILE}", baseline)
        self.assertEqual(command[command.index("-each") + 1 : command.index("-rounds")], [
            "tc=5+0.05",
            "option.Hash=16",
        ])
        self.assertEqual(command[command.index("-rounds") + 1], "40")
        self.assertEqual(command[command.index("-repeat") + 1], "2")
        self.assertEqual(command[command.index("-concurrency") + 1], "1")
        self.assertNotIn("-sprt", command)
        self.assertNotIn("-recover", command)

    def test_formal_command_has_candidate_first_sprt_and_full_rounds(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            command = build_command(
                root / "cutechess-cli.exe",
                root / "chess-engine-demo.exe",
                root / "openings.epd",
                root,
                "Sprt",
                2,
            )
        self.assertEqual(command[command.index("-rounds") + 1], "9704")
        self.assertEqual(command[command.index("-concurrency") + 1], "2")
        self.assertEqual(command[command.index("-each") + 1 : command.index("-rounds")], [
            "tc=10+0.1",
            "option.Hash=16",
        ])
        sprt = command.index("-sprt")
        self.assertEqual(command[sprt + 1 : sprt + 5], [
            "elo0=0",
            "elo1=5",
            "alpha=0.05",
            "beta=0.05",
        ])

    def test_identity_probe_sends_uci_and_quit(self):
        script = (
            "import sys\n"
            "for line in sys.stdin:\n"
            "    if line.strip() == 'uci':\n"
            "        print('id name D114Fake', flush=True)\n"
            "        print('id author tests', flush=True)\n"
            "        print('info string search profile current-lmr', flush=True)\n"
            "        print('uciok', flush=True)\n"
        )
        identity = probe_engine_identity([sys.executable, "-u", "-c", script])
        self.assertEqual(identity["return_code"], 0)
        self.assertTrue(identity["uciok"])
        self.assertEqual(identity["id_name"], "D114Fake")
        self.assertEqual(identity["reported_profile"], "current-lmr")

    def test_reused_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "run"
            output.mkdir()
            (output / "existing.txt").write_text("do not overwrite", encoding="utf-8")
            with self.assertRaises(D114Error):
                refuse_reused_output(output)


if __name__ == "__main__":
    unittest.main()
