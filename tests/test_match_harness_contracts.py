"""Fast harness tripwires; no models, builds or games required."""
import importlib.util
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S15 = load("s15_screen", "tools/s15/s15_screen.py")
S17 = load("s17_match", "tools/s17/s17_ab_match.py")


class EvaluatorContract(unittest.TestCase):
    def test_explicit_nnue_in_both_game_commands(self):
        cmd = S15.build_command(Path("book.epd"), Path("match.pgn"), 4)
        self.assertEqual(cmd.count("arg=--evaluation"), 2)
        self.assertEqual(cmd.count("arg=nnue"), 2)
        self.assertEqual(cmd[cmd.index("-rounds") + 1], "256")
        self.assertEqual(cmd[cmd.index("-repeat") + 1], "2")

    def test_real_handshake_readback_contract(self):
        good = ("info string profile current-final\ninfo string eval nnue-v2q\n"
                "info string network nnue-v2q\ninfo string evalfile model.bin\nuciok\nreadyok\n")
        variants = [(good, 0, True),
                    (good.replace("eval nnue-v2q", "eval handcrafted"), 0, False),
                    (good.replace("network nnue-v2q", "network none"), 0, False),
                    (good.replace("model.bin", "wrong.bin"), 0, False),
                    ("", 0, False), (good, 2, False)]
        for module, verify in [(S15, S15.verify_evaluator), (S17, S17.verify_arm)]:
            for stdout, rc, passes in variants:
                with self.subTest(module=module.__name__, rc=rc, stdout=stdout):
                    result = subprocess.CompletedProcess([], rc, stdout=stdout, stderr="")
                    with patch.object(module.subprocess, "run", return_value=result) as run:
                        if passes:
                            self.assertEqual(verify(Path("engine.exe"), Path("model.bin"), "test")["eval"], "nnue-v2q")
                        else:
                            with self.assertRaises(SystemExit):
                                verify(Path("engine.exe"), Path("model.bin"), "test")
                        command = run.call_args.args[0]
                        self.assertEqual(command[command.index("--evaluation") + 1], "nnue")
                        self.assertIn("--nnue-model", command)


if __name__ == "__main__":
    unittest.main()
