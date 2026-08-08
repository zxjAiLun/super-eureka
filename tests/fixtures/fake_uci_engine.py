#!/usr/bin/env python3
"""Fake UCI engine for P4.2 Phase B script tests.

Emits a minimal UCI handshake (id name, option lines, uciok, readyok) and
exits on "quit".  The UCI_Elo range can be narrowed via env vars so script
tests can exercise out-of-range rejection.
"""

import os
import sys


def main() -> int:
    elo_min = os.environ.get("FAKE_UCI_ELO_MIN", "-200")
    elo_max = os.environ.get("FAKE_UCI_ELO_MAX", "2850")
    lines = [
        "id name FakeStockfish 17.1",
        "option name UCI_LimitStrength type check default false",
        f"option name UCI_Elo type spin default 1350 min {elo_min} max {elo_max}",
        "option name Hash type spin default 16 min 1 max 1024",
        "option name Threads type spin default 1 min 1 max 1",
        "option name Ponder type check default false",
        "option name Style type combo default Normal var Solid var Normal var Risky",
        "option name Clear Hash type button",
        "option name SyzygyPath type string default <empty>",
        "option name My Custom Option type string default some default value",
        "option name Move Overhead type spin default 10 min 0 max 5000",
        "uciok",
    ]
    for line in lines:
        print(line, flush=True)
    for line in sys.stdin:
        cmd = line.strip()
        if cmd == "isready":
            print("readyok", flush=True)
        elif cmd == "quit":
            return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
