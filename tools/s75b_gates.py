#!/usr/bin/env python3
"""Run the S7.5B candidate through the shared S7.5 offline gates."""

from __future__ import annotations

import sys

from s75a_gates import main


if "--candidate-profile" not in sys.argv:
    sys.argv.extend(["--candidate-profile", "current-final-bounded-check2"])
if "--out" not in sys.argv:
    sys.argv.extend(["--out", "results/s7/s75b-gates.json"])

raise SystemExit(main())
