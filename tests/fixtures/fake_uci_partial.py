#!/usr/bin/env python3
"""Fake UCI engine that writes a partial line WITHOUT a newline, then hangs.

readline() on the probe side blocks forever waiting for the newline; this
exercises the real-deadline path in probe_uci.
"""

import sys
import time

sys.stdout.write("id name PartialHang")
sys.stdout.flush()
for line in sys.stdin:
    pass
time.sleep(1000)
