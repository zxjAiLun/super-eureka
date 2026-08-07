#!/usr/bin/env python3
"""Fake UCI engine that hangs forever after receiving 'uci' (no output)."""

import sys
import time

for line in sys.stdin:
    pass  # deliberately never emits anything
time.sleep(1000)
