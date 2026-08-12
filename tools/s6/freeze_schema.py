#!/usr/bin/env python3
"""S6.1A: freeze the FeatureSetV1 schema.

Runs `bench eval-features-schema` (the canonical schema content), computes a
REAL SHA-256 (64 lowercase hex) over those exact bytes, and writes
docs/s6/s6-feature-v1.json including schema_sha256. The Rust
`FEATURE_SCHEMA_SHA256` constant must be set to the printed value; a Rust
unit test fails while the constant is empty or malformed (see features.rs).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, help="release binary")
    parser.add_argument("--out", default="docs/s6/s6-feature-v1.json")
    args = parser.parse_args(sys.argv[1:])

    out = subprocess.run(
        [args.engine, "bench", "eval-features-schema"],
        capture_output=True, text=True, timeout=120,
    )
    if out.returncode != 0:
        print(out.stderr)
        return 2
    canonical = out.stdout.strip()
    schema = json.loads(canonical)
    assert schema["feature_count"] == 227
    sha = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert len(sha) == 64
    schema["schema_sha256"] = sha
    out_path = Path(args.out)
    out_path.write_text(json.dumps(schema, indent=1) + "\n", encoding="utf-8")
    print(f"schema_sha256={sha}")
    print(f"wrote {out_path}")
    print("set FEATURE_SCHEMA_SHA256 in src/engine/features.rs to the value above")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
