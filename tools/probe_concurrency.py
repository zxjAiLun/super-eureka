"""Report host CPU topology and a conservative Fastchess concurrency hint.

Fastchess concurrency is process-level parallelism.  The engine remains
single-threaded; this tool intentionally does not modify a profile or start a
match.  The default recommendation leaves one physical core for the host.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Optional


def _windows_physical_cores() -> Optional[int]:
    try:
        completed = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores -Sum).Sum",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    try:
        value = int(completed.stdout.strip())
    except ValueError:
        return None
    return value if value > 0 else None


def _linux_physical_cores() -> Optional[int]:
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.is_file():
        return None
    physical: set[tuple[str, str]] = set()
    processor: Optional[str] = None
    core: Optional[str] = None
    for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines() + [""]:
        key, _, value = line.partition(":")
        if key.strip() == "physical id":
            processor = value.strip()
        elif key.strip() == "core id":
            core = value.strip()
        elif not line.strip():
            if processor is not None and core is not None:
                physical.add((processor, core))
            processor = None
            core = None
    return len(physical) or None


def _mac_physical_cores() -> Optional[int]:
    try:
        completed = subprocess.run(
            ["sysctl", "-n", "hw.physicalcpu"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    try:
        value = int(completed.stdout.strip())
    except ValueError:
        return None
    return value if value > 0 else None


def physical_cores() -> Optional[int]:
    system = platform.system()
    if system == "Windows":
        return _windows_physical_cores()
    if system == "Linux":
        return _linux_physical_cores()
    if system == "Darwin":
        return _mac_physical_cores()
    return None


def probe() -> dict[str, object]:
    logical = os.cpu_count() or 1
    physical = physical_cores() or logical
    recommendation = max(1, min(logical, physical - 1 if physical > 1 else 1))
    return {
        "schema_version": 1,
        "platform": platform.platform(),
        "logical_processors": logical,
        "physical_cores": physical,
        "recommended_fastchess_concurrency": recommendation,
        "engine_thread_model": "single-threaded",
        "policy": "physical-cores-minus-one",
        "formal_match_started": False,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--json", action="store_true", dest="as_json")
    return result


def main() -> int:
    args = parser().parse_args()
    report = probe()
    if args.as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"physical cores: {report['physical_cores']}")
        print(f"logical processors: {report['logical_processors']}")
        print(f"recommended Fastchess concurrency: {report['recommended_fastchess_concurrency']}")
        print("engine thread model: single-threaded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
