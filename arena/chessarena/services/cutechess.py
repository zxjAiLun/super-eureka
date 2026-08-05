"""cutechess-cli process supervision (section 12).

Every pair runs one cutechess invocation in its own process group.  The argv
is built from validated database records and fixed presets only; user input
can never reach the command line as a free-form argument.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from ..config import ENGINE_A_NAME, ENGINE_B_NAME, Settings
from . import artifacts


class CutechessLaunchError(RuntimeError):
    pass


def engine_a_argv(build: Dict[str, Any]) -> List[str]:
    return [
        "cmd=" + build["binary_path"],
        "proto=uci",
        "arg=--profile",
        "arg=" + build["profile"],
    ]


def build_pair_command(
    settings: Settings,
    *,
    engine_a: Dict[str, Any],
    engine_b: Dict[str, Any],
    time_control: str,
    hash_mb: int,
    opening_epd: Path,
    pgn_out: Path,
) -> List[str]:
    """Build the cutechess-cli argv for one 2-game color-swapped pair."""
    argv: List[str] = [
        str(settings.cutechess),
        "-engine",
        "name=" + ENGINE_A_NAME,
        *engine_a_argv(engine_a),
        "-engine",
        "name=" + ENGINE_B_NAME,
        *engine_a_argv(engine_b),
        "-variant",
        "standard",
        "-openings",
        f"file={opening_epd}",
        "format=epd",
        "order=sequential",
        "policy=default",
        "-each",
        f"tc={time_control}",
        f"option.Hash={hash_mb}",
        # One opening position per pair; -repeat 2 plays it twice with the
        # sides swapped, giving exactly two games with strict color reversal.
        "-rounds",
        "2",
        "-repeat",
        "2",
        "-concurrency",
        "1",
        "-pgnout",
        str(pgn_out),
        "-resultformat",
        "short",
    ]
    return argv


def write_command_artifacts(pair_dir: Path, argv: List[str], extra: Dict[str, Any]) -> None:
    """Persist the exact command as text and JSON before launch (section 12)."""
    pair_dir.mkdir(parents=True, exist_ok=True)
    (pair_dir / "command.txt").write_text(" ".join(argv) + "\n", encoding="utf-8")
    artifacts.write_json(
        pair_dir,
        "command.json",
        {
            "schema_version": 1,
            "argv": argv,
            "cwd": str(pair_dir),
            "shell": False,
            **extra,
        },
    )


def check_cutechess(settings: Settings) -> str:
    """Return the cutechess version string; raise if missing or broken."""
    path = settings.cutechess
    if not path.exists():
        raise CutechessLaunchError(f"cutechess-cli not found: {path}")
    try:
        result = subprocess.run(
            [str(path), "-version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CutechessLaunchError(f"cannot run cutechess-cli: {exc}") from exc
    if result.returncode != 0:
        raise CutechessLaunchError(
            f"cutechess-cli -version failed: rc={result.returncode}"
        )
    return (result.stdout or result.stderr or "").strip().splitlines()[0]


def check_engine_binary(build: Dict[str, Any]) -> None:
    """Re-check the engine binary SHA before launching (section 12)."""
    from . import artifacts

    path = Path(build["binary_path"])
    if not path.exists():
        raise CutechessLaunchError(
            f"engine binary missing: {path} (build {build.get('build_id')})"
        )
    actual = artifacts.sha256_file(path)
    if actual != build["binary_sha256"]:
        raise CutechessLaunchError(
            f"engine binary SHA mismatch for {path}: "
            f"expected {build['binary_sha256']} got {actual}"
        )


def launch_cutechess(argv: List[str], pair_dir: Path) -> subprocess.Popen:
    """Launch cutechess in a new process group with file redirection.

    ``shell`` is always False; args go directly to exec.
    """
    stdout_fh = open(pair_dir / "stdout.log", "wb")
    stderr_fh = open(pair_dir / "stderr.log", "wb")
    try:
        proc = subprocess.Popen(
            argv,
            cwd=str(pair_dir),
            stdin=subprocess.DEVNULL,
            stdout=stdout_fh,
            stderr=stderr_fh,
            start_new_session=True,  # own process group -> killable as a unit
            shell=False,
        )
    except Exception:
        stdout_fh.close()
        stderr_fh.close()
        raise
    # Hand ownership of the file handles to the Popen object so they are
    # closed when the process exits.
    proc._stdout_fh = stdout_fh  # type: ignore[attr-defined]
    proc._stderr_fh = stderr_fh  # type: ignore[attr-defined]
    return proc


def _kill_group(proc: subprocess.Popen, sig) -> None:
    """Send a signal to the whole process group (POSIX) or the process (Windows)."""
    if hasattr(os, "killpg"):
        try:
            os.killpg(os.getpgid(proc.pid), sig)
            return
        except (ProcessLookupError, PermissionError, OSError):
            return
    try:
        proc.send_signal(sig)
    except OSError:
        pass


def terminate_process_group(proc: subprocess.Popen, grace_seconds: float) -> None:
    """SIGTERM the process group, wait, then SIGKILL (section 19)."""
    if proc.poll() is not None:
        return
    _kill_group(proc, signal.SIGTERM)
    try:
        proc.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    sigkill = getattr(signal, "SIGKILL", signal.SIGTERM)
    _kill_group(proc, sigkill)
    proc.wait(timeout=10)


def read_output_lines(path: Path, max_bytes: int = 4 * 1024 * 1024) -> List[str]:
    """Read tail of an output file for inspection (worker-incremental reads)."""
    if not path.exists():
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    if len(text) > max_bytes:
        text = text[-max_bytes:]
    return [line.rstrip("\n") for line in text.splitlines()]


# ---------------------------------------------------------------------------
# Process identity (P1: safe orphan cleanup)
# ---------------------------------------------------------------------------
def process_start_marker(pid: int) -> str | None:
    """A value that uniquely identifies a process across time.

    On Linux this is the kernel starttime (``/proc/<pid>/stat`` field 22),
    which is NOT reused after process exit, so an old PID that has been
    recycled for an unrelated process will have a different marker.  Returns
    None on platforms without /proc.
    """
    if not sys.platform.startswith("linux"):
        return None
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace")
        rparen = stat.rfind(")")
        if rparen < 0:
            return None
        # Fields after the comm field: field 3 onwards.  starttime is field 22.
        fields = stat[rparen + 2:].split()
        if len(fields) >= 20:
            return fields[19]
    except OSError:
        return None
    return None


def process_cmdline(pid: int) -> list[str] | None:
    """The argv of ``pid`` (Linux /proc), or None when unavailable."""
    if not sys.platform.startswith("linux"):
        return None
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        args = [a for a in raw.split(b"\x00") if a]
        return [a.decode("utf-8", errors="replace") for a in args]
    except OSError:
        return None


def verify_process_identity(pid: int, recorded_marker: str | None,
                            recorded_cmdline: list[str] | None) -> bool:
    """Confirm ``pid`` still refers to the same process that was recorded.

    On Linux both the start marker and the cmdline must match.  When no
    identity was recorded (non-Linux test environments), we cannot prove
    ownership, so the caller should not kill on that basis alone; this returns
    True only when both pieces of evidence are available and consistent.
    """
    if recorded_marker is None and not recorded_cmdline:
        return False  # no identity evidence recorded -> do not kill blindly
    current_marker = process_start_marker(pid)
    if recorded_marker is not None and current_marker != recorded_marker:
        return False  # PID was reused by an unrelated process
    current_cmdline = process_cmdline(pid)
    if recorded_cmdline is not None:
        if current_cmdline is None or current_cmdline != recorded_cmdline:
            return False
    # cmdline/starttime agree; on platforms where we could not record either
    # (we returned early above) this function is not reached.
    return True
