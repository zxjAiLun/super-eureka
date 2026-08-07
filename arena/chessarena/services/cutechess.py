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


def engine_argv(engine: Dict[str, Any]) -> List[str]:
    """The ``-engine`` sub-args for one side.

    ``command_args`` (from the validated EnginePreset) are passed to the
    engine as ``arg=<value>``; engines without extra args (e.g. Stockfish)
    simply have an empty list.  This replaces the old hard-coded
    ``--profile`` assumption.
    """
    argv = [
        "cmd=" + engine["binary_path"],
        "proto=uci",
    ]
    for a in engine.get("command_args") or []:
        argv.append("arg=" + a)
    return argv


def _option_value(value: Any) -> str | None:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return None
    return str(value)


RESERVED_OPTIONS = frozenset({"Hash", "Threads", "Ponder"})


def engine_option_args(engine: Dict[str, Any]) -> List[str]:
    """Engine-specific UCI ``option.<name>=<value>`` lines.

    These belong to the engine's OWN ``-engine`` block (cutechess applies
    them to that engine only).  Putting them under ``-each`` would apply
    e.g. Stockfish's UCI_Elo to the opponent as well, which breaks
    different-strength slices.  Sorted by name for determinism; booleans
    render as true/false.
    """
    merged: Dict[str, Any] = dict(engine.get("uci_options") or {})
    args: List[str] = []
    for name in sorted(merged):
        rendered = _option_value(merged[name])
        if rendered is not None:
            args.append(f"option.{name}={rendered}")
    return args


def each_option_args(hash_mb: int, threads: int) -> List[str]:
    """Common UCI options applied to EVERY engine via ``-each``.

    Only arena-owned options live here (Hash/Threads); engine-specific
    preset options never go under ``-each``.
    """
    return [
        f"option.Hash={hash_mb}",
        f"option.Threads={threads}",
    ]


def validate_preset_options(uci_options: dict) -> None:
    """Reject presets that try to own runtime-reserved options.

    Hash/Threads/Ponder are owned by the arena runtime; a preset must not
    override them, otherwise the cutechess command would carry duplicate
    options with unclear precedence.
    """
    conflicts = sorted(RESERVED_OPTIONS & set(uci_options))
    if conflicts:
        raise CutechessLaunchError(
            f"preset must not set reserved options: {conflicts}"
        )


def build_pair_command(
    settings: Settings,
    *,
    engine_a: Dict[str, Any],
    engine_b: Dict[str, Any],
    time_control: str,
    hash_mb: int,
    opening_epd: Path,
    pgn_out: Path,
    threads: int = 1,
) -> List[str]:
    """Build the cutechess-cli argv for one 2-game color-swapped pair.

    The ``name=`` values are the preset display names (PGN-visible) with the
    legacy EngineA/EngineB constants as fallback for pre-preset tournaments.
    """
    a_name = engine_a.get("display_name") or ENGINE_A_NAME
    b_name = engine_b.get("display_name") or ENGINE_B_NAME
    argv: List[str] = [
        str(settings.cutechess),
        "-engine",
        "name=" + a_name,
        *engine_argv(engine_a),
        *engine_option_args(engine_a),
        "-engine",
        "name=" + b_name,
        *engine_argv(engine_b),
        *engine_option_args(engine_b),
        "-variant",
        "standard",
        "-openings",
        f"file={opening_epd}",
        "format=epd",
        "order=sequential",
        "policy=default",
        "-each",
        f"tc={time_control}",
        *each_option_args(hash_mb, threads),
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


def _group_alive_killpg(pgid: int) -> bool:
    """killpg(0) based group-liveness check (non-Linux fallback)."""
    if pgid <= 0:
        return False
    if hasattr(os, "killpg"):
        try:
            os.killpg(pgid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False
    return _pid_alive(pgid)


def _group_alive(pgid: int) -> bool:
    """True when any RUNNABLE member of the process group still exists.

    On Linux this scans /proc/*/stat for members whose pgrp matches and whose
    state is NOT 'Z' (zombie).  A zombie can no longer execute, so it must not
    keep the group "alive": otherwise an unreaped leader would make cleanup
    wait out the full grace period and falsely report a SIGKILL survivor
    (P1, zombie/reaping).  Non-Linux falls back to killpg(0).
    """
    if pgid <= 0:
        return False
    if sys.platform.startswith("linux"):
        try:
            entries = os.listdir("/proc")
        except OSError:
            return _group_alive_killpg(pgid)
        for name in entries:
            if not name.isdigit():
                continue
            try:
                stat = Path(f"/proc/{name}/stat").read_text(
                    encoding="utf-8", errors="replace"
                )
                rparen = stat.rfind(")")
                if rparen < 0:
                    continue
                # After the (comm) field: [0]=state, [1]=ppid, [2]=pgrp, ...
                fields = stat[rparen + 2:].split()
                if len(fields) < 3:
                    continue
                state, pgrp = fields[0], int(fields[2])
                if pgrp == pgid and state != "Z":
                    return True
            except (OSError, ValueError, IndexError):
                continue
        return False
    return _group_alive_killpg(pgid)


def _pid_alive(pid: int) -> bool:
    """True when a process with ``pid`` exists.

    On POSIX this uses ``os.kill(pid, 0)`` which is a pure existence probe.
    On Windows ``os.kill(pid, 0)`` would call TerminateProcess (i.e. it KILLS
    the process), so a handle-based OpenProcess probe is used instead.
    """
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid)
        )
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def _terminate_group_polling(pgid: int, grace_seconds: float,
                             reap: callable | None = None) -> bool:
    """Shared SIGTERM -> wait -> SIGKILL -> wait loop for a process group.

    ``reap`` is an optional per-iteration hook (e.g. ``proc.poll()``) that
    reaps an exited leader so a zombie cannot keep the group "alive" via
    killpg(0).  Returns True only when no runnable member remains; False when
    the group survived SIGKILL.
    """
    import time

    def signal_group(sig):
        if hasattr(os, "killpg"):
            try:
                os.killpg(pgid, sig)
            except (ProcessLookupError, PermissionError, OSError):
                pass
        else:
            try:
                os.kill(pgid, sig)
            except (ProcessLookupError, PermissionError, OSError):
                pass

    signal_group(signal.SIGTERM)
    deadline = time.time() + grace_seconds
    while time.time() < deadline:
        if reap is not None:
            reap()
        if not _group_alive(pgid):
            return True
        time.sleep(0.1)

    signal_group(getattr(signal, "SIGKILL", signal.SIGTERM))
    deadline = time.time() + 10
    while time.time() < deadline:
        if reap is not None:
            reap()
        if not _group_alive(pgid):
            return True
        time.sleep(0.1)
    return False  # group survived SIGKILL: identity must be retained


def terminate_process_group_by_pid(pgid: int, grace_seconds: float) -> bool:
    """Group-aware termination by PGID (used by recovery, P1).

    Returns True only when the entire process group is confirmed gone.  The
    caller has no Popen handle to reap an exited leader, so the zombie-aware
    /proc liveness check is what distinguishes a reaped-able zombie from a
    surviving child.
    """
    return _terminate_group_polling(pgid, grace_seconds, reap=None)


def terminate_process_group(proc: subprocess.Popen, grace_seconds: float) -> bool:
    """Terminate a Popen's whole process group; reaps the leader handle.

    Returns True only when the whole group is confirmed gone (P1): a reaped
    leader is NOT a success by itself - an engine child in the same PGID may
    still be running, so the entry point must keep checking ``_group_alive``
    even after ``proc.poll()`` returns non-None (second/retry cleanup).

    On POSIX the leader is polled/reaped during each wait round so an exited
    leader is removed from the process table promptly, and the zombie-aware
    /proc check distinguishes a reaped zombie from a surviving child.
    Windows has no process groups and os.kill/OpenProcess cannot track a
    killed-but-unreaped leader reliably, so the leader is terminated and
    reaped through its Popen handle (the production host is Linux; Windows is
    the test/development platform).
    """
    if os.name != "posix":
        if proc.poll() is not None:
            return True
        try:
            proc.send_signal(signal.SIGTERM)
        except OSError:
            pass
        try:
            proc.wait(timeout=grace_seconds)
            return True
        except subprocess.TimeoutExpired:
            pass
        try:
            proc.kill()
        except OSError:
            pass
        try:
            proc.wait(timeout=10)
            return True
        except subprocess.TimeoutExpired:
            return False

    # POSIX: leader exited != process group gone.
    pgid = proc.pid  # launched with start_new_session -> leader is group leader
    proc.poll()  # reap the leader when possible
    if not _group_alive(pgid):
        return True
    terminated = _terminate_group_polling(
        pgid, grace_seconds, reap=lambda: proc.poll()
    )
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    return terminated


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

    BOTH pieces of evidence must be present and must match exactly (P1):
    - the kernel starttime marker (not reusable after process exit), and
    - the full argv from /proc.
    The cmdline alone cannot guard against PID reuse (two cutechess
    invocations can share an argv), so if either piece of evidence cannot be
    read the check fails closed and the caller must not kill the PID.
    """
    if recorded_marker is None or recorded_cmdline is None:
        return False  # no recorded identity -> fail closed
    current_marker = process_start_marker(pid)
    current_cmdline = process_cmdline(pid)
    if current_marker is None or current_cmdline is None:
        return False
    return current_marker == recorded_marker and current_cmdline == recorded_cmdline
