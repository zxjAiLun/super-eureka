"""Process identity helpers used for safe orphan cleanup (P1).

- On non-Linux platforms /proc is unavailable: the identity helpers must
  return None and ``verify_process_identity`` must refuse to claim ownership
  (so recovery never kills a PID it cannot positively identify).
- On Linux a spawned process must be positively identified by starttime +
  cmdline, and a PID whose recorded identity no longer matches (reuse guard)
  must be rejected.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from chessarena.services import cutechess as cc


def _spawn_sleeper() -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def test_non_linux_identity_is_unavailable():
    if sys.platform.startswith("linux"):
        pytest.skip("linux platform")
    proc = _spawn_sleeper()
    try:
        pid = proc.pid
        assert cc.process_start_marker(pid) is None
        assert cc.process_cmdline(pid) is None
        # Never claim ownership without identity evidence.
        assert cc.verify_process_identity(pid, None, None) is False
    finally:
        proc.kill()
        proc.wait()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="needs /proc")
def test_linux_positive_identification():
    proc = _spawn_sleeper()
    try:
        pid = proc.pid
        marker = cc.process_start_marker(pid)
        cmdline = cc.process_cmdline(pid)
        assert marker is not None
        assert cmdline is not None
        assert cc.verify_process_identity(pid, marker, cmdline) is True
    finally:
        proc.kill()
        proc.wait()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="needs /proc")
def test_linux_pid_reuse_guard():
    """A recycled PID with a different recorded identity must be refused."""
    proc = _spawn_sleeper()
    other = _spawn_sleeper()
    try:
        pid = proc.pid
        marker = cc.process_start_marker(pid)
        cmdline = cc.process_cmdline(pid)
        # Wrong start marker -> PID reuse -> refuse.
        assert cc.verify_process_identity(pid, "0", cmdline) is False
        # Wrong cmdline -> refuse.
        assert cc.verify_process_identity(pid, marker, ["/bin/not-cutechess"]) is False
        # Same marker/cmdline from another live process -> refuse.
        other_pid = other.pid
        other_marker = cc.process_start_marker(other_pid)
        other_cmdline = cc.process_cmdline(other_pid)
        assert cc.verify_process_identity(pid, other_marker, other_cmdline) is False
        # cmdline must match exactly, not just contain the binary.
        assert cc.verify_process_identity(
            pid, marker, [cmdline[0], "extra-arg"]
        ) is False
    finally:
        proc.kill()
        proc.wait()
        other.kill()
        other.wait()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="needs /proc")
def test_linux_dead_pid_identity_disappears():
    proc = _spawn_sleeper()
    pid = proc.pid
    marker = cc.process_start_marker(pid)
    cmdline = cc.process_cmdline(pid)
    proc.kill()
    proc.wait()
    for _ in range(100):
        if not _pid_alive(pid):
            break
        time.sleep(0.05)
    assert _pid_alive(pid) is False
    # After death, reading identity yields None and verification fails closed.
    assert cc.process_start_marker(pid) is None
    assert cc.verify_process_identity(pid, marker, cmdline) is False


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="needs process groups")
def test_linux_cleanup_waits_for_whole_group():
    """P1: when the group leader exits but a SIGTERM-ignoring child survives,
    cleanup must escalate to SIGKILL for the whole group, not stop at the
    leader PID."""
    from chessarena.services import recovery

    leader = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import subprocess, time, sys\n"
            "subprocess.Popen([sys.executable, '-c', "
            "'import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "time.sleep(30)'])\n"
            "time.sleep(30)\n",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,  # leader is its own group leader
    )
    pgid = leader.pid
    try:
        # Leader exits; the child (same PGID) ignores SIGTERM and survives.
        leader.kill()
        leader.wait()
        assert recovery._group_alive(pgid) is True, "child should still be alive"

        class _ShortGrace:
            shutdown_grace_seconds = 1.0

        recovery._terminate_pid_group_wait(pgid, settings=_ShortGrace())
        # The whole group must be gone: SIGTERM was ignored, so SIGKILL was
        # required, and cleanup waited for the group rather than the leader.
        assert recovery._group_alive(pgid) is False
    finally:
        # Belt and braces: never leave a stray sleeping child behind.
        try:
            os.killpg(pgid, 9)
        except (ProcessLookupError, OSError):
            pass


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
