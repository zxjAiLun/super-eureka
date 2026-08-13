"""Measure host topology and empirical Fastchess process concurrency.

Topology mode is read-only and does not start an engine.  Empirical mode
starts independent single-threaded UCI engine processes, runs the same fixed
node suite in each worker, and selects the concurrency with the best aggregate
tournament throughput subject to failure, tail-latency, and per-game speed
safety bounds.

For a successful ``go nodes N`` search, the requested node target is the
stable work-unit measurement.  The last completed iterative-deepening ``info``
line is retained only as diagnostics: it can under-report work when the node
limit interrupts an unfinished next iteration.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import platform
from pathlib import Path
import queue
import re
import statistics
import subprocess
import sys
import threading
import time
from typing import Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENGINE = REPO_ROOT / "target" / "release" / "eureka.exe"
DEFAULT_CONCURRENCY_POINTS = (1, 2, 4, 8, 12, 13)
MIN_RELATIVE_WORKER_SPEED = 0.50
MAX_P95_DURATION_RATIO = 1.35
DEFAULT_FIXTURES = ("open-tactical", "high-branch")
FIXTURES = {
    "open-tactical": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 5",
    "high-branch": "r3k2r/pppb1ppp/2np1n2/2q1p3/3pP3/2NP1N2/PPPQBPPP/R3K2R w KQkq - 0 1",
}
INFO_RE = re.compile(
    r"\bdepth (?P<depth>\d+)\b.*?\bnodes (?P<nodes>\d+)\b"
    r".*?\btime (?P<time>\d+)\b.*?\bnps (?P<nps>\d+)\b"
)


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
        "schema_version": 2,
        "mode": "topology",
        "platform": platform.platform(),
        "logical_processors": logical,
        "physical_cores": physical,
        "recommended_fastchess_concurrency": recommendation,
        "engine_thread_model": "single-threaded",
        "policy": "physical-cores-minus-one",
        "formal_match_started": False,
    }


def _engine_command(engine: Path, profile: str) -> list[str]:
    resolved = engine.expanduser().resolve()
    if resolved.suffix.lower() == ".py":
        return [sys.executable, str(resolved), "--profile", profile]
    return [str(resolved), "--profile", profile]


class _UciReader:
    def __init__(self, stream) -> None:
        self.lines: queue.Queue[str] = queue.Queue()
        self.thread = threading.Thread(target=self._read, args=(stream,), daemon=True)
        self.thread.start()

    def _read(self, stream) -> None:
        for line in stream:
            self.lines.put(line.rstrip("\r\n"))

    def wait_for(self, predicate, timeout_s: float) -> str:
        deadline = time.perf_counter() + timeout_s
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise TimeoutError("timed out waiting for UCI output")
            try:
                line = self.lines.get(timeout=remaining)
            except queue.Empty as exc:
                raise TimeoutError("timed out waiting for UCI output") from exc
            if predicate(line):
                return line


def _parse_info(line: str) -> Optional[dict[str, int]]:
    match = INFO_RE.search(line)
    if not match:
        return None
    return {key: int(value) for key, value in match.groupdict().items()}


def _close_process_streams(process: subprocess.Popen[str]) -> None:
    for stream in (process.stdin, process.stdout, process.stderr):
        if stream is not None:
            stream.close()


def _run_worker(
    engine: Path,
    profile: str,
    fixtures: Sequence[str],
    nodes: int,
    repeat: int,
    warmup: int,
    hash_mb: int,
    timeout_s: float,
) -> dict[str, object]:
    command = _engine_command(engine, profile)
    process: Optional[subprocess.Popen[str]] = None
    stdout_reader: Optional[_UciReader] = None
    started = time.perf_counter()
    try:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert process.stdin is not None
        assert process.stdout is not None
        stdout_reader = _UciReader(process.stdout)

        def send(line: str) -> None:
            assert process is not None and process.stdin is not None
            process.stdin.write(line + "\n")
            process.stdin.flush()

        send("uci")
        stdout_reader.wait_for(lambda line: line == "uciok", timeout_s)
        send(f"setoption name Hash value {hash_mb}")
        send("isready")
        stdout_reader.wait_for(lambda line: line == "readyok", timeout_s)

        def search(fixture: str) -> dict[str, int]:
            # Keep every measured search at the same cold-TT workload. Without
            # this reset, repeated FENs can return from the persistent UCI TT
            # immediately and make the worker NPS incomparable.
            send("ucinewgame")
            send(f"position fen {FIXTURES[fixture]}")
            send(f"go nodes {nodes}")
            latest: Optional[dict[str, int]] = None
            deadline = time.perf_counter() + timeout_s
            while True:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    raise TimeoutError("timed out waiting for bestmove")
                line = stdout_reader.lines.get(timeout=remaining)
                parsed = _parse_info(line) if line.startswith("info ") else None
                if parsed is not None:
                    latest = parsed
                if line.startswith("bestmove "):
                    if latest is None or line.split(maxsplit=1)[1] == "0000":
                        raise RuntimeError("engine returned no usable measured search result")
                    return latest

        for _ in range(warmup):
            for fixture in fixtures:
                search(fixture)

        measured_started = time.perf_counter()
        measured: list[dict[str, int]] = []
        for _ in range(repeat):
            for fixture in fixtures:
                measured.append(search(fixture))
        measured_finished = time.perf_counter()
        duration_ms = (measured_finished - measured_started) * 1000.0
        # Fastchess workload is defined by completed fixed-node searches.  The
        # final info line is not a reliable work counter because the node cap
        # can land inside an incomplete next iteration.
        work_nodes = nodes * len(measured)
        worker_nps = work_nodes * 1000.0 / max(duration_ms, 0.001)
        send("quit")
        process.wait(timeout=timeout_s)
        if process.returncode != 0:
            raise RuntimeError(f"engine exited with code {process.returncode}")
        _close_process_streams(process)
        return {
            "status": "PASS",
            "duration_ms": duration_ms,
            "measurement_started_s": measured_started,
            "measurement_finished_s": measured_finished,
            "work_nodes": work_nodes,
            "reported_info_nodes": [item["nodes"] for item in measured],
            "searches_completed": len(measured),
            "worker_nps": worker_nps,
            "depths": [item["depth"] for item in measured],
            "think_times_ms": [item["time"] for item in measured],
        }
    except Exception as exc:
        if process is not None:
            process.kill()
            process.wait()
            _close_process_streams(process)
        return {
            "status": "FAIL",
            "duration_ms": (time.perf_counter() - started) * 1000.0,
            "measurement_started_s": None,
            "measurement_finished_s": None,
            "work_nodes": 0,
            "reported_info_nodes": [],
            "searches_completed": 0,
            "worker_nps": 0.0,
            "error": str(exc),
        }


def _p95(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(0.95 * len(ordered) + 0.999999) - 1))
    return ordered[index]


def _run_point(
    engine: Path,
    profile: str,
    concurrency: int,
    fixtures: Sequence[str],
    nodes: int,
    repeat: int,
    warmup: int,
    hash_mb: int,
    timeout_s: float,
) -> dict[str, object]:
    started = time.perf_counter()
    workers: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                _run_worker,
                engine,
                profile,
                fixtures,
                nodes,
                repeat,
                warmup,
                hash_mb,
                timeout_s,
            )
            for _ in range(concurrency)
        ]
        for future in as_completed(futures):
            workers.append(future.result())
    batch_duration_ms = (time.perf_counter() - started) * 1000.0
    completed = [worker for worker in workers if worker["status"] == "PASS"]
    durations = [float(worker["duration_ms"]) for worker in completed]
    worker_nps = [float(worker["worker_nps"]) for worker in completed]
    total_work_nodes = sum(int(worker["work_nodes"]) for worker in completed)
    measurement_starts = [float(worker["measurement_started_s"]) for worker in completed]
    measurement_finishes = [float(worker["measurement_finished_s"]) for worker in completed]
    measurement_duration_ms = (
        max(measurement_finishes) - min(measurement_starts)
    ) * 1000.0 if measurement_starts else 0.0
    aggregate_nps = total_work_nodes * 1000.0 / max(measurement_duration_ms, 0.001)
    nodes_target_per_worker = nodes * len(fixtures) * repeat
    return {
        "concurrency": concurrency,
        "workers_completed": len(completed),
        "workers_failed": concurrency - len(completed),
        "median_worker_nps": statistics.median(worker_nps) if worker_nps else 0.0,
        "aggregate_nps": aggregate_nps,
        "aggregate_work_nodes": total_work_nodes,
        "measurement_duration_ms": measurement_duration_ms,
        "median_duration_ms": statistics.median(durations) if durations else 0.0,
        "p95_duration_ms": _p95(durations),
        "worker_results": workers,
        "batch_duration_ms": batch_duration_ms,
        "nodes_target_per_worker": nodes_target_per_worker,
        "fixtures": list(fixtures),
        "repeat": repeat,
        "warmup": warmup,
    }


def recommend_concurrency(points: Sequence[dict[str, object]]) -> tuple[int, float]:
    baseline_point = next((point for point in points if point["concurrency"] == 1), None)
    if baseline_point is None or baseline_point["workers_failed"] != 0:
        return 1, 0.0
    baseline_worker_speed = float(baseline_point["median_worker_nps"])
    baseline_aggregate_nps = float(baseline_point["aggregate_nps"])
    if baseline_worker_speed <= 0 or baseline_aggregate_nps <= 0:
        return 1, 0.0

    for point in points:
        point["relative_worker_speed"] = (
            float(point["median_worker_nps"]) / baseline_worker_speed
        )
        point["aggregate_throughput_ratio"] = (
            float(point["aggregate_nps"]) / baseline_aggregate_nps
        )
        median_duration = float(point["median_duration_ms"])
        p95_duration = float(point["p95_duration_ms"])
        duration_ratio = p95_duration / median_duration if median_duration > 0 else float("inf")
        point["duration_ratio_p95_median"] = duration_ratio
        point["eligible"] = (
            point["workers_failed"] == 0
            and point["relative_worker_speed"] >= MIN_RELATIVE_WORKER_SPEED
            and duration_ratio <= MAX_P95_DURATION_RATIO
        )
    eligible = [point for point in points if point.get("eligible")]
    winner = max(
        eligible,
        key=lambda point: (
            float(point["aggregate_throughput_ratio"]),
            int(point["concurrency"]),
        ),
        default=None,
    )
    recommendation = int(winner["concurrency"]) if winner is not None else 1
    return recommendation, baseline_worker_speed


def empirical_probe(
    engine: Path,
    profile: str = "current",
    concurrency_points: Sequence[int] = DEFAULT_CONCURRENCY_POINTS,
    fixtures: Sequence[str] = DEFAULT_FIXTURES,
    nodes: int = 1_000_000,
    repeat: int = 3,
    warmup: int = 1,
    hash_mb: int = 16,
    timeout_s: float = 300.0,
) -> dict[str, object]:
    unknown = [fixture for fixture in fixtures if fixture not in FIXTURES]
    if unknown:
        raise ValueError(f"unknown empirical fixture(s): {', '.join(unknown)}")
    points = [
        _run_point(
            engine,
            profile,
            concurrency,
            fixtures,
            nodes,
            repeat,
            warmup,
            hash_mb,
            timeout_s,
        )
        for concurrency in concurrency_points
    ]
    recommendation, baseline_nps = recommend_concurrency(points)
    return {
        "schema_version": 3,
        "mode": "empirical",
        "engine": str(engine.expanduser().resolve()),
        "profile": profile,
        "engine_thread_model": "single-threaded",
        "nodes": nodes,
        "repeat": repeat,
        "warmup": warmup,
        "hash_mb": hash_mb,
        "concurrency_points": list(concurrency_points),
        "baseline_median_worker_nps": baseline_nps,
        "baseline_aggregate_nps": float(
            next(point["aggregate_nps"] for point in points if point["concurrency"] == 1)
        ),
        "recommended_fastchess_concurrency": recommendation,
        "selection_policy": {
            "workers_failed": 0,
            "relative_worker_speed_min": MIN_RELATIVE_WORKER_SPEED,
            "p95_duration_over_median_max": MAX_P95_DURATION_RATIO,
            "objective": "max_aggregate_throughput_ratio",
            "work_units": "completed_fixed_node_search_targets",
        },
        "points": points,
        "formal_match_started": False,
    }


def _parse_int_list(value: str) -> tuple[int, ...]:
    points = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not points or any(point < 1 for point in points):
        raise argparse.ArgumentTypeError("concurrency points must be positive integers")
    return points


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    modes = result.add_mutually_exclusive_group()
    modes.add_argument("--topology", action="store_true")
    modes.add_argument("--empirical", action="store_true")
    result.add_argument("--json", action="store_true", dest="as_json")
    result.add_argument("--output", type=Path, default=None, help="write JSON report to this path")
    result.add_argument("--engine", type=Path, default=DEFAULT_ENGINE)
    result.add_argument("--profile", default="current")
    result.add_argument("--concurrency-points", type=_parse_int_list, default=DEFAULT_CONCURRENCY_POINTS)
    result.add_argument("--fixtures", default=",".join(DEFAULT_FIXTURES))
    result.add_argument("--nodes", type=int, default=1_000_000)
    result.add_argument("--repeat", type=int, default=3)
    result.add_argument("--warmup", type=int, default=1)
    result.add_argument("--hash-mb", type=int, default=16)
    result.add_argument("--worker-timeout-s", type=float, default=300.0)
    return result


def _print_human(report: dict[str, object]) -> None:
    if report["mode"] == "topology":
        print(f"physical cores: {report['physical_cores']}")
        print(f"logical processors: {report['logical_processors']}")
        print(f"recommended Fastchess concurrency: {report['recommended_fastchess_concurrency']}")
        print("engine thread model: single-threaded")
        return
    print(f"empirical engine: {report['engine']}")
    print(f"profile: {report['profile']}")
    for point in report["points"]:
        print(
            f"concurrency={point['concurrency']} workers="
            f"{point['workers_completed']}/{point['workers_completed'] + point['workers_failed']} "
            f"median_worker_nps={point['median_worker_nps']:.0f} "
            f"aggregate_nps={point['aggregate_nps']:.0f} "
            f"relative_worker_speed={point.get('relative_worker_speed', 0.0):.3f} "
            f"aggregate_ratio={point.get('aggregate_throughput_ratio', 0.0):.3f} "
            f"p95/median={point.get('duration_ratio_p95_median', 0.0):.3f} "
            f"eligible={point.get('eligible', False)}"
        )
    print(f"recommended Fastchess concurrency: {report['recommended_fastchess_concurrency']}")
    print("formal match started: false")


def main() -> int:
    args = parser().parse_args()
    empirical = args.empirical
    if empirical:
        engine = args.engine.expanduser().resolve()
        if not engine.is_file():
            print(f"empirical probe error: engine does not exist: {engine}", file=sys.stderr)
            return 2
        report = empirical_probe(
            engine,
            profile=args.profile,
            concurrency_points=args.concurrency_points,
            fixtures=tuple(item.strip() for item in args.fixtures.split(",") if item.strip()),
            nodes=args.nodes,
            repeat=args.repeat,
            warmup=args.warmup,
            hash_mb=args.hash_mb,
            timeout_s=args.worker_timeout_s,
        )
    else:
        report = probe()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_human(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
