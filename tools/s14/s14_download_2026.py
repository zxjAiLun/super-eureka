#!/usr/bin/env python3
"""S14: download the 2026 Fishtest LTC PGNs (official-stockfish/fishtest_pgns).

Network policy (user requirement):
  - hf-mirror.com ONLY, with an explicit NO-PROXY opener — huggingface.co
    is unreachable without a proxy on this network, and the proxy (clash)
    has metered traffic that must NOT be consumed. urllib otherwise honors
    HTTP_PROXY/HTTPS_PROXY env vars; we build every request through a
    ProxyHandler({}) opener so nothing ever touches clash.
  - Resumable: skips files already on disk with the right size (HEAD first).
  - Bounded worker pool (4 streams), exponential backoff on 429/5xx.
  - Downloads BOTH <test>.pgn.gz and <test>.json per test dir (json =
    metadata for Base/master-side filtering later).
  - Writes a manifest of completed files + sizes for the converter.

Usage:
  python tools/s14/s14_download_2026.py [--workers 4] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = "official-stockfish/fishtest_pgns"
MIRROR = "https://hf-mirror.com"
DIRS_CACHE = Path(
    r"C:\Users\81489\AppData\Local\Temp\opencode\hf-2026-tests.json")
OUT_ROOT = Path(r"data\s14\fishtest-2026")
MANIFEST = OUT_ROOT / "download-manifest.json"
UA = {"User-Agent": "eureka-s14-downloader/1.0"}

# NO-PROXY opener: never touch clash regardless of environment variables.
_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def http(url: str, method: str = "GET", dest: Path | None = None,
         timeout: int = 120):
    req = urllib.request.Request(url, headers=UA, method=method)
    with _opener.open(req, timeout=timeout) as resp:
        if method == "HEAD":
            return int(resp.headers.get("Content-Length", "0")), resp.geturl()
        data = resp.read()
        if dest is not None:
            dest.write_bytes(data)
        return len(data), resp.geturl()


def url_for(rel: str) -> str:
    return f"{MIRROR}/datasets/{REPO}/resolve/main/{rel}"


def remote_size(url: str) -> int | None:
    try:
        n, _ = http(url, method="HEAD", timeout=30)
        return n
    except Exception:
        return None


def fetch_with_backoff(url: str, dest: Path, max_tries: int = 6):
    delay = 2.0
    for i in range(max_tries):
        try:
            n, _ = http(url, dest=dest, timeout=300)
            return n
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue
            raise
        except Exception:
            time.sleep(delay)
            delay = min(delay * 2, 60)
    raise RuntimeError(f"failed after {max_tries}: {url}")


def download_one(test_dir: str):
    tid = test_dir.split("/")[-1]
    day_dir = OUT_ROOT / test_dir
    day_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for ext in (".pgn.gz", ".json"):
        fname = tid + ext
        dest = day_dir / fname
        rel = f"{test_dir}/{fname}"
        url = url_for(rel)
        # resume/skip: verify local size against the mirror's HEAD
        if dest.exists() and dest.stat().st_size > 0:
            rs = remote_size(url)
            if rs is not None and dest.stat().st_size == rs:
                results[ext] = {
                    "status": "exists", "bytes": dest.stat().st_size}
                continue
            dest.unlink()
        try:
            n = fetch_with_backoff(url, dest)
            results[ext] = {"status": "ok", "bytes": n}
        except Exception as e:
            results[ext] = {"status": "failed", "error": str(e)[:200]}
            if dest.exists():
                dest.unlink()
    return test_dir, results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0,
                    help="download only the first N test dirs (smoke)")
    args = ap.parse_args()

    tests_by_day = json.load(open(DIRS_CACHE, encoding="utf-8"))
    all_tests = [t for subs in tests_by_day.values() for t in subs]
    if args.limit:
        all_tests = all_tests[: args.limit]
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    manifest = {}
    if MANIFEST.exists():
        manifest = json.load(open(MANIFEST, encoding="utf-8"))

    done = fail = 0
    t0 = time.time()
    total_bytes = 0
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(download_one, t): t for t in all_tests}
        for fut in as_completed(futs):
            test_dir, results = fut.result()
            ok = all(v["status"] in ("ok", "exists")
                     for v in results.values())
            total_bytes += sum(v.get("bytes", 0) for v in results.values())
            manifest[test_dir] = results
            done += 1
            if not ok:
                fail += 1
            if done % 10 == 0 or done == len(all_tests):
                el = time.time() - t0
                print(f"{done}/{len(all_tests)} tests, "
                      f"{total_bytes/2**30:.2f} GiB, "
                      f"{total_bytes/max(el,1)/2**20:.1f} MiB/s, "
                      f"fail={fail}", flush=True)
                MANIFEST.write_text(json.dumps(manifest, indent=1),
                                    encoding="utf-8")
    MANIFEST.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    failed = [k for k, v in manifest.items()
              if any(r["status"] == "failed" for r in v.values())]
    print(f"DONE: {done} tests, {fail} with failures, "
          f"{total_bytes/2**30:.2f} GiB in {(time.time()-t0)/60:.0f} min")
    if failed:
        print("failed dirs (first 10):", failed[:10])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

