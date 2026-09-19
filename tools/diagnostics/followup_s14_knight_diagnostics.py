"""A bounded follow-up declared after the first ladder: resolve g187's d8 cap
and tighten the already-observed one-ply quiet-mate boundary. No engine edits.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

from run_s14_knight_diagnostics import search, sha

JOBS = [
    (127, "after_Qf3_check", "go depth 1 movetime 5000", "boundary-d1-repeat"),
    (127, "after_Qf3_check", "go depth 2 movetime 5000", "boundary-d2-repeat"),
    (187, "root", "go depth 8 movetime 30000", "root-d8-30s-cap"),
    (187, "after_Nd5", "go depth 7 movetime 15000", "after-Nd5-d7"),
    (187, "after_Nd5", "go depth 8 movetime 15000", "after-Nd5-d8"),
    (187, "after_exd5_black", "go depth 7 movetime 15000", "after-exd5-d7"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    plan = json.loads(args.plan.read_text())
    assert sha(args.engine) == plan['binary_sha256']
    assert sha(args.model) == plan['model_sha256']
    os.nice(10)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/'jobs.json').write_text(json.dumps({
        'plan_sha256': sha(args.plan), 'driver_sha256': sha(__file__),
        'runner_sha256': sha(Path(__file__).with_name('run_s14_knight_diagnostics.py')),
        'binary_sha256': sha(args.engine), 'model_sha256': sha(args.model),
        'niceness': os.nice(0), 'jobs': JOBS}, indent=2))
    results = []
    for number, point_name, go, label in JOBS:
        case = next(c for c in plan['cases'] if c['game_number'] == number)
        point = case['root'] if point_name == 'root' else next(p for p in case['branches'] if p['name'] == point_name)
        # Runner timeout parameter is explicit; no larger engine budget than
        # the job's own 30s or 15s movetime cap is permitted.
        result = search(args.engine, args.model, plan, case, point, go, args.out,
                        f'g{number}-{label}', wait_seconds=40)
        results.append(result)
        with (args.out/'results.jsonl').open('a') as f:
            f.write(json.dumps(result)+'\n')
        print(result['name'], result['bestmove'], result['last'], flush=True)
    (args.out/'complete.json').write_text(json.dumps({'status':'complete','searches':len(results)}))


if __name__ == '__main__':
    main()
