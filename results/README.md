# Results artifact policy

Keep **conclusions**, not execution debris, in Git:

- Final reports (including explicit invalidation of superseded experiments).
- Small evidence sufficient to audit the conclusion: paired outcomes, final
  search observations, cost readings, identity hashes and stated limitations.
- Reusable runners belong in `tools/`, with lightweight checks in `tests/`.

Do not commit model/checkpoint binaries, full PGNs, stdout/stderr, PID files,
training caches, downloaded corpora, packaging directories, copied opening-book
slices or one-off queue wrappers. Keep reusable large inputs local and ignored;
delete obsolete scratch files only when their final evidence is preserved.

The existing `/results/*` rule makes new run directories opt-in. Add a narrow
allowlist for the final report/evidence in `.gitignore`, not the whole run.
Check visibility with `git check-ignore --no-index <path>`: a report should not
be ignored; model/log/PGN/PID paths should be ignored. Already-tracked historical
files are not removed by ignore rules; this closeout does not rewrite history.

## Current reference artifacts

- `s15/`, `s15-long/`, `s16/`, `s17/`: old HCE-only screens, explicitly
  `INVALID_EVALUATOR_SELECTION`; original numbers retained, not model ratings.
- `s15short-valid/`, `s15long-valid/`, `s16-valid/`, `s17-valid/`: final replacement
  reports, with hashes and offline WDL/color-pair checks against local PGNs.
- `s18-ab/qc1-match-report.json`: QC1 KILLED, 99/42/115; cutechess Elo/CI only.
- `s18-ab/qc1-pairs.json`: 128 color-swapped opening outcomes, no full move text.
- `s18-ab/qc1-cost-evidence.json`: 96 paired readings (32 positions ×3), plus
  compact boundary/tactical observations; retains rounding/quantile limitations.

No production switch, model change or new search heuristic is authorized by
retaining an experiment's evidence or runner.
