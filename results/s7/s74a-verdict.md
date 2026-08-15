# S7.4A — LMR-on-Null-Window Candidate Verdict

> **Repair 1 update (`df7f324`) — QUALIFIED_FOR_ARENA, STRONG.**
> Review P1 was: the candidate's full-depth verification re-search was a new
> real search entry but did not call `try_enter_node`. Repair 1 adds exact-once
> acquisition + `s74_lmr_nw_research_entered` accounting and regression tests.
> All gates below were re-run or regenerated with the repaired binary.
> Production code remains accounting-only for unlimited fixed-depth search.

BASE: `6cf45063ef4c8eb2db03069149327d843fe47fba` (implementation commit `d50729a`)
REPAIR: `df7f324` on top of evidence base `ea1c60b`

## Status

- S7.3 observation implementation: APPROVED (data used with repairs)
- S7.3 diagnosis: `SELECTIVITY_TOO_CONSERVATIVE`, precise mechanism
  `LMR suppressed by null-window/PVS coupling`
- S7.4A candidate: `current-final-lmr-null-window`
  (CurrentFinal + apply EXISTING LMR policy to null-window nodes;
  NO threshold/formula changes, no LMP, no null-move changes)

## Mechanism summary (S7.3 repair values)

- `late_move_reduction()` proposed R1 on 232,058 depth>=4 quiets; only
  26,982 were actually reduced (11.6%) because `pvs_child_window()`
  returns `ChildWindow::Full` when the caller is already null-window,
  discarding the reduction.
- S7.4A routes those moves through reduced null-window search with
  full-depth fail-high verification.

## Fixed-depth node gate (80 S7 corpus, cold 16MB, threads 1)

| depth | baseline nodes | candidate nodes | reduction | verdict |
|---|---|---|---|---|
| 6  | 9,941,996  | 5,785,742  | **-41.805%** | STRONG_MECHANISM |
| 7  | 43,104,125 | 21,810,945 | **-49.399%** | STRONG_MECHANISM |
| 8 (20-pos subset) | 50,168,172 | 12,286,278 | **-75.510%** | STRONG_MECHANISM |

Predeclared gate: <= -15% STRONG. All depths qualify.

The d8 subset exercises the existing R2 rule (remaining depth >= 7,
move index >= 8) which S7.3 could not reach at root d6/d7; at d8 the
reduction compounds through the tree (-75.5%) with LMR proposed 143,892 /
applied 134,876 (93.7%).

- LMR proposed at d6: 65,835; applied on null-window: 58,195 (88.4%)
- LMR proposed at d7: 151,971; applied on null-window: 133,011 (87.5%)
- Zero-reduction rows (6/80 at d7) are sparse piece/pawn endgames where
  the existing `non_pawn_material >= 4` guard never proposes LMR —
  identical trees are the intended conservative behavior, not a bug.
- d8 subset (first 20 positions, exercising R2): see below.

## Teacher challenge (S6 corpus, 178 rows / 176 evaluated at depth 6; 2 rows excluded as engine errors, fixed-depth)

- baseline teacher-bestmove matches: 33
- candidate teacher-bestmove matches: **34** (+1)
- cp divergence >= 100 / 300 / 500: **0 / 0 / 0**
- 15 positions with small score differences, all < 100cp
- 3 mate-labelled positions: candidate identical to baseline
  (same mate scores, same bestmoves); no mate transitions
- baseline-only matches: 0; candidate-only matches: 1

## Search stability gate (d6 -> d7 on 80 S7 corpus)

| metric | baseline | candidate |
|---|---|---|
| bestmove changes      | 19 | **15** |
| >=200cp reversals     |  1 | 1 |
| cp <-> mate transitions | 0 | 0 |

Candidate is not less stable; bestmove flips actually decrease.

## Production invariance

- current-final, 30 S4, depth 6, before/after implementation:
  nodes/score/bestmove/PV **30/30 exact match**.
- fmt / clippy -D warnings / debug tests / release tests: clean.

## Fixed-wall depth gate (pre-repair, superseded — missing `try_enter_node` also skipped a deadline check)

| movetime | median depth A→B | mean depth A→B | median seldepth A→B | gained | lost |
|---|---|---|---|---|---|
| 1000 ms | 7 → 7 | 6.775 → 7.362 | 17 → 18 | **44 / 80** | **0** |
| 3000 ms | 7 → **9** | 7.513 → 8.512 | 18 → 19.5 | **62 / 80** | **0** |

Qualification minimum was 10/80 gained with gained > lost; the candidate
gains completed depth on 55% (1s) to 78% (3s) of positions and loses
depth on none. Median completed depth at 3s rises a full two plies.

## Repair 1 re-run evidence (`df7f324`)

### Implementation contract

- Initial reduced null-window search: NO new acquisition (`probe_child_draw`
  already owns that child's initial acquisition).
- Full-depth verification when `nw_reduced > alpha_before_move`: exactly one
  `try_enter_node` before `negamax_entered_impl_with_null_and_extensions`,
  with the production PVS re-search cleanup shape (pop + unmake + return
  `None` on acquisition failure).
- Counters: `s74_lmr_nw_research` (requested) and
  `s74_lmr_nw_research_entered` (entered).

### Mechanism accounting tests (`src/engine/search.rs`)

- A. All-fail-low fixture: `research == 0`, `research_entered == 0` — reduced
  fail-low performs no verification acquisition.
- B. Improving fixture: every request emits exactly one test event, every
  successful acquisition emits exactly one entry event, and unlimited
  fixed-depth `research_entered == research_requested`.
- C. Exact acquisition-failure budget: search returns `None`; path/FEN/root
  state restored; `entered == 0` and `verified_cutoff == 0`; killer/history/
  quiet-reward call counts equal the immediately preceding budget; root PV
  row unchanged (no fake PV commit).
- D. Budget sweep: `nodes <= budget` always, and an abort consumes exactly
  its budget; `entered <= requested` and `verified_cutoff <= entered`.

### Accounting-only identity (`tools/s74a_repair_verify.py`)

Across all 180 pre-repair fixed-depth rows (`ea1c60b` JSON vs repaired JSON):

- score / bestmove / PV: **0 differences**
- qsearch_nodes / seldepth: **0 differences**
- per-row `repaired_nodes - old_candidate_nodes == research_requested`
- totals: `39,885,460 - 39,882,965 = 2,495 == sum(research_requested)`
- `research_entered == research_requested` on every unlimited row

### Fixed-depth node gate from scratch (80 S7 d6/d7 + 20 subset d8, cold 16MB)

| depth | baseline nodes | candidate nodes | reduction | main A→B | qsearch A→B | wall A→B (s) | NPS A→B | seldepth median A→B | requested / entered / verified cutoff |
|---|---|---|---|---|---|---|---|---|---|
| 6 | 9,941,996 | 5,786,086 | **-41.802%** | 1.65M→0.84M | 8.29M→4.95M | 41.19→22.75 | 241k→254k | 16→15.5 | 344 / 344 / 100 |
| 7 | 43,104,125 | 21,812,127 | **-49.397%** | 5.82M→3.17M | 37.28M→18.64M | 168.39→81.06 | 256k→269k | 18→17.5 | 1,182 / 1,182 / 341 |
| 8 (20) | 50,168,172 | 12,287,247 | **-75.508%** | 9.64M→1.89M | 40.53M→10.40M | 238.48→56.37 | 210k→218k | 25→23 | 969 / 969 / 129 |

Verdict by original gate: all three depths **STRONG**.

### Fixed-wall depth gate from scratch (80 S7, fresh process, rotated/interleaved A/B)

| movetime | median depth A→B | mean depth A→B | median seldepth A→B | gained | lost |
|---|---|---|---|---|---|
| 1000 ms | 6 → **7** | 6.412 → 6.850 | 16.5 → 16.5 | **35 / 80** | 1 |
| 3000 ms | 7 → **8** | 7.138 → 8.088 | 18 → 19 | **60 / 80** | 0 |

Qualification unchanged: >=10 gained, gained > lost, no material regression.
Both time controls pass clearly.

> Host note: the rerun landed while the machine was under an unrelated
> high-CPU training load. A/B order is rotated/interleaved and both arms
> share the same load, so the relative depth gate remains valid; absolute
> depths are conservative. The single 1000ms loss (`s7_mid_25`, 6→5) is
> 1/80 and does not approach material regression.

### Teacher challenge (d6, regenerated with repaired binary)

- evaluable rows: **176**; terminal/no-move rows 158 and 164 labelled
  `TERMINAL/NOT_APPLICABLE`; genuine engine failures **0**
- teacher bestmove matches: baseline 33 → candidate **34**
- baseline-only / candidate-only: **0 / 1**
- >=100 / 300 / 500 cp divergence: **0 / 0 / 0**
- teacher_mate-labelled positions: **5**; mate transitions /
  mate-distance changes: **0 / 0**

### R2 tactical/horizon gate (d8, new frozen corpus)

- corpus: `data/s7/s74a-r2-tactical-corpus.jsonl`, 120 positions
- SHA256: `7eeecf0ef79501cc28c80385b787dae361e365e32210636be0f4c261c4ee3337`
- composition: 37 S6 teacher-challenge rows + 83 S6 eval-shard rows;
  mate-labelled **11** (all five teacher_mate-labelled challenge rows included)
- completed: **120 / 120** (mate-labelled 11 / 11); timeouts/errors **0**
- teacher bestmove agreement: A 58 / B 59
- baseline-only / candidate-only: **2 / 3**
- >=100 / 300 / 500 cp divergence: **0 / 0 / 0**
- mate detection: A 8 / B 8; mate side mismatches **0**
- cp<->mate transitions **0**; mate-distance changes **0**
- hard-reject scan (baseline finds teacher-signed mate, candidate loses it):
  **0**

### Production invariance (base `ea1c60b` vs repaired `df7f324`)

- CurrentFinal, 30 S4 positions, depth 6: nodes / score / bestmove / PV
  **30/30 exact**.

### Build hygiene

- fmt: clean
- clippy `--all-targets -- -D warnings`: clean
- `cargo test`: **314/314** debug, **314/314** release

## Verdict

**QUALIFIED_FOR_ARENA — STRONG.**

1. Fixed-depth node reduction: **-41.8% (d6) / -49.4% (d7) / -75.5% (d8 subset)** — far beyond the 15% STRONG threshold.
2. Fixed-wall depth evidence: clearly positive (1000ms 35/80 gained, 1 lost; 3000ms 60/80 gained, 0 lost; median depth 6→7 and 7→8).
3. Teacher challenge: not worse — matches 33 → 34, zero >=100cp divergences, all 5 teacher_mate-labelled positions unchanged.
4. No mate/tactical regression: R2 d8 120/120 complete, zero cp divergence, zero mate transitions/side mismatches/distance changes, zero hard-reject cases.
5. Production CurrentFinal unchanged: 30/30 S4 depth-6 exact (nodes/score/bestmove/PV).
6. fmt / clippy -D warnings / debug + release tests: clean.

Per the predeclared contract: STOP after offline evidence. No promotion,
no LMR threshold tuning, no LMP, no null-move changes, no forcing
extensions, no Arena run before review.
