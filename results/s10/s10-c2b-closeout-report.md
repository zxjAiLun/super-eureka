# S10-C2B Closeout — Search-Stack Integration

**Status: CLOSED / PASS**

## Question answered

> With the quantized NNUE accumulator traveling through negamax / qsearch /
> PVS / LMR / null-move / aspiration / abort-unwind, does the incremental
> profile produce EXACTLY the same search as the full-refresh profile?

**Yes — 18 fixtures, 0 mismatches, zero tolerance.**

## Profiles (production untouched)

```
current-final-nnue-v2q-full         CurrentFinalNnueV2QFull
current-final-nnue-v2q               CurrentFinalNnueV2QIncremental
```

Both inherit CurrentFinal policy bit-for-bit (enforced by the extended
anti-drift test `s10c2b_nnue_profiles_inherit_current_final_policy` incl.
the "exactly one evaluator" rule; PRODUCTION_PROFILE still CurrentFinal).
UCI keeps `None` (fail-closed for any future NNUE profile without a model);
bench requires `--nnue-model` for NNUE profiles (fail-closed).

## Architecture

```
NnueSearchState (src/engine/nnue_search.rs, new):
    model: Arc<NnueV2QuantizedModel>   loaded ONCE per run (never per node/go)
    mode:  FullRefresh | Incremental
    frames: Vec<NnueV2Accumulator>     search-local; NEVER in Position/Undo

Edge contract (all four real edges: negamax move loop, qsearch move loop,
root loop, final-evasion ply — one code shape each):
    prepare delta BEFORE make -> make_move_profiled -> path.push_child
    -> NNUE push child AFTER make -> probe/recurse (scout + re-search REUSE
    the same top frame; exactly one push) -> NNUE pop -> path.pop
    -> unmake_move_profiled

Null move: child frame is a plain copy (C2A bit-identity invariant); the
dense forward swaps STM/NSTM via the flipped side-to-move. No fake delta.
Abort/unwind: every None arm pops the NNUE frame with the path;
search_best_move_with_history_tt_and_profile additionally runs
state.restore_root() (mirror of path.restore_root) as a safety net.

evaluate_profiled dispatch (classical paths byte-identical):
    NNUE profile -> exact KQK/KRK mop-up law applied to the NNUE base
    (eval::exact_mop_up_for_search, new pub helper mirroring
    finish_evaluation's tail) -> else the frozen quantized network.

Search-eval is INTEGER-ONLY: evaluate_cp_i32_from_accumulator computes
raw * 1000 / 2^12 in i64 with signed round-half-away (no f32 anywhere).
```

## Gate result (results/s10/s10-c2b-search-parity.json)

```
engine:    bf0cc118d159d23400929c1575d92361d7ce50cc7b9ff6c89b517d4e105e9806
model:     b51a79b1... (frozen, verified at run)
corpus:    18 fixtures — startpos/middlegame/tactical x2/endgame/promotion/
           EP/castling/KQK/KRK mop-ups/nullmove-heavy + node-limited
           5000/1000/100/7/1 (abort/unwind)
fields:    score, bestmove, completed_depth, nodes, PV — all EXACT
mismatches: 0                                                     PASS

path coverage (proven exercised):
  qsearch_nodes 395,016 | null_move_attempts 19 | lmr_reductions 187
  aspiration_retries 51 | abort budgets 1/7/100/1000/5000
```

Harness: `tools/s10/parity_search_c2b.py` (fixed limits, no wall-clock,
same binary + same artifact for both arms).

## Tests (cargo 412 pass, +7 C2B)

```
s10c2b_nnue_profiles_inherit_current_final_policy   policy bits + single
                                                    evaluator + not default
s10c2b_nnue_profiles_preserve_exact_mopup           mop-up law identical
                                                    on the NNUE base; KQK/
                                                    KRK trigger, startpos
                                                    passes through the net
s10c2b_nnue_stack_balance_after_abort               budgets 1/7/100/1000 x
                                                    3 FENs: clean return,
                                                    root restored
nnue_search: stack_push_pop_balance_and_restore_root
             null_child_push_is_bit_identical
             incremental_eval_matches_full_refresh_in_stack
             audit_counters_detect_tampered_stack
```

Integration repair recorded for provenance: the initial edge instrumentation
briefly dropped `path.pop()` on the unwind arms (regex error), which
corrupted repetition detection — caught immediately by the existing locked
smoke tests (score drift), root-caused, and fixed; all 405 pre-C2B tests
pass unchanged, and no classical profile code path differs from the
pre-C2B tree (verified by the locked node-count/score/PV baselines).

## Untouched

```
PRODUCTION_PROFILE = CurrentFinal        (UCI default unchanged)
frozen artifact b51a79b1...              (loads; startpos raw 190)
classical evaluators                     (dispatch order byte-identical)
NPS / performance claims                 (C3 scope — deliberately none here)
```
