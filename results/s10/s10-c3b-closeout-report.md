# S10-C3-B Closeout — Three-Arm Search NPS Gate

**Status: CLOSED / PASS (measurement) — decision gate: bounded runtime
optimization before Arena**

## Frozen protocol

```
arms:       A = current-final (Eval2)
            B = current-final-nnue-v2q-full
            C = current-final-nnue-v2q (incremental)
corpus:     24 balanced FENs (4 opening / 8 middlegame / 4 tactical /
            4 endgame / 2 king-heavy / 2 special), in-file in
            tools/s10/search_nps_c3b.py (corpus SHA in artifact)
stop:       fixed 200,000 nodes per fixture/arm/round (never wall-clock)
TT:         fresh cold 64 MiB per run (--hash-mb 64); Threads = 1
rounds:     12 per fixture, 6-order Latin rotation x2 (ABC ACB BAC BCA
            CAB CBA), rotated per fixture
diagnostics OFF; model b51a79b1... verified per run
B/C tree identity fail-closed per fixture-round (nodes, qsearch_nodes,
score, bestmove, PV)
harness:    tools/s10/search_nps_c3b.py
artifact:   results/s10/s10-c3b-search-nps.json
```

## Headline (paired per-fixture-round ratios, n = 288 each)

```
                    median    p25      p75      min      max
B/A (full/Eval2)    0.7001   0.6412   0.7245   ...      ...
C/A (incr/Eval2)    0.8200   0.6906   0.8619   0.5206   0.9397
C/B (incr/full)     1.1633   1.0380   1.1896   0.9711   1.8902

raw NPS medians:    A 272,934 | B 193,569 | C 235,232

B/C tree identity:  288/288 checks, 0 failures
node budget:        exactly 200,000 consumed in every arm/run
```

## Interpretation (against the frozen gate)

```
C/B = 1.163  →  the incremental accumulator stack delivers a REAL
                +16% search speedup over full-refresh NNUE. C2A/C2B's
                move-aware design pays off exactly as intended.
C/A = 0.820  →  < 0.85 threshold → decision: bounded_optimization_first
```

The NNUE evaluator itself is currently ~22% slower than production Eval2
in real search — consistent with C3-A's microcost finding that the dense
forward (~2.8 us, 63% of eval cost) dominates. The accumulator/update
machinery is NOT the problem (incremental beats full refresh by 16%).

## What this means for the next step (bounded optimization)

Per the C3-A hotspot ranking, the one candidate cut is the dense forward
MAC cost (256x32 + 32x32 + 32x1 integer MACs per eval). Options to
evaluate in a bounded pass — no model/data/training changes, exact-output
preservation required (bit-exact vs current integer reference):

```
1. memory layout / cache blocking of l1 weights
2. i32 x i16 widening-MAC ordering (compiler auto-vectorization hints)
3. reduced accumulator-precision dense input (would change outputs —
   REJECTED unless re-gated through B5-style numerical gates)
```

Alternatively the Arena question can be revisited if +22% eval cost is
deemed acceptable for a possible eval-strength gain — but the frozen gate
says optimize first.

## Corpus note

`tactical-2` originally repeated the C2B mate-position FEN (terminal
root); replaced with a live tactical middlegame before any measurements
were recorded. All 288 recorded paired samples are from the final
corpus.
