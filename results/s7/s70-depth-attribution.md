# S7.0 — Search Depth Attribution

ATTRIBUTION ONLY — no optimization candidate.

## Contract

```text
observation source:  d71c3e7 (telemetry/build only)
chess baseline:       Eureka v0.1.0
profile:              current-final
TT:                   16 MB cold per depth | threads 1
depths:               [4, 5, 6, 7, 8]
depth-9 wall cap:     10.0 s | per-run timeout 180.0 s
corpus SHA-256:       8786ffca6c8e6277b711c990bf9788d88eaedbb0b4b894f85fc2b18de62d5b1b
positions:            80
```

## Aggregate per depth

| depth | completed | median iter nodes | growth | median wall ms | seldepth-depth | qsearch % | 1st-move cutoff % | searched moves/main-node | TT hit % | LMR research % | bestmove changes | big swings (>=200cp) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 80 | 5,770 | 0.00 | 24 | 8 | 84.7 | 82.9 | 3.21 | 28.9 | 0.0 | 0 | 0 |
| 5 | 80 | 15,044 | 2.61 | 89 | 8 | 86.6 | 86.0 | 4.68 | 20.0 | 0.0 | 16 | 2 |
| 6 | 80 | 83,276 | 5.54 | 424 | 10 | 81.3 | 83.1 | 3.02 | 24.2 | 0.0 | 19 | 2 |
| 7 | 80 | 269,555 | 3.24 | 1,543 | 11 | 84.1 | 85.9 | 4.25 | 15.7 | 1.0 | 19 | 1 |
| 8 | 80 | 1,270,812 | 4.71 | 6,931 | 13 | 78.4 | 84.5 | 2.72 | 11.1 | 1.3 | 15 | 0 |

## Beta-cutoff mover split

| depth | TT move | capture/promotion | killer | quiet |
|---|---|---|---|---|
| 4 | 23.8 | 44.4 | 29.3 | 2.5 |
| 5 | 42.4 | 35.8 | 19.5 | 2.3 |
| 6 | 15.7 | 51.0 | 31.8 | 1.5 |
| 7 | 32.3 | 43.1 | 23.2 | 1.4 |
| 8 | 9.8 | 55.4 | 33.6 | 1.1 |

## Diagnosis — top-3 depth bottlenecks

### 1. QSEARCH_DOMINATED (primary)

Quiescence consumes **78.4% of all nodes at depth 8** (84.7% at depth 4), while the seldepth-depth gap grows from 8 to 13 plies. The main tree is only ~15-20% of the search; the engine spends most of its node budget resolving capture chains and check extensions in quiescence. This is the direct answer to "why depth 7 and not 10-12": each nominal ply drags a deep qsearch tail behind it.

### 2. ORDERING_LIMITED (secondary)

First-move beta-cutoff is only **84.5%** (a strong engine reaches ~90%+), and the cutoff mover split shows captures/pruning dominate (55.4% tactical) with quiet moves rarely cutting off (1.1%). Killer cutoffs are healthy (33.6%) but history/quiet ordering is weak, so late quiet moves still get searched. Median searched branching is ~2.7 moves/main-node with a heavy 17+ tail.

### 3. HIGH_EFFECTIVE_BRANCHING (consequence)

Iteration growth factors are 2.61 / 5.54 / 3.24 / 4.71 (depth 5..8). A well-tuned engine is ~2.0. This is the combined symptom of qsearch dominance and sub-90% move ordering, not an independent cause.

### NOT the bottleneck (yet)

- **LMR** is barely active: only ~3% of reduced searches re-search, and the reduction count is tiny relative to node count (LMR requires depth>=4 + quiet + late index).
- **TT** shows low hit rates only because each depth is measured cold; this is a measurement artifact of the cold-per-depth contract, not a live-search finding.
- **Eval instability** is present (bestmove changes at every depth) but not yet dominant.

### S7.1 direction

Attack quiescence first (SEE/delta pruning on capture chains, cap/qsearch movegen), and only then move ordering (history). Do NOT start with LMR/TT/aspiration.
