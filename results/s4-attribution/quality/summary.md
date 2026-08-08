# S4.0B Search-Quality Attribution summary

Diagnostic only; no S4.1 candidate implemented.

- pgns: S3-PROMOTION (106) + S3-FINAL (100) = 206 CurrentFinal games
  (CurrentFinal vs Current, 10+0.1)
- sampled positions: 700 (after ply 12, every ~5 plies, non-terminal, dedup)
- teacher: Stockfish-18, MultiPV=3, 60k nodes/position
- normal CurrentFinal: 100 / 500 / 1000 ms
- disagreement threshold: >= 50 cp teacher loss (mate handled separately)
- high-confidence disagreements: 105

## Classification

- SEARCH_SUSPECT: 23
- EVAL_SUSPECT: 45
- HORIZON_SUSPECT: 5
- UNRESOLVED: 32

EVAL_SUSPECT is the largest and most consistent class: for these, forcing the
teacher (quiet) move at equal depth 3..7 still leaves it statically inferior to
CurrentFinal's own line (e.g. flat `cp:-55` across depths). SEARCH_SUSPECT (23)
is where forcing the teacher move recovers a >= line at 1000 ms, so CurrentFinal
*can* recognize it once adequately searched.

## Teacher root rank (disagreements, forced_1000)

- median rank: 12; rank >= 8: 77 / 105 (73%)
- by class: SEARCH 11 (14/23 >= 8), EVAL 13 (34/45 >= 8), HORIZON 9, UNRESOLVED 13
- quiet teacher moves: n=87, median rank 13, 70/87 (80%) rank >= 8

Teacher quiet moves frequently rank 10+. Combined with 83% of disagreements
being quiet moves, this is a strong ordering + quiet-move-recognition signal:
good quiet moves are tried too late and get weakened by LMR/pruning.

## Teacher move type (disagreements)

- quiet: 87, capture: 11, check: 7  (83% quiet)

## Forced-root findings

- SEARCH_SUSPECT (23): forced teacher move >= normal line at equal time ->
  a real search-allocation/ordering component.
- EVAL_SUSPECT (45): forced teacher move stays flat-inferior across depth 3..7 ->
  a real static-evaluation component for quiet positional moves.
- HORIZON_SUSPECT (5): clear convergence with depth (e.g. 708 -> 1191 cp).

## Ablation (targeted, 10 representative suspects)

Forced teacher move was already chosen in every ablation (teacher_move_chosen all
True), so ablations mostly confirm the move is searchable once forced; they do
not by themselves argue for removing LMR/futility/null/qSEE globally. The primary
cost is that these quiet moves are reached too late / undervalued before forcing.

## Combined S4.0A + S4.0B interpretation

- S4.0A: per-node cost is concentrated in legal-move make/unmake legality
  filtering (~114-122/node); completed depth is bound by tree size (qsearch
  ~80% of nodes, iteration growth ~4.6); LMR re-search is low; TT under-utilized.
- S4.0B: CurrentFinal's concrete failures are overwhelmingly QUIET positional
  moves (83%) that are ranked late (median 12-13) and, when searched at equal
  depth, are still statically undervalued (EVAL_SUSPECT is the largest class).

## Top 3 plausible bottlenecks

1. Quiet-move ordering (root + history): teacher quiet moves rank median 12-13,
   73-80% at rank >= 8 -> they are LMR-reduced/pruned before being recognized.
2. Static evaluation of quiet positional moves: EVAL_SUSPECT (45) - forced quiet
   teacher moves remain inferior at depth 3..7, i.e. the evaluator under-values
   positional/strategic compensation (king safety, mobility, activity, pawns).
3. Core node cost: legal-move make/unmake filtering (~120/node) limits NPS/depth.

## Recommended S4.1 candidate

**Evaluation improvement for quiet positional moves.**

Rationale: EVAL_SUSPECT is the largest, most consistent class, and it is
isolated for search depth (forced teacher move at equal depth 3..7 stays
inferior). This points at the static evaluator rather than a search-mechanism
bug. The quiet-heavy, late-ranked profile also implicates ordering, so quiet-move
ordering (root/history) is the strongest secondary candidate and likely
interacts (late rank -> LMR/prune -> less depth -> eval sees it as bad).

Per S4.0B decision rules, the evidence (systematic forced-root misevaluation of
quiet positional moves, not merely "missed by normal search") advances
**Evaluation** ahead of a MovePicker/bitboard Core rewrite. S4.1 should add a
small, SPRT-gated evaluator term (or set of terms) targeting the quiet
positional features CurrentFinal currently lacks; it must not be a blind
multi-term stack, and must be evaluated as a single attributable candidate.

## Stop point

S4.0B complete. No S4.1 candidate has been implemented.
