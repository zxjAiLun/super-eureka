# S4.0B Search-Quality Attribution summary (paired fixed-depth repair)

Diagnostic only; no S4.1 candidate implemented.

Corpus terminology: CurrentFinal-vs-teacher disagreement positions, sampled
from CurrentFinal match games (not necessarily 'positions CurrentFinal misplayed').
- pgns: [{"path": "results\\s3-promotion\\run-001\\match.pgn", "games": 106, "pairs": [["Current", "CurrentFinal"], ["CurrentFinal", "Current"]], "results": {"1/2-1/2": 16, "0-1": 45, "1-0": 45}}, {"path": "results\\s3-final\\match\\match.pgn", "games": 100, "pairs": [["Current", "CurrentFinal"], ["CurrentFinal", "Current"]], "results": {"1-0": 45, "0-1": 36, "1/2-1/2": 19}}]
- sampled positions: 700
- teacher: Stockfish-18, MultiPV=3 coherent snapshot, nodes=60000
- normal CurrentFinal: [100, 500, 1000] ms
- disagreement threshold (cp): 50
- high-confidence disagreements (coherent teacher): 113

## Paired fixed-depth classification (T=teacher best, O=CurrentFinal normal best)

For every disagreement we force T AND O at the same depths 3..7 and classify on
delta[d] = score_T[d] - score_O[d] (both CurrentFinal scores, one scale).

- SEARCH_LIKE: 82
- EVAL_LIKE: 11
- HORIZON: 9
- UNRESOLVED: 11

## SEARCH_LIKE (root candidate ordering) - dominant

Teacher move is competitive at matched depth but normal search fails to choose it.
- n=82; median initial root rank=13.5; rank>=8: 62/82

This is evidence of WEAK INITIAL ROOT CANDIDATE ORDERING (CurrentFinal's root uses
movegen order + TT hash lift + previous-iteration best-move swap; it does NOT apply
the MVV-LVA/killer/history ordering used at non-root nodes). Quiet teacher moves are
therefore tried late and get too little search budget.

## EVAL_LIKE (static evaluation) - small, clean

median delta (T-O) per depth: d3=-80, d4=-83, d5=-79, d6=-79, d7=-88
- still <= -40cp below O at d7: 11/11
- teacher move type: all quiet

## HORIZON / UNRESOLVED
- HORIZON: 9 (clear convergence with depth)
- UNRESOLVED: 11 (noisy / mate-incompatible)

## Teacher root rank (all disagreements)
- median: 13; rank>=8: 83/113
- histogram: {"1": 10, "2": 4, "3": 6, "4": 1, "5": 3, "6": 4, "7": 2, "8": 4, "9": 6, "10": 7, "11": 6, "12": 3, "13": 5, "14": 4, "15": 3, "16": 6, "17": 4, "18": 2, "19": 1, "20": 3, "21": 4, "22": 2, "24": 5, "25": 2, "26": 2, "27": 1, "28": 1, "29": 1, "30": 1, "31": 1, "33": 1, "34": 1, "35": 1, "36": 1, "39": 1, "40": 1, "42": 1, "46": 1, "50": 1}

## Teacher move type (disagreements)
- {"quiet": 93, "check": 9, "capture": 11}

## Combined S4.0A + S4.0B interpretation

- S4.0A: per-node cost concentrated in legal-move make/unmake filtering (~120/node);
  depth bound by tree size (qsearch ~80% of nodes); LMR re-search low; TT under-utilized.
- S4.0B (paired): failures are overwhelmingly quiet moves (83%) ranked late, and at
  MATCHED depth the teacher move is competitive with CurrentFinal's own move for 73% of
  disagreements (SEARCH_LIKE). Only ~10% are genuine static-evaluation cases (EVAL_LIKE).
- The original EVAL_SUSPECT=45 was largely a measurement artifact: of the 42 surviving,
  31 reclassify as SEARCH_LIKE under the paired comparison; only 6 remain EVAL_LIKE.

## Recommended S4.1 direction

**Search guidance - root quiet-move ordering.** The dominant, cleanest signal is that
good quiet moves are tried too late at the root (median initial rank ~13, 76% at rank>=8)
and lose to LMR/pruning before they are recognized, yet they are competitive when given
equal depth. This points to root-level candidate ordering (e.g. applying the history/
killer/static ordering already used at non-root nodes at the root, or a quiet-move
history-based root prioritization), NOT a broad evaluator stack.

EVAL_LIKE (11, all quiet, flat ~-80cp at depth 3..7) remains a real but secondary target;
a single attributable positional evaluator term would be the later evaluation step.
