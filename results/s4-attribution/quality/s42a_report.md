# S4.2A — Evaluation Residual Attribution report

Target: the 11 clean EVAL_LIKE positions (repaired S4.0B corpus): teacher
move T vs CurrentFinal own move O, matched forced depths 3..7, delta(T-O)
stays materially negative (~-80cp) without convergence.

Method: `bench eval-breakdown` (new, bench-only) emits the base material/PST
lane plus the six dormant E2 positional components (pawn_structure, mobility,
piece_activity, rook_activity, development_space, king_safety), all in WHITE
perspective, phase-interpolated. For each position: apply T -> PT, apply O ->
PO, compute component_delta = component(PT) - component(PO), re-signed to the
ORIGINAL root side's perspective (positive = the component favours the teacher
move T, i.e. directionally consistent with Stockfish).

## 11-position EVAL_LIKE component table (delta T-O, root perspective)

| component | pos (favours T) | neg (favours O) | zero | median delta | median |delta| |
|---|---|---|---|---|---|---|
| material_pst | 0 | 10 | 1 | -15.0 | 15.0 |
| pawn_structure | 0 | 3 | 8 | 0.0 | 0.0 |
| mobility | 3 | 4 | 4 | 0.0 | 6.0 |
| piece_activity | 1 | 2 | 8 | 0.0 | 0.0 |
| rook_activity | 1 | 4 | 6 | 0.0 | 0.0 |
| development_space | 1 | 6 | 4 | -1.0 | 1.0 |
| king_safety | 1 | 0 | 10 | 0.0 | 0.0 |

The base material/PST lane is the strongest signal and it is NEGATIVE (10/11):
CurrentFinal's own move O has better material+PST than the teacher's T. None of
the six positional components shows positive directional consistency on the 11.

## Broader directional table (all 113 repaired disagreements, same one-ply
T-vs-O component deltas)

| component | pos | neg | zero | median delta | median |delta| |
|---|---|---|---|---|---|---|
| material_pst | 33 | 67 | 13 | -5.0 | 10.0 |
| pawn_structure | 16 | 12 | 85 | 0.0 | 0.0 |
| mobility | 46 | 48 | 19 | 0.0 | 8.0 |
| piece_activity | 16 | 11 | 86 | 0.0 | 0.0 |
| rook_activity | 29 | 31 | 53 | 0.0 | 1.0 |
| development_space | 39 | 43 | 31 | 0.0 | 2.0 |
| king_safety | 24 | 15 | 74 | 0.0 | 0.0 |

Mobility is near-noise (46/48) on the broader set; everything else is
mostly-zero or weak.

## Cost estimate

`bench eval-breakdown --repeats` on a middlegame FEN: full 6-term breakdown =
~6.4x base `evaluate()` wall time (200k reps: 30ms vs 191ms). The full set
builds shared attack maps plus all six term scans; a SINGLE term candidate
would cost far less (base lane + one term scan, roughly 1.1-1.5x base). Cost is
moot here because no term qualifies.

## Candidate-selection verdict: NO CLEAR TERM

Per the S4.2A selection rule (strong directional consistency on the 11 AND
non-pathological on the broader set): no individual dormant term qualifies.

- No positional component has positive directional consistency on the 11.
- The only consistent signal is the base material/PST lane going NEGATIVE
  (10/11 on the 11, 67/113 on the broader set): CurrentFinal's base evaluator
  prefers its own move O over the teacher's T, and no dormant term reverses
  that.
- Therefore no S4.2 evaluator candidate is recommended from the dormant
  component set. Do not combine several weak terms into a new integrated E2.

## What this means

The 11 EVAL_LIKE gaps are NOT explained by a single missing positional term in
the dormant E2 set. The wrong preference lives in the base material/PST lane
itself (the teacher move is often a quiet/positional move whose compensation
the base lane does not credit, and the six dormant lanes add only
weak/noise-level deltas). Possible next steps (not chosen yet):
- deeper positional modelling (passed pawns, outposts, shelter) - none exist
  in the dormant set;
- or accept that at ~-80cp / 11 positions the residual is below the current
  attribution resolution and re-focus on the Core (make/unmake legality
  filtering) line, which remains the strongest measured bottleneck.

Artifacts: results/s4-attribution/quality/s42a_components.jsonl (per-position
one-ply component deltas for the 11 and the 113).
