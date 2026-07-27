# EVAL 1B — Exact KQK/KRK mop-up evaluation

Status: **IMPLEMENTED — independent final review pending**
Parent: EVAL 1A tapered evaluation and King PST.

## Scope

The mop-up term is enabled only when the position contains exactly two Kings
and exactly one Queen or Rook. Any pawn, minor piece, second heavy piece, or
additional King falls back to the ordinary tapered evaluation. This prevents a
special-case endgame heuristic from leaking into normal middlegames.

The ordinary material/PST score remains primary. The mop-up term is bounded
and is evaluated from the winning piece's color before being converted to the
side-to-move perspective.

## Winning plan encoded

For the exact KQK/KRK positions, the bonus combines:

```text
king proximity  = max(0, 7 - Chebyshev(strong_king, weak_king)) * 20
edge pressure   = max(0, 3 - distance_to_edge(weak_king)) * 18
confinement     = max(0, 8 - weak_king_legal_moves) * 16
```

The bonus is clamped to `[-300, 400]`. If the sole Queen/Rook is adjacent to
the weak King and is not defended by the strong King, a material-preservation
discount is applied (`300` for a Queen, `260` for a Rook). This discourages
walking the only winning piece into an immediate king capture.

The weak King's mobility is counted on temporary board copies, so moving off a
rook/queen ray and capturing the sole strong piece are tested on the resulting
board. The caller's position remains untouched.

## Stalemate boundary

When the weak side is to move, has no legal King move, and is not in check, the
position is scored as `0`. Search's existing terminal classification remains
the authoritative result; this evaluator guard prevents a large static
mop-up score from encouraging a stalemate. Checkmate is still handled by the
search terminal path.

## Regression coverage

`tests/eval1b.rs` covers:

- KQK edge pressure and strong-King proximity;
- KRK confinement and positive material semantics;
- no special-case leakage when other material is present;
- stalemate scoring as zero;
- read-only evaluation;
- fixed KQK/KRK search positions that choose a constricting move and see a
  short mate while restoring the root position.

The root of the standard queen-win smoke fixture contains an additional black
Queen and pawn, so the exact-material detector does not apply at the root.
The winning PV can then capture the pawn and reach an exact KQK descendant;
the mop-up term is therefore intentionally visible in the depth-3 score. Its
approved pre-EVAL M4.0 result remains `963 / e4a4 / cp 890`; under EVAL 1B the
M4Reference regression is:

```text
nodes = 969
bestmove = e4a4
score = cp 990
PV = e4a4 h4h3 a4h4 h8g7 h4h3
```

The KQK/KRK changes are validated by the dedicated exact-material fixtures
above. This milestone proves the exact-material evaluator and its regression
closure; it does not provide an Elo estimate or claim a playing-strength gain.
Current-profile SEE/pruning measurements are separate search candidates, not
an Elo claim and not a rewrite of the approved M4.2 historical benchmark
record.

## Non-goals and next boundary

This milestone does not add general mobility, pawn structure, king safety,
SEE, qsearch shrinking, aspiration windows, null move, LMR, futility,
continuation history, bitboards, NNUE, tablebases, or multithreading. The next
authorized investigation is profiling first; the separate Depth 7–8 problem
must be measured through node/qsearch/evaluation costs before search pruning is
selected.
