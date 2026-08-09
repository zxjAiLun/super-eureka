# S4.3B — Unpinned Non-Check Legality Fast Path: local gate report

Candidate: `current-final-legality-fast` (SearchProfile::CurrentFinalLegalityFast).
Baseline: CurrentFinal. S4.3A APPROVED (dominant target LEGALITY_FILTER).

## Candidate behavior

Exactly CurrentFinal, except the FULL legal generator uses an unpinned
non-check fast path: when the side to move is NOT in check, non-king,
non-en-passant, non-castling moves from absolutely-pin-free squares are
accepted directly as legal (no make -> attack-test -> unmake probe). Pin mask
is computed locally per call by scanning the 8 slider rays from the king.
Everything else keeps the exact legacy probe. Generated move lists, order and
search tree are identical. tactical/evasion/has-any generators untouched.

## Correctness

- Direct differential (legacy vs fast): pin/EP/castle/check/promotion classes
  -> exact equality, position + zobrist fully restored.
- Deterministic legal-walk differential (1500 reachable positions): exact.
- Perft differential on the standard fixtures (startpos d4, Kiwipete d3,
  d4-ish, castling/EP-heavy, promotion): exact.
- Search-tree equivalence at fixed depth: identical nodes/score/bestmove/PV.
- Full test suite: 289 lib tests + integration all green; fmt + clippy clean.

## Probe reduction (30-position S4.0A corpus, depth 6, cold)

- legacy legality make probes: 207.28M
- candidate legality make probes: 51.40M  (-75.2%)
- fast accepts: 155.88M (84.0% of legality decisions)
- fallback probes: 29.80M (mostly in-check nodes + king moves + pins + EP + castle)

## Wall-time A/B (depth 6, cold 16MB TT, release, 30 positions)

- aggregate elapsed: CurrentFinal 19.89s vs candidate 14.10s -> -29.1% (repeat 1)
- repeat-3 interleaved on 3 representative positions: -33.3% / -22.2% / -35.5%
  (stable, far above the 10% strong-success band)
- candidate wall profile: movegen_legal share falls from ~64% to ~45-48%
  (remaining legal cost: in-check fallbacks, king moves, fast-path scan overhead)

## Fixed-time depth gate (500/1000/3000 ms, 6 corpus positions)

- 500 ms: equal completed depth (5->5)
- 1000 ms: 5->6 on two positions, equal elsewhere
- 3000 ms: 6->7 on three positions, equal elsewhere
- bestmove identical in every fixed-time comparison (no DIFF)

## Verdict: PROMISING

Local gate fully qualified:
- exact correctness (differential + perft + full tests)
- identical fixed-depth search tree (0 mismatches)
- clear wall-time gain (-29% aggregate, -22..-36% per-position repeat-3)
- fixed-time depth benefit (+1 depth at several positions, no bestmove drift)

Next step per the S4.3B gate: build an immutable EngineArtifact
(current-final-legality-fast) and run the Arena screen (20-50 paired openings,
color reversal, same Hash/Threads/TC) vs CurrentFinal before any larger paired
test / SPRT. Do NOT stack the same shortcut onto tactical/evasion/has-any yet.

Artifacts: results/s4-attribution/core/s43b_gate.json, results/s4-attribution/core/
