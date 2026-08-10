# S4.3E — Promotion of LegalityFast into production CurrentFinal

## S4.3D evidence

FORMAL_SPRT_PASS (tournament 86835da4, ACCEPT_H1 at pair 263, LLR
2.975553182682472 >= upper 2.9444389791664403; see
../s43d-formal-sprt/final/s43d-final-report.md).

## Promotion

- promotion source SHA: `26604c4` (feat(search): promote legality fast path
  into current-final)
- exact semantic change: `SearchProfile::uses_legality_fast()` now returns
  true for CurrentFinal, CurrentFinalRootHistory, CurrentFinalRootPrevScore,
  CurrentFinalLegalityFast (compatibility alias). Historical/experimental
  profiles (Current, CurrentLmr, CurrentEval2, threat variants, qsearch
  standalone candidates) keep legacy behavior. No algorithm change; the
  unpinned non-check fast path scope is unchanged (FULL legal generator only;
  tactical/evasion/has-any untouched).
- default UCI profile: current-final (no-argument startup unchanged)

## Verification

- fmt: PASS
- clippy (-D warnings): PASS
- cargo test: PASS (290 lib tests + integration)
- cargo test --release: PASS (290 lib tests)
- locked release build: PASS
- legality_fast_promotion_policy regression test: PASS
- movegen differential (pin/EP/castle/check/king/promotion/capture): PASS
- 1500-position reachable legal-walk differential: PASS
- perft differential (standard fixtures): PASS
- fixed-depth CurrentFinal vs CurrentFinalLegalityFast tree: PASS
  (nodes/score/bestmove/PV identical)

## Cross-artifact semantic check

OLD experimental artifact (20260809-b4de653, --profile
current-final-legality-fast) vs NEW production artifact (default/current-final)
on 6 representative S4 corpus positions at fixed depth 6 (cold):

- bestmove / score / nodes / PV: IDENTICAL on all 6 (startpos, open-tactical,
  castled Italian, S2.1 mid, Kiwipete, rook-pawn endgame)
- timing differs (expected; not compared)

## Artifact

- build ID: `20260811-26604c4-linux-x86_64`
- source SHA: `26604c4` (manifest git_sha)
- binary SHA-256: `f0e8f91a3a0828a158672cecdf7859dbd9a3c9bac36b965bdcc90db31b51189d`
- Cargo.lock SHA-256: `370d4ebcacd639d0bad97efc2d7c4f1eb419ef13ce9a518d1d8b2f836d64e9ee`
- rustc: 1.94.1 (e408947bf 2026-03-25); cargo: 1.94.1 (29ea6fb6a 2026-03-24)
- platform: linux-x86_64
- supported_profiles: ["current-final", "current"]
  (current-final-legality-fast remains accepted by the binary as a
  compatibility alias but is not a public production supported profile)
- tarball: build-20260811-26604c4-linux-x86_64.tar.gz
- tarball SHA-256: `34db8dce82886beaf8b69c8b631b46250a621ac7aeca63ff2b302607973ae253`
- unpack-and-rehash == binary SHA: PASS; executable bit: PASS; manifest JSON: PASS

## UCI probes (exact artifact binary)

- default: PASS (id name ChessEngineDemo, id author Rust-learner,
  search profile current-final, uciok, readyok, bestmove b1c3)
- --profile current-final: PASS (search profile current-final, behavior
  equivalent to default)
- --profile current (historical rollback): PASS (search profile current)

## Scope notes

- CurrentFinalLegalityFast compatibility alias retained (behavior equivalent
  to CurrentFinal after promotion); not the release identity.
- The S4.3B/C/D experimental artifact 20260809-b4de653 stays untouched as
  experiment provenance.
- tactical/evasion/has-any legality fast path: NOT extended in S4.3E.
- EngineVersion schema / Arena DB / ratings: NOT touched in S4.3E.

## Verdict

S4.3E_PROMOTION_READY_FOR_REVIEW
