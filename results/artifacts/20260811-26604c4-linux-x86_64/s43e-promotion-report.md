# S4.3E — Promotion of LegalityFast into production CurrentFinal (REPAIRED)

## S4.3D evidence

FORMAL_SPRT_PASS (tournament 86835da4, ACCEPT_H1 at pair 263, LLR
2.975553182682472 >= upper 2.9444389791664403; see
../s43d-formal-sprt/final/s43d-final-report.md).

## Promotion

- promotion source SHA: `26604c425625d69e5b7e7b967db8926f4da01b8a`
- exact semantic change: `SearchProfile::uses_legality_fast()` now returns
  true for CurrentFinal, CurrentFinalRootHistory, CurrentFinalRootPrevScore,
  CurrentFinalLegalityFast (compatibility alias). Historical/experimental
  profiles keep legacy behavior. No algorithm change; the unpinned non-check
  fast path scope is unchanged (FULL legal generator only; tactical/evasion/
  has-any untouched).
- default UCI profile: current-final (no-argument startup unchanged)

## Verification (approved statements, unchanged)

- fmt: PASS
- clippy (-D warnings): PASS
- cargo test: PASS (290 lib tests + integration)
- cargo test --release: PASS (290 lib tests)
- locked release build: PASS
- legality_fast_promotion_policy regression test: PASS
- movegen differential: PASS
- 1500-position reachable legal-walk differential: PASS
- perft differential: PASS
- fixed-depth CurrentFinal vs CurrentFinalLegalityFast tree: PASS

## Cross-artifact semantic check (unchanged)

OLD experimental artifact (20260809-b4de653, --profile
current-final-legality-fast) vs NEW production artifact (default/current-final)
on 6 representative S4 corpus positions at fixed depth 6 (cold):
bestmove / score / nodes / PV IDENTICAL on all 6; timing differs (not compared).

## Artifact

- build ID: `20260811-26604c4-linux-x86_64`
- source SHA: `26604c425625d69e5b7e7b967db8926f4da01b8a` (manifest git_sha)
- binary SHA-256: `f0e8f91a3a0828a158672cecdf7859dbd9a3c9bac36b965bdcc90db31b51189d`
- Cargo.lock SHA-256: `370d4ebcacd639d0bad97efc2d7c4f1eb419ef13ce9a518d1d8b2f836d64e9ee`
- rustc: 1.94.1 (e408947bf 2026-03-25); cargo: 1.94.1 (29ea6fb6a 2026-03-24)
- platform: linux-x86_64
- created_utc: `2026-08-10T18:37:15Z` (binary mtime, durable evidence)
- supported_profiles: ["current-final", "current"]
- tarball: build-20260811-26604c4-linux-x86_64.tar.gz
- tarball SHA-256: `74fdacaffc8167b3cdbd768bcd33aadcf891a44d6e40ba9691371696c88bcaf8`
  (repacked with corrected manifest around the SAME byte-identical binary)
- unpack-and-rehash == binary SHA: PASS; executable bit: PASS; manifest JSON: PASS

## UCI probes (regenerated from the exact binary, actual stdout)

- default: PASS (id name ChessEngineDemo, id author Rust-learner,
  option name Hash ..., search profile current-final, uciok, readyok,
  bestmove b1c3)
- --profile current-final: PASS (search profile current-final, uciok,
  readyok, bestmove b1c3)
- --profile current (historical rollback): PASS (search profile current,
  uciok, readyok, bestmove b1c3)

## Artifact record repair

Supersedes the broken records from `c5dc6cf`:

- committed probe files contained shell invocation errors ("command not
  found"); regenerated from the exact binary via a subprocess harness (rc=0,
  no stderr). Reject-scan for "command not found"/"No such file"/"bash:"/"sh:"
  = 0 hits.
- created_utc was the invented future-dated `2026-08-11T00:00:00Z`; restored
  to the real artifact creation UTC `2026-08-10T18:37:15Z` (durable binary
  mtime evidence).
- git_sha was short (`26604c4`); now the full promotion SHA.
- tarball repacked with the corrected manifest (same binary; old tarball SHA
  `34db8dce...` superseded by `74fdacaf...`).

## Scope notes (unchanged)

- CurrentFinalLegalityFast compatibility alias retained.
- S4.3B/C/D experimental artifact 20260809-b4de653 untouched.
- tactical/evasion/has-any not extended.
- EngineVersion schema / Arena DB / ratings not touched.

## Verdict

S4.3E_ARTIFACT_REPAIR_READY_FOR_REVIEW
