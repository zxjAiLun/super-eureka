# S4.3D — Formal Pentanomial SPRT (started)

## SANITY (S4.3C evidence, local structural/statistical check)

- pair structure: 400 games / 200 pairs, reversed colors, same TC, engines OK
- recomputed candidate W/L/D: **185 / 142 / 73 = 55.375%** (exact match)
- **Ptnml(0-2) = [25, 20, 80, 37, 38]** (sum 200)
- color: candidate White 60.75%, Black 50.00%; baseline White 50.00%,
  Black 39.25%; same-color improvement **+10.75pp both** (opening bias, not a
  candidate-only color effect)
- termination: checkmate 327, stalemate 2, insufficient 19, fifty 9,
  repetition 43, unknown 0; **time forfeits 0, crash 0, illegal 0**
- timing (per-move comments): candidate/baseline × White/Black all median
  0.19s / p95 0.35s / max 0.38s — no pathological asymmetry
- opening overlap vs 50-pair screen: **0** -> independent confirmation = YES

**SANITY = PASS** -> automatic authorization to proceed (B7).

## ARENA SPRT PLUMBING

- Arena commit: `debf8e8` (feat: formal pentanomial SPRT) + `c9c1dd5`
  (fix: opening exclusion one-pass)
- chessarena/services/sprt.py: dependency-free pentanomial SPRT mirroring
  Fishtest LLRcalc (logistic model, regularize 1e-3, MLE_expected via
  bisection); pair classification; Wald bounds; sequential decision
- differential validation: 18 unit tests incl. parity with the OFFICIAL
  Fishtest LLRcalc.py (vendored tests/reference, scipy test-only) on fixed
  pentanomial vectors, neutral/H1/H0/mixed/boundary/inversion-symmetry
- scheduler integration: Ptnml recomputed from VERIFIED COMPLETE pairs only;
  sprt.json persisted per pair; stops at Wald boundary (SPRT_ACCEPT_H1/H0) or
  max-pairs (SPRT_MAX_PAIRS); half-completed pairs never counted
- scheduler tests: ACCEPT_H1 + MAX_PAIRS end-to-end; full Arena suite green

## FORMAL SPRT

- **tournament ID: 86835da4-bdb4-4514-a950-7a5ecf1f132a** (name
  s43d-formal-sprt) — status RUNNING at report time
- EngineBuild: 20260809-b4de653-linux-x86_64, binary SHA
  c6a08996d14c4746df77a783a81d660ba9f0db0ae8fc1bcd1239896f8c7e607f
- candidate preset: s43b-legality-fast (`--profile current-final-legality-fast`)
- baseline preset: s43b-current-final (`--profile current-final`)
- opening set: stockfish-8moves-v3 (SHA 5835239f88cc2c7511b177c32392a69f3ede21819cf0616f80a7f907cd21d17e,
  34,700 positions), plies=16
- seed: 20260811; exclusion: 250 prior openings (50-pair screen + 200-pair
  confirmation); frozen indices: 2000 fresh, overlap with exclusion = 0
- TC 10+0.1, Hash 16MB, Threads 1, concurrency 1, strict color reversal,
  Arena Elo OFF
- frozen SPRT contract: pentanomial / logistic / elo0=10 / elo1=30 /
  alpha=beta=0.05 / bounds +-2.9444389791664403 / max_pairs=2000
- starts from ZERO (no prior screens seed the LLR)

Evidence: frozen_opening_indices.json (this dir); sprt.json + pairs/PGN +
verification live under /var/lib/chessarena/runs/86835da4-... on the server;
sanity tool: tools/s43d_sanity.py.

## Deployment

- Arena release 20260810185329 (current), DEPLOY_SOURCE_SHA c9c1dd5,
  services active, health ok (four fields ok)
- formal test running on the server; final decision (ACCEPT_H1 / ACCEPT_H0 /
  MAX_PAIRS / INTEGRITY_FAIL) to be reported when the tournament terminates
