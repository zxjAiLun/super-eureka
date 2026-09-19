# S14 cloud diagnostics: optimistic knight losses (read-only inspection, 2026-09-19)

> Follow-up: the owner subsequently authorized bounded local searches. Results are in
> [the time/depth and forced-continuation report](2026-09-19-s14-knight-search-ladder.md).
> Statements below about searches not yet run describe the initial read-only inspection.

## Scope and evidence

Owner asked to inspect existing **Diagnostics ready** games in
[the S14 vs SF2400 match](https://pearllover.site/chessarena/matches/435e4630-7722-4ba4-a036-5b0eb2afd5c3),
after observing apparent knight donations in the separate S16 cloud match.

- Server access: `ssh server1`; SQLite opened using `mode=ro` plus `PRAGMA query_only=ON`.
- No match/diagnostics launched, stopped, restarted or modified. No build, training, deployment or engine source edits.
- First enumeration found 69 ready diagnostic artifacts; the existing worker continued independently. The local snapshot captured **81 ready games**, all Eureka losses, from 365 requested loss diagnostics. Analysis below is on that fixed snapshot, NOT a random sample and NOT a full-match blunder-frequency estimate.
- Snapshot: `C:/Users/81489/AppData/Local/Temp/eureka-diag-20260919/s14-cloud-snapshot.json` (81 results, 65 pair PGNs/commands/verifications). Selected UCI logs: `selected-stdout.json`; aligned searches: `selected-searches.json`; extraction scripts: `scan.py`, `details.py` in the same temporary directory.
- Every diagnostic FEN was checked against replaying its exact PGN: **81/81 games aligned**, covering 5,141 Eureka moves. PGNs selected by full game block and White/Black headers, never assuming that FEN precedes Result.

### Identities and scoring

S14 production source: `315534991628ce084c34392d6abb98d71f3f591a`.
Binary: `c2428a8453d871c9a6f2ff14780f077b2c2503bdb321419d4334361c54733986`.
Model: `329b717066082798d2f42d9e4f137385a1482a8cda03ae7849ddc654e097e4b0`.

Stored actual commands use explicit `--evaluation nnue --nnue-model <absolute S14 path>`.
Selected pairs' original UCI logs report `eval=nnue-v2q`, `network=nnue-v2q`, correct EvalFile and source commit. Model SHA was independently read from the installed file. This is not the local HCE-selection incident.

- Match: **60 seconds for the entire game, no increment**, opponent SF18 with `UCI_LimitStrength=true`, `UCI_Elo=2400`.
- Diagnostics: **unrestricted SF18**, **100,000 nodes per position**, White-perspective scores; analyzer SHA `6b087694916228c905a5e14db74cca8c7e5643602226af1fa5d42353c455b9f9`. Implementation sets Threads=2, Hash=256 and clears with ucinewgame at each position. These are bounded post-game analyses, not omniscient truth or a same-budget comparison with Eureka.
- All three principal cases below have Eureka as White, so positive means Eureka is better in both quoted score streams. Eureka values are completed-search scores, **not raw NNUE static outputs**.
- Move numbers are the replay/PGN numbers. These games begin from opening FENs with move counters reset, so `4.Nd5` is not move four from the standard starting position.

## Case 1: game 187, replay ply 7, `4.Nd5?`

[Replay](https://pearllover.site/chessarena/games/1e3adbe7-188f-4b41-8786-3db5655e6628)

Before the move:

```text
r2qk1nr/pp2bpp1/2bpp3/7p/3QP3/2N1B1P1/PPPR1PBP/2K4R w kq - 2 4
```

- Actual clock command: `go wtime 54310 btime 53745`.
- Search wall duration: **1775 ms**; last completed iteration: depth **6**, seldepth 18, **+105 cp**, 87,911 nodes at 608 ms. Do not treat those nodes as the total at termination: subsequent incomplete search is not summarized by this info line.
- Printed PV: `Nd5 Nf6 Nxe7 Qxe7 Kb1 O-O Qxd6`.
- Actual continuation: `Nd5 exd5 exd5 Bd7`.
- SF100k: **+0.63 before → −2.58 after Nd5**; recommends `Rhd1` instead.
- Eureka after `...exd5`, choosing `exd5`: **−0.37**, depth 6, 1.7 s; SF100k after that recapture: **−2.66**. Later Eureka returns to about zero while the diagnostic remains around −2.5.

The knight-for-pawn loss is real and the live search is materially more optimistic. Its PV expects a different reply. A PV is not an exhaustive trace, so it does **not** prove `...exd5` was never searched.

## Case 2: game 175, replay ply 55, `28.Qc1?`

[Replay](https://pearllover.site/chessarena/games/4e24d418-91b3-4e25-ab8f-31521f8420b4)

```text
6r1/p1p1kp2/5n1Q/8/1P6/1B1q1N2/P4PP1/6K1 w - - 8 28
```

- Clock: `go wtime 24587 btime 7118`; search **803 ms**.
- Last completed iteration: depth **6**, seldepth 14, **+73 cp**, 43,344 nodes at 295 ms.
- Printed PV: `Qc1 Qd6 Qe3+ Kf8 Ne5 Ng4`.
- Actual: `Qc1 Qxf3 Qc5+ Kd8`.
- SF100k: **−1.42 before → −5.05 after Qc1**; recommends `Ne1`.
- The **g2 pawn is pinned to Kg1 by Rg8**. After `...Qxf3`, `gxf3` is pseudo-legal but **illegal** because it exposes the king. Verified with python-chess, not inferred from material counts.
- On its next turn, Eureka reports **−1.44**, later −2.53 and −9.88. It does eventually recognize trouble, but too late.

This is a concrete tactical/pin case with 24.6 seconds still on the clock, not simply a final-second flag scramble. It does not by itself identify SEE, NNUE or pruning as the faulty subsystem.

## Case 3: game 127, replay ply 49, `25.Nxe5?` — strongest search-boundary clue

[Replay](https://pearllover.site/chessarena/games/e2070801-9dd6-4261-bed7-87cb72c9c347)

```text
rr6/2pR1pbk/8/2Q1p2p/PBN1Pnq1/1PP3Pp/5P2/3R3K w - - 1 25
```

- Clock: `go wtime 27072 btime 12587`; search **885 ms**.
- Last completed iteration: depth **5**, seldepth **20**, **+200 cp**, 28,353 nodes at 198 ms.
- Printed PV: **`Nxe5 Bxe5 Qxe5 Qf3+ Kg1`**.
- After that exact printed line, **`...Qg2#` is a legal non-capturing checkmate**.
- After `...Qf3+`, both legal evasions (`Kh2`, `Kg1`) permit `...Qg2#`. Legality and terminal mate were independently verified with python-chess. Thus the anticipated `Qxe5` recapture runs into a forcing mate in this continuation.
- Actual play: `Nxe5 Bxe5 Kh2` — once the knight has been captured, Eureka does **not** play its previously anticipated queen recapture. Its score falls to **−0.27**, depth 5.
- SF100k: **+0.51 before → −1.19 after Nxe5**, then **−1.44 after Bxe5**. Best move before the sacrifice: `Qe3`.

### Production-code correspondence, not a completed causal proof

Read the exact production source, not the current working tree:

- `src/engine/search.rs` at `3155349`: `CurrentFinal` enables qsearch movegen (lines 132–135).
- Non-check qsearch calls the tactical generator; in-check qsearch searches evasions. The non-check branch uses stand-pat and then tactical moves (around lines 5611–5653).
- `src/chess/movegen.rs` lines 481–541: the tactical generator produces **captures, en passant, promotions**, not quiet non-promotion moves. The fallback `is_tactical` in `search.rs:4789` has the same scope.

At the printed PV endpoint, Black is not in check and `Qg2#` is not a capture or promotion. **That move is outside the non-check qsearch move set.** This provides a specific mechanism compatible with the observed optimistic horizon. `seldepth=20` means some branch got that deep, not that this mating continuation did.

However, the logs are not a full search trace. A PV reconstruction/TT issue, reductions or another interaction could also matter. We have **not** run forced-line, larger-depth or feature-isolation tests; do not call qsearch the sole proven root cause or extrapolate this case to every blunder.

## Conclusions and bounded next step

1. **The reported symptom exists in S14**, independently of the S16 model: optimistic search scores coexist with avoidable knight losses.
2. It occurs with substantial clock time remaining (54.3 / 24.6 / 27.1 seconds), while individual moves receive only about 0.8–1.8 seconds and complete depth 5–6. Short per-move budgets are a plausible contributor, not yet a proven cure.
3. **Do not conclude FT256/FT512 or NNUE architecture inherently causes it.** The evidence points first to search-horizon/quiet-check coverage, selective search and evaluation at those boundaries. A shallow search score is not a direct measurement of network quality.
4. Preserve these three roots and full histories, then do a small same-production-identity diagnostic: time/depth ladder; explicit candidate/refutation branches; especially whether `Nxe5 Bxe5 Qxe5 Qf3+ ...Qg2#` is valued correctly. Only after that consider a bounded search change or longer-TC candidate screen. No blanket quiet-check expansion or new training is justified yet.

## S16 cloud status (incidental read-only verification)

The separate match `50d43c9a-c3de-42d1-bccb-bdd80091b50b` is now **COMPLETED, 500 pairs**, W/D/L **463/112/425**, score **51.9%** versus limited SF2400 at 1+0. Its actual stored pair command and original handshake explicitly use NNUE and `nnue-s16-shared-v5.bin`; installed model SHA equals `7d442e9cc31ab6c8cee3cd625f0ab0cdf54be58844a8e37e10ac789c4a53c35a`.

No ready diagnostics existed for that match at enumeration. No stop/restart or new analysis request was made. Its separate result cannot be subtracted from S14's result to claim a directly measured S16–S14 Elo difference.
