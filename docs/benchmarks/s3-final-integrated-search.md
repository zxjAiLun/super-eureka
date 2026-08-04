# S3-FINAL — Integrated selective-search quick screen

Status: **QUICK SCREEN PASS — CANDIDATE ONLY**

This milestone evaluates one explicitly composed candidate profile. It does
not change the production `Current` profile and does not establish Elo, SPRT,
or a promotion decision.

## Candidate boundary

The candidate is `current-final`:

```text
Current PVS
+ specialized qsearch move generation
+ aspiration windows
+ conservative LMR
+ verified null-move probe path
+ shallow futility pruning
+ conservative qsearch SEE pruning
```

The candidate does not enable CurrentEval2/threat-aware evaluation, forcing
extensions, quiet qchecks, threat ordering, root reordering, fast SEE, or any
new evaluation term or weight. `Current` remains the default production
profile and is unchanged.

The composition was added in `91347775906f3f5d3730c9e9596037493429776d` and
the gate/tooling and artifacts are committed by the follow-up test commit.

## Fixed-time depth gate

The gate used the committed S2.2a manifest:

- 25 positions across five fixed groups;
- `current` versus `current-final`;
- 1,000 ms and 3,000 ms per search;
- one fresh process per search, cold 16 MB hash, one thread;
- 100 searches total with rotated profile order;
- legal bestmove and PV replay for every row;
- Current candidate counters must remain zero;
- unauthorized candidate counters must remain zero;
- every authorized feature must produce activity somewhere in the matrix.

The committed artifact reports:

| Time | Current median depth | CurrentFinal median depth |
| ---: | ---: | ---: |
| 1 s | 5 | 6 |
| 3 s | 6 | 7 |

The gate decision is `PASS`: both tiers have a one-ply median uplift, no
position is two plies behind at both tiers, Current remains free of candidate
counters, and the authorized candidate counters are active. This is a
search-efficiency result, not a strength claim.

Artifact: [`depth-gate.json`](../../results/s3-final/depth-gate.json)

## Independent 100-game quick screen

Because the depth gate passed, the bounded match was run on an independent
opening source rather than the 25 debug positions:

```text
candidate: CurrentFinal
baseline:  Current
time control: 10+0.1
Hash: 16 MB
Threads: 1
games: 100 = 50 opening pairs
pairing: strict candidate color reversal
```

The opening positions are the first 50 sequential entries of
`tests/data/openings/d1.14-openings-v1.epd`, with each position played twice
and the candidate color reversed. The cached `cutechess-cli` 1.5.1 manager was
used because no `fastchess` executable was available in the workspace.

The final PGN verifier found:

```text
CurrentFinal: 51 wins, 30 losses, 19 draws
candidate score: 60.5%
candidate as White: 29-14-7
candidate as Black: 22-16-12
```

All 100 games parsed, every move was legal, all 50 opening pairs were
complete and color-reversed, and the manager reported no crash, illegal move,
timeout, or forfeit. The only stderr line was the expected opening-repeat
warning.

Artifacts:

- [`match.pgn`](../../results/s3-final/match/match.pgn)
- [`match-summary.json`](../../results/s3-final/match-summary.json)
- [`cutechess.stdout.log`](../../results/s3-final/match/cutechess.stdout.log)
- [`cutechess.stderr.log`](../../results/s3-final/match/cutechess.stderr.log)

The 60% line was a predeclared quick-screen threshold. The result clears that
line and therefore gives `CurrentFinal` permission for a separate engineering
review and a longer, independently audited comparison. It is not a formal Elo
claim, not an SPRT decision, and does not authorize replacing `Current` today.
No Stockfish analysis was run for this milestone.

## Final boundary

```text
S3-FINAL fixed-time depth gate: PASS
S3-FINAL 100-game quick screen: PASS (60.5%)
Current: unchanged / production
CurrentFinal: candidate-only / eligible for next review
Formal Elo/SPRT: not run
Current promotion: not authorized
```
