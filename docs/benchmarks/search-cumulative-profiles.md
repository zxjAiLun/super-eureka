# Cumulative Search Candidate Profiles

Status: `INFRASTRUCTURE ONLY — bench candidates, not enabled in Current`

Parent: approved infrastructure tip `f23402a`.

This record adds cumulative, bench-only profile names so the existing search
candidates can be compared without changing the production `Current` profile.
The UCI path continues to select `SearchProfile::Current`; no candidate is
being accepted as an Elo improvement by this commit.

## Profile contract

| CLI profile | Enabled search features |
|---|---|
| `current` | M4.1 ordering + PVS |
| `current-aspiration` | `current` + aspiration |
| `current-aspiration-lmr` | previous profile + LMR |
| `current-aspiration-lmr-futility` | previous profile + futility |
| `current-aspiration-lmr-futility-see` | previous profile + SEE ordering |

All cumulative profiles retain the existing Current PVS and quiet-move
ordering behavior. `null` and `NullMoveCandidate` are deliberately absent from
the cumulative sequence because the current implementation is a verified
probe, not pruning, and is not part of this comparison stack.

The feature predicates are tested directly so each profile enables exactly its
declared suffix. Existing single-feature candidates remain available for
isolated diagnostics, while the fixed smoke locks and production profile stay
unchanged.

This is a configuration and test milestone only. Relative Elo/SPRT testing is
the next separate milestone and must compare the cumulative profiles in order.
