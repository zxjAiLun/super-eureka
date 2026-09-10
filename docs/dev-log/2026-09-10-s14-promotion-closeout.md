# 2026-09-10 · S14 Promotion SPRT Closeout — ACCEPT_H1 / promotion-qualified, production HOLD (rollback blocker)

commits: `b3145e1` (S14 wall-clock timer fix — last code commit prior to this closeout);
closeout docs commit = this entry (`handoff.md` + this dev-log + checkpoint file).

Checkpoint (handoff artifact, read first for deep context):
`docs/dev-log/2026-09-10-s14-promotion-sprt-CHECKPOINT.md`

## Verdict (frozen)

**S14 passed the frozen 0/+10 pentanomial promotion SPRT (`ACCEPT_H1`, LLR 2.9696 ≥ 2.9444
after 177 pairs / 354 games) and is therefore promotion-qualified on chess-strength
evidence. Production `current-final` was intentionally left unchanged because the deployed
V2.1 EngineVersion control plane currently has no verified controlled rollback path once
the existing HCE production version becomes historical. No additional match was run and no
production mutation was made.**

Status table (frozen):

```text
S14 NNUE development reference      YES
S14 promotion SPRT                  PASS / ACCEPT_H1
S14 promotion-qualified             YES
production default                  HCE-20260825
production switch                   HOLD — operational safety
additional chess validation         NONE REQUIRED
```

## Termination evidence (read-only verified, 2026-09-10)

- Tournament `6cd87ee8-cf21-49d2-8e69-e49d94576868` (`s14-datasupply-promotion`,
  stage `promotion`, decision_rule `sprt`): status `SPRT_ACCEPT_H1`,
  `failure_reason=null`, finished `2026-09-10T04:17:56`; 0 active tournaments at
  verification time; not rated (`arena_elo_enabled=0`).
- Terminal `sprt.json` (server `/var/lib/chessarena/runs/6cd87ee8-.../sprt.json`):
  `llr=2.969566983734695` > `upper_bound=2.9444389791664403`; `pairs=177`; `games=354`;
  `ptnml=[15,15,54,37,56]`; `decision=ACCEPT_H1`.
- API tallies: candidate W198 / D62 / L94. Both arms same binary sha `dceacfb7…`
  (same-binary discipline held). SPRT params: pentanomial/logistic, elo0=0, elo1=10,
  α=β=0.05, `max_pairs=500`, bounds ±2.9444, hash 16 / threads 1 / concurrency 1,
  TC `bullet_1_0`.
- No in-repo result copy exists; the server-side artifact is authoritative.
- Frozen SHAs re-verified end-to-end: S14 artifact `329b7170…` (= `data/s14/run/seed-20260908/
  nnue-s14-datasupply-v5.bin` = build `models/` copy); engine `dceacfb7…`; opening set
  `5835239f…`; indices sha independently recomputed → `019930ca…` match.

## Why HOLD (operational, not a chess verdict)

- Forward switch has a controlled path (mint EngineVersion → `plan_channel_promotion` →
  controlled `POST /engine-channels/current-final/promote`); rollback does **not**: once
  the current HCE `ce-currentfinal-20260825` becomes `historical`, the deployed control
  plane rejects promoting a historical target back (`services/versions.py:410-414`) and
  the duplicate-fingerprint guard blocks re-minting the same identity (`_commit_version`).
- "File retained" ≠ "operable rollback". For a default production engine this is a
  production-level blocking gap; the switch is deferred until a verified controlled
  rollback path exists.
- This HOLD does NOT re-open the chess result. NOT because of: SPRT inconclusive /
  candidate weak / artifact invalid / same-binary violation. No retraining, no SPRT
  re-run, no additional matches. The promotion qualification stands and does not expire.

## Scope compliance (this round)

Allowed & done: record `ACCEPT_H1` / promotion-qualified; record the operational rollback
blocker; checkpoint errata; handoff + dev-log updates; commit + push; STOP.

Forbidden & confirmed not done: production channel switch; new S14 production-default
artifact; Arena rollback/promotion rule changes; new SPRT / additional matches; R12 match;
10M/20M training. All server access this round was read-only.

## Errata — corrections to the checkpoint

1. **Checkpoint §10.2a promo recipe is not executable as written.** Minting an
   EngineVersion with `--profile current-final-s12 --nnue-model …` then promoting it fails
   the live V2.1 gate: a production launch must use artifact-default
   `command_args=[] / uci_options={}` (`services/versions.py:340-368`; test
   `tests/test_engine_versions.py:1094`; design `docs/design/engine-version-identity.md:174-177`).
   A real production switch therefore requires building a **new default-production S14
   artifact** ("default is S14") and promoting that version — not the profile-alias version.
2. **"Retain old HCE as rollback" is wrong wording.** Correct: the old HCE is retained as
   a historical artifact/version only; it is currently **not** an operable rollback
   (see "Why HOLD").
3. Minor doc fixes: match duration ≈ 10h25m (started `2026-09-09T17:52:39`, finished
   `2026-09-10T04:17:56`; 354 games @ concurrency 1), not "a couple minutes"
   (checkpoint §6 corrected in place); checkpoint has 12 sections (§0-§11), not ~11; the
   MEMORY.md entries cited in checkpoint §0/§1 were never actually written — memory was
   re-persisted 2026-09-10 as `project_s14_promotion_state.md` (+ `MEMORY.md` index).

## Future promotion path (the only open item)

Independent ops task — **no retraining, no re-SPRT**: establish and verify a controlled
rollback path for the EngineVersion production channel. When done, re-verify:

- S14 artifact SHA `329b7170…` unchanged;
- promotion SPRT identity/result unchanged (`6cd87ee8…`, `ACCEPT_H1`);
- new default-production S14 build identity decided + built;
- rollback dry-run feasible.

Then the production promotion can be executed directly. S14's chess approval is already
earned; it does not need to prove itself again.

## References

- Checkpoint: `docs/dev-log/2026-09-10-s14-promotion-sprt-CHECKPOINT.md`
- Memory: `memory/project_s14_promotion_state.md` (CodeBuddy project memory dir)
- Repo: branch `s10/nnue-production-foundation`; last code commit `b3145e1` + this
  closeout docs commit.
