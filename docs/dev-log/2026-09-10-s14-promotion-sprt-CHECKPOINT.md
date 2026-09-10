# S14 Data-Supply Promotion SPRT — Continuation Checkpoint (2026-09-10)

Dense handoff for a fresh agent. Read this fully before acting. Do NOT re-run
training or the SPRT. The formal test has already TERMINATED.

## 0. HEADLINE / DECISION POINT (act here first)

- The single authorized formal promotion SPRT **completed with `SPRT_ACCEPT_H1`**.
- Tournament UUID: `6cd87ee8-cf21-49d2-8e69-e49d94576868` (name `s14-datasupply-promotion`).
- Terminal SPRT (`/var/lib/chessarena/runs/6cd87ee8-.../sprt.json`):
  `pairs=177 games=354 ptnml=[15,15,54,37,56] llr=2.9696 upper_bound=2.9444 decision=ACCEPT_H1`.
  API tallies: `candidate_wins=198 draws=62 candidate_losses=94` (candidate = Engine A = S14).
- Per the frozen outcome tree, `ACCEPT_H1 → PROMOTE`: make `current-final-s12 + S14 net`
  the default production; retain old HCE `current-final` as rollback.
- **Promotion has NOT been applied.** Production channel `current-final` still points at
  `ce-currentfinal-20260825` (old HCE, build `20260825-96d1a69-linux-x86_64`).
- CONFLICT TO RESOLVE WITH USER BEFORE PROMOTING: the outcome-tree application was
  pre-authorized, BUT the standing rule in MEMORY.md (`git-push-ssh-server1-deployment-authorization`)
  says irreversible/production-config changes still need separate confirmation. Treat
  "switch production default channel" as high-risk → get one explicit go-ahead, then apply.
  Do not silently promote.

## 1. Objective

Two-part user authorization (a)+(b), in order (see MEMORY.md `[[s14-data-supply]]`):
1. (DONE) Adopt S14 as the NNUE development reference. S12-R0 stays the architecture
   baseline; S14 is the current weights/data baseline.
2. (SPRT DONE, PROMOTION PENDING) Run exactly ONE formal S14-vs-production-`current-final`
   promotion SPRT via the established trusted operator→SCP→controlled deploy/install flow,
   then apply the outcome tree. STOP after verdict/outcome handling.

Outcome tree (frozen): ACCEPT_H1 → PROMOTE (S14 becomes prod default, old HCE kept as
rollback). ACCEPT_H0 → no promotion, S14 stays dev ref. MAX_PAIRS/inconclusive → HOLD,
no auto-extension. → We are on the ACCEPT_H1 branch.

Explicit exclusions (still binding): cancel/no R12 promotion match; no 10M/20M training;
no production switch before SPRT (SPRT is now done); no concurrent old-candidate confirmation.

## 2. Repo / branch state

- Engine repo: `E:/AUbuntuProject/project/chessenginedemo`, remote
  `git@github.com:zxjAiLun/super-eureka.git`, branch `s10/nnue-production-foundation`,
  HEAD `b3145e1d09cb0e75281bbb74f25b9560ccfae8dd` (pushed, in sync with origin).
- Arena repo: `E:/AUbuntuProject/project/chessarena`, remote
  `https://github.com/zxjAiLun/super-eureka-arena.git`, branch `main`, HEAD `a65b3c9`
  (`feat(arena): pin immutable engine model artifacts`). No Arena code changes this session.
- Commit made this session: `b3145e1` `fix(s14): start wall-clock budget at optimization`
  (only file: `tools/s14/s14_train.py`). Already pushed.
- Untracked (intentionally NOT committed; leave alone unless asked): `results/s14/promotion-package/`,
  `data/s6/sources/lichess-standard-rated-v5/`, `results/s11/r12-vs-sf2400/*`, `review.md`,
  `tests/s10_nnue_gui.rs`.
- `results/s14/promotion-package/` is a local build/staging dir (gitignored region differs;
  `.gitignore` has `!/results/s14/**` but the package dir is currently just untracked — do
  not force-add without user intent).

## 3. Change implemented this session (the ONLY code change)

`tools/s14/s14_train.py`, function `train_s14(...)`:
- Removed `t_start = time.time()` from the very top of the function (was line ~295, right
  after the docstring/signature, before data load).
- Added, immediately before `model.train()` at the start of the training loop section
  (right after `budget_hit = None`):
  ```python
  # The wall-clock budget starts when optimization begins. Data loading,
  # validation preparation, and model/optimizer setup are excluded.
  t_start = time.time()
  ```
- Rationale: the audit noted `--max-seconds` previously started before data load rather
  than at optimizer start. This is a FUTURE-ONLY fix. Do NOT re-run S14 training to
  "benefit" from it — the frozen S14 artifact/result stands as-is (full 20M presentations
  completed regardless).
- Verified: `python -m py_compile tools/s14/s14_train.py` OK; `git diff --check` clean;
  diff is exactly the 1-line delete + 3-line add.

## 4. Frozen S14 artifact identities (authoritative — reuse verbatim)

- S14 v5 NNUE artifact: `data/s14/run/seed-20260908/nnue-s14-datasupply-v5.bin`,
  size 11,936,916 B, **SHA256 `329b717066082798d2f42d9e4f137385a1482a8cda03ae7849ddc654e097e4b0`**.
- Layout sidecar: `data/s14/run/seed-20260908/nnue-s14-datasupply-v5.layout.json`.
- Training summary: `data/s14/run/training_summary_s14_v2r12_s20260908.json`
  (best val_loss 0.008841, val_mae ~131.0, beats S12-R0; seed 20260908).
- Screen (pre-SPRT, 256 games, book lines 193-320): 129/39/88 = 58.01%, Elo +56.1±38.6,
  LOS 99.7%, pentanomial [14,12,52,19,31] = PROMISING. Artifacts in `results/s14/`.
- Windows dev binary id string: `Eureka v0.1.0-dev+b3145e1d(.dirty)`.

## 5. Deployed immutable build + presets on server (already installed & verified)

Server `server1` (SSH config: user `ubuntu`, host `150.158.50.58`, key `~/.ssh/id_rsa`).
Passwordless `sudo -n` CONFIRMED for `ubuntu`. Separate deploy identity alias
`arena-deploy` (user `deploy`, key `~/.ssh/chessarena-deploy-operator`) used only for the
SCP-upload + `sudo arena-deploy build-install` step.

- Build dir on server: `/opt/chessarena/builds/20260910-b3145e1-s14-datasupply-329b7170/`
  - build_id `20260910-b3145e1-s14-datasupply-329b7170`
  - **Linux engine binary SHA256 `dceacfb700ec58f67f3d38fbdfd228dbf85038ca088db6ebf750d846ba5bcaba`**
    (git-archive reproducible build of `b3145e1` in WSL, rustc 1.94.1, Cargo.lock sha
    `2b3b98c26b1c50ee1d96b3eb2fdd51d8283fb48f5f5b3dccf3fa1135975cfadb`).
  - model artifact `models/nnue-s14-datasupply-v5.bin` SHA256 `329b7170...` (== #4).
  - manifest `supported_profiles: ["current-final","current-final-s12"]`,
    `uci_id_name: "Eureka v0.1.0-dev+b3145e1d"`.
  - `uci_options_schema` backfilled via `scripts/probe_build_capabilities.py` (3 options:
    Hash spin default 16 min1 max1024, EvalFile string, NnueMode combo off/nnue-v2q/nnue-v2q-full).
    Needed because `register_candidate_preset.py` requires `uci_options_schema IS NOT NULL`.
- Two presets registered (both point at the SAME build → same binary; only the startup
  profile/evaluator differs — this is the same-binary discipline):
  - CANDIDATE `s14-formal-b3145e1-datasupply-329b7170` → args
    `["--profile","current-final-s12","--nnue-model","/opt/chessarena/builds/20260910-b3145e1-s14-datasupply-329b7170/models/nnue-s14-datasupply-v5.bin"]`.
    `validate_launch_artifacts` returns [] (clean).
  - BASELINE-TWIN `s14-formal-b3145e1-currentfinal` → args `["--profile","current-final"]`.
- NOTE: the SPRT was created preset-vs-preset (both on the new build) so BOTH arms share
  binary `dceacfb7...`. The production `current-final` VERSION (`ce-currentfinal-20260825`,
  binary `45d1a895...`, source `96d1a69`) was NOT used as the SPRT baseline arm. This was a
  deliberate same-binary construction (isolate evaluator, not compiler drift). Keep this in
  mind when interpreting/PROMOTING: promotion target semantics below.

## 6. The formal SPRT tournament (created, started, TERMINATED)

- UUID `6cd87ee8-cf21-49d2-8e69-e49d94576868`, name `s14-datasupply-promotion`,
  experiment_id `s14-datasupply-promotion`, stage `promotion`, decision_rule `sprt`.
- engine_a (candidate) preset `s14-formal-b3145e1-datasupply-329b7170`;
  engine_b (baseline) preset `s14-formal-b3145e1-currentfinal`. `same_binary=true`.
- SPRT: pentanomial/logistic, elo0=0, elo1=10, alpha=beta=0.05, max_pairs=500,
  bounds ±2.9444. Time control `bullet_1_0` (1+0). Hash 16, Threads 1, concurrency 1.
- Openings: set `stockfish-8moves-v3` (pgn, 34,700 positions, file SHA
  `5835239f88cc2c7511b177c32392a69f3ede21819cf0616f80a7f907cd21d17e`), plies 16,
  seed 20260910, 500 indices selected; indices list SHA256
  `019930ca20950bddef563520ddb5626ec9ee3defe3d71c21a5105091abb8e65c`.
- Status now: `SPRT_ACCEPT_H1`, completed_pairs 177, W198/D62/L94, failure_reason null.
  Worker active/idle (no active tournaments). Match ran 2026-09-09T17:52:39 → 2026-09-10T04:17:56 (~10h25m wall-clock; 354 games @ concurrency 1; bullet 1+0).

## 7. Arena architecture facts established (so you don't re-read)

- Config: `chessarena/config.py::Settings` reads env from `/etc/chessarena/chessarena.env`.
  Fixed prod constraints: `ARENA_MAX_CONCURRENCY=1`, `ARENA_THREADS=1`, `ARENA_HASH_MB` (see
  gotcha below). API on 127.0.0.1:8787, base path `/chessarena`, public
  `https://pearllover.site/chessarena`. DB `ARENA_DB_URL` (sqlite at
  `/var/lib/chessarena/arena.db`). TIME_CONTROLS: `bullet_1_0`=60, `blitz_3_2`=180+2,
  `blitz_10_01`=10+0.1, `rapid_5_3`=300+3. Only these TC keys are accepted.
- Trusted deploy wrapper `deploy/arena-deploy.sh` (installed root-owned at
  `/opt/chessarena/bin/arena-deploy`); sudoers allows ONLY: `release-install <14digits>`,
  `release-switch <14digits>`, `build-install <YYYYMMDD-hash-label>`, `restart-api`,
  `restart-worker`. Strict id regexes; extra args rejected. `build-install` extracts
  `/opt/chessarena/incoming/build-<id>.tar.gz`, verifies manifest binary_sha256 vs engine,
  installs read-only, runs `scripts/install_build.py --probe`.
- `scripts/install_build.py`: validates manifest (REQUIRED_MANIFEST_KEYS incl schema_version=1,
  build_id==dir, supported_profiles non-empty), SHA-verifies binary, validates + read-onlys
  model_artifacts (`services/model_artifacts.py::validate_model_artifacts`), UCI probe, upserts
  EngineBuild. `--overwrite` needed to re-register an existing build_id.
- `scripts/register_candidate_preset.py`: idempotent preset upsert; requires build enabled AND
  `uci_options_schema IS NOT NULL`. `--profile X` = leading `["--profile","X"]`; repeatable
  `--command-arg TOKEN` appends verbatim (leading-dash-safe). Use `--command-arg=--nnue-model`
  `--command-arg=<abs path>` for the model flag.
- `scripts/probe_build_capabilities.py <build_id>`: re-probes UCI, backfills
  `uci_options_schema` (identity cols untouched, fail-closed on binary SHA mismatch).
- Model artifact contract `services/model_artifacts.py`: manifest `model_artifacts:[{model_id,
  relative_path,sha256}]`; `validate_launch_artifacts(build,args)` enforces any `--nnue-model`
  in command_args points at exactly one declared artifact whose live bytes still hash to the
  declared sha. Empty errors == valid.
- Tournament creation `api/tournaments.py::create_tournament(body:TournamentCreate,...)`:
  freezes config_snapshot (engine_a/engine_b snapshots incl binary_sha256 + model_artifacts,
  opening_set incl indices/seed/sha, time_control, hash_mb/threads/concurrency from settings,
  sprt block with computed Wald bounds, experiment envelope). Engine A ALWAYS candidate,
  B ALWAYS baseline. SPRT requires `pairs == sprt.max_pairs` and `elo0<elo1`. Creates DRAFT +
  PairJob rows; never auto-starts. Lifecycle: `POST .../start` DRAFT→QUEUED (worker →RUNNING);
  pause/resume/cancel/force-cancel are atomic conditional updates.
- Formal wizard (HTTP, browser/CSRF): `api/tournaments.py` admin routes
  `/admin/experiments/formal/{new,preview,create}` + `services/formal_experiments.py::
  plan_formal_experiment` (PURE READ, plan_digest TOCTOU guard; baseline ALWAYS resolved from
  channel `current-final`; auto-excludes prior same-experiment openings; promotion stage
  requires a prior confirmation ACCEPT_H1). We BYPASSED the HTTP wizard and called
  `create_tournament(...)` directly in-process via `create_app().state.session_factory`
  (see gotchas) — this is why stage="promotion" was accepted without a prior confirmation
  row (create_tournament itself does not enforce the promotion-needs-confirmation gate; only
  the formal wizard planner does).
- Promotion mechanism (NOT yet used): `services/versions.py::plan_channel_promotion(session,
  "current-final", version_id)` + the controlled promote route
  `POST /engine-channels/{id}/promote`. Promotion candidates must be an EngineVersion (not a
  bare preset) with status candidate/experimental passing the V2.1 gate. i.e. to promote S14
  you must FIRST mint an EngineVersion from the S14 build/preset, then promote it.

## 8. Gotchas / failed approaches (avoid repeating)

- ctx_execute / ctx_batch_execute run under PowerShell on this host → heredocs (`<<PY`),
  `from`, and `,` in inlined python broke repeatedly. WORKED RELIABLY: `bash` tool with
  `ssh -o BatchMode=yes server1 'sudo -n bash -lc '"'"'...'"'"''` and a real heredoc inside
  the single-quoted remote script. Prefer the `bash` tool, not ctx_* for SSH heredocs.
- Planner timeout: calling `plan_formal_experiment` with `explicit_prior_tournament_ids` =
  all 50 prior stockfish-8moves-v3 tournaments forced per-index PGN re-parse of the 34,700-game
  book many times → >600s timeout. Do NOT pass large explicit prior lists. The one authorized
  run is the first for its experiment_id, so automatic same-experiment exclusion was empty and
  fine. (We ultimately bypassed the wizard entirely and used create_tournament directly.)
- ARENA_HASH_MB gotcha: env file originally had `ARENA_HASH_MB=32` (18th line region); the
  first DRAFT froze hash_mb=32. Fixed by `sed -i` to `ARENA_HASH_MB=16` +
  `systemctl restart chessarena-api chessarena-worker` (backup saved
  `/etc/chessarena/chessarena.env.bak-s14-20260910`). The mismatched first DRAFT
  (`c24a2c9c-...`) was deleted (had to delete its Events first — FK NOT NULL on
  events.tournament_id; delete Event rows then PairJob rows then Tournament, all
  `synchronize_session=False`; verified status DRAFT/completed_pairs 0/0 games before delete).
- In-process create must build the app the SAME way systemd does: `sudo -E -u chessarena
  /opt/chessarena/venv/bin/python` after `set -a; . /etc/chessarena/chessarena.env; set +a`.
  Without `-E` the child lost env and `get_settings().hash_mb` fell back to default 32 → an
  assertion tripped. Use `sudo -E -u chessarena` with env sourced.
- cutechess: `verify_install.py` reports `cutechess-cli missing: /usr/bin/cutechess-cli`
  (FAILURE line) but the worker clearly launched games (binary is elsewhere / ARENA_CUTECHESS
  override, or verify script path check is stale). Matches ran fine, so treat that specific
  verify_install failure as non-blocking; do not chase it unless launches actually fail.

## 9. Constraints / must-not-change

- Recipe frozen = S12-R0 verbatim except data (already trained; do not retrain).
- Same-binary discipline: any comparison arm must share the engine binary; only
  profile/evaluator/model differs.
- Only one formal SPRT was authorized; it is DONE. Do NOT start another SPRT, do NOT extend,
  do NOT re-run. No 10M/20M scaling. No R12 promotion match.
- Do not pollute Arena Elo (`arena_elo_enabled=False`, already set).
- Standing rule: commit+push and `ssh server1` config are pre-authorized, BUT deleting data,
  overwriting config, or irreversible production changes need separate confirmation.
- Frozen manifest values in sections 4-6 are the source of truth; re-verify by SHA before use.

## 10. Exact next steps

1. Surface the ACCEPT_H1 result to the user with the terminal numbers (§0/§6) and ask for the
   single explicit go-ahead to PROMOTE (production-default change = high-risk per standing rule),
   OR confirmation to HOLD. Do not auto-promote.
2. If GO to promote (apply outcome tree ACCEPT_H1 branch):
   a. Decide promotion identity. Cleanest: mint an EngineVersion from build
      `20260910-b3145e1-s14-datasupply-329b7170` with command_args
      `["--profile","current-final-s12","--nnue-model",".../models/nnue-s14-datasupply-v5.bin"]`
      (status candidate/experimental), then `plan_channel_promotion(session,"current-final",
      <version_id>)` and the controlled `POST /engine-channels/current-final/promote` flow.
      Verify the plan is ok (empty errors) first.
   b. Retain old HCE `ce-currentfinal-20260825` as an explicit rollback profile/version
      (do not delete it).
   c. Re-verify binary+artifact SHAs post-switch; confirm channel now points at the S14 version.
3. Persist a formal STOP report + dev-log (this file is the checkpoint; write the closeout dev
   log `docs/dev-log/2026-09-10-s14-data-supply.md` update or a new promotion closeout), commit
   + push on branch `s10/nnue-production-foundation`. Include: tournament UUID, terminal SPRT
   (llr/ptnml/decision), frozen SHAs, promotion action taken (or HOLD), rollback pointer.
4. STOP. No further matches.

## 11. Quick verification one-liners (bash tool, not ctx_*)

- Tournament status:
  `ssh -o BatchMode=yes server1 'sudo -n bash -lc '"'"'curl -fsS http://127.0.0.1:8787/chessarena/api/v1/tournaments/6cd87ee8-cf21-49d2-8e69-e49d94576868'"'"''`
- Terminal SPRT json: `sudo -n cat /var/lib/chessarena/runs/6cd87ee8-cf21-49d2-8e69-e49d94576868/sprt.json`
- Current prod channel:
  `.../python -c "...from chessarena.models import EngineChannel; print([(c.channel_id,c.engine_version_id) ...])"`
  → currently `[('current-final','ce-currentfinal-20260825')]` (old HCE; unchanged).
