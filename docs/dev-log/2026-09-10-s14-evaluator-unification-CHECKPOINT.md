# S14 评估器统一 + 入口收敛 — Continuation Checkpoint (2026-09-10)

Dense handoff for a fresh agent. Read fully before acting. Engine work in this
round is COMPLETE and pushed; this checkpoint preserves context for follow-ups
(validation, future features, or incident response).

## 0. State at handoff

- Repo `E:/AUbuntuProject/project/chessenginedemo`, remote `git@github.com:zxjAiLun/super-eureka.git`,
  branch `s10/nnue-production-foundation`, HEAD **`da5f2d8`** (pushed, == origin).
  Recent commits (newest first): `da5f2d8` refactor(entry) SearchProfile 49→2;
  `52f062e` feat(evaluator) model-driven semantics + unified entry;
  `e26a2a5` docs option-route warning (superseded by 52f062e);
  `7acbce8` docs En Croissant option-route (superseded);
  `c0b627a` feat(uci) relative `--nnue-model` resolves vs exe dir;
  `b3145e1` last pre-docs engine commit (S14 trainer timer fix).
- Engine code at HEAD == `b3145e1` semantics + (a) relative `--nnue-model`
  resolution, (b) model-driven evaluator/entry refactor. NNUE numerics, search
  policy of `current-final`, default no-arg launch (HCE) all UNCHANGED by design.
- `cargo build --release` has been run at HEAD (exe at `target\release\eureka.exe`
  reports `source 52f062e7…` — built from the dirty tree at commit-1 HEAD; rebuild
  re-embeds current HEAD sha; engine content identical).
- Untracked leftovers (intentionally NOT committed): `results/s14/promotion-package/`,
  `data/s6/sources/lichess-standard-rated-v5/` (2.1G), `results/s11/r12-vs-sf2400/*`,
  `review.md`, `tests/s10_nnue_gui.rs`, `review.md`.
- Arena repo `E:/AUbuntuProject/project/chessarena` @ `a65b3c9`, untouched.
- Server `server1` (150.158.50.58): production channel `current-final` →
  `ce-currentfinal-20260825` (HCE-20260825) — **S14 promotion still HOLD**
  (rollback control-plane gap; see `docs/dev-log/2026-09-10-s14-promotion-closeout.md`).
  Deployed immutable build `20260910-b3145e1-s14-datasupply-329b7170`, engine binary
  SHA `dceacfb7…`. **The deployed build predates this refactor** — server binaries
  do NOT contain the evaluator unification; redeploy would be needed if we ever
  push these changes to the server (NOT authorized this round).

## 1. What this round delivered (authoritative summary)

User-visible bug: in En Croissant (GUI), S14 played `3...d5` ignoring a hanging
white queen (`1.e3 e6 2.Qg4 Qe7 3.Qxe6??`), eval ≈ −20 cp. Root cause chain,
fully diagnosed:

1. The GUI can ONLY use the UCI-option route (`path` + `settings`; En Croissant
   has NO launch-arguments field — user-confirmed via config JSON).
2. The option route ran under the classical profile, but material composition was
   keyed on the startup profile: `search.rs:2754`
   `let base = if profile.uses_nnue_material_residual() { raw + material_cp_stm(pos) }`.
   S14 artifact is **material_residual + V2R12** → residual was used as an
   absolute score → eval ≈ 0, material ignored → blunders.
3. Correction of an earlier wrong attribution (user-confirmed): R12 relation
   features were NEVER missing in the option route —
   `nnue_search.rs` (~:477/:487-491) computes relation rows fresh for
   Incremental+V2R12 with `r12_incremental=false` (B2 fresh-hybrid). Missing R12
   stack = SPEED only, not correctness. Do not re-lump these.
4. Additional bug found & fixed: UCI `EvalUpdated` outcome never cleared the TT
   (`uci.rs` run loop only handled `Resized`) → stale scores from the previous
   evaluator could steer the next search.

### Commit 1 — `52f062e` "feat(evaluator): model-driven evaluation semantics + unified entry"

- `nnue_search.rs`: raw/full split. `evaluate_raw_cp_i32` (was `evaluate_cp_i32`)
  = raw network output; `evaluate_raw_cp_i32_audited` (was `evaluate_cp_i32_audited`);
  NEW `evaluate_full_cp_i32(_audited)` = raw + `material_cp_stm(pos)` iff
  `model.target_mode() == NnueV2TargetMode::MaterialResidual` (private `compose_full`).
  This is the ONLY evaluation exit the search may use.
- NEW `NnueSearchState::for_search(model, root, telemetry, audit)`: delivery chosen
  from artifact metadata — `feature_set()==V2R12` → `with_r12_incremental`
  (R12 relation stack), else `with_options(Incremental)`. Full refresh remains the
  reference implementation via `with_options(FullRefresh)` for diagnostics only.
- `search.rs` `evaluate_profiled`: NNUE branch = `nnue.evaluate_full_cp_i32_audited(pos)`
  then mop-up; profile NO LONGER decides material composition. `profile` param kept
  (HCE variants dispatch) until commit 2 removed those variants.
- `uci.rs`: UCI option `NnueMode` REMOVED → `option name Evaluation type combo
  default classical var classical var nnue` (+ existing `EvalFile`).
  `EvalBackendConfig { evaluation: Evaluation, eval_file, model }`;
  `SearchNnueBackend::from_model(model)` derives `r12_incremental` from
  `feature_set()==V2R12`, mode=Incremental. `select_search_nnue_backend` has ONE
  NNUE construction branch. Run loop: `EvalUpdated` → `clear_tt_on_evaluator_change(&tt)`
  (guard.clear() + `info string evaluator changed; transposition table cleared`).
- Tests (release lib 446 PASS at this commit): `s14_full_score_equals_material_for_zero_residual_model`
  (nnue_v2q_runtime tests; uses NEW `synthetic_zero_output_artifact_bytes(fen, mode)`
  — final layer zeroed so raw==0, asserts full==material both POV);
  `s14_evaluator_change_clears_the_transposition_table`; `s14_nnue_model_*` trio
  (absolute unchanged / relative-next-to-exe / missing fail-closed) from the
  relative-`--nnue-model` feature; updated s10d_* tests to `Evaluation`.
- Verification: clippy 0 findings in changed files (pre-existing debt lives in
  nnue_v2q_runtime/bench TEST code — `-D warnings` fails on it historically);
  WSL smoke: option route (Evaluation=nnue+EvalFile=S14) → depth1 cp +684 `f7e6`,
  depth8 cp 695 `f7e6` — IDENTICAL to profile route (same PV), TT-cleared lines
  present; missing model → `Evaluation=nnue requires a loadable EvalFile; refusing
  to search` + bestmove 0000.

### Commit 2 — `da5f2d8` "refactor(entry): converge evaluator entry, collapse SearchProfile surface (49 → 2 variants)"

- `SearchProfile` enum collapsed to EXACTLY `Current` (rollback) + `CurrentFinal`
  (production), `pub(crate)`, at `src/engine/search.rs:70`. PRODUCTION_PROFILE/
  ROLLBACK_PROFILE consts unchanged.
- Deleted: 21 flag fns (uses_nnue_eval/uses_nnue_material_residual/
  uses_nnue_incremental_stack/uses_nnue_r12_incremental_frames/uses_qsearch_delta/
  uses_qsearch_fast_pruning/uses_qsearch_lazy/uses_root_quiet_history/
  uses_root_prev_score/uses_phase_affine_eval/uses_bounded_check2_extension/
  uses_threat_aware_eval/uses_threat_ordering/uses_threat_aware_qsearch/
  uses_forcing_search/eval2_mask/uses_legality_fast/uses_single_buffer_legal/
  uses_single_generation_probe/uses_lmr_null_window/uses_qsearch_pruning) and
  dead helpers (evaluate_phase_affine/evaluate_threat_aware/eval2-mask machinery,
  threat ordering, fast-SEE, M4 wrappers, probe_tt_for_search_exact_depth, …).
  Kept 8 flags (uses_pvs/see/aspiration/lmr/null_move/futility/qsearch_movegen/
  single_evasion_extension) + uses_eval2, shrunk to 2-variant matches.
  `evaluate_profiled` lost its phase-affine/eval2/threat branches. eval.rs helpers
  KEPT (bench eval subcommands still reference them).
- `uci.rs`: `--profile` accepts `current`, `current-final`, and SEVEN legacy NNUE
  names (`current-final-s12`, `current-final-nnue-v2q{,-full}`,
  `current-final-nnue-v2q-material{,-cal-fut,-r12,-r12-inc}`) — all NNUE names map
  to search=CurrentFinal + evaluation=Nnue (require `--nnue-model`); other names
  rejected listing the supported set. `StartupSelection` carries the resolved name
  string so handshake still reports e.g. `info string profile current-final-s12`
  for legacy launches (Arena preset compatibility). Startup model folded into
  `EvalBackendConfig` (ONE evaluator path); "profile-fixed option ignored" special
  cases removed. `startup_profile_name`/help updated. Net −550 lines.
- `bench.rs`: `profile_str`/`--profile` to the same 2+7 set; ablation suite =
  [Current, CurrentFinal]; NNUE block via `for_search`; smoke locks re-anchored to
  CurrentFinal (665/768 nodes). Bench-only historical variant names (M4Reference
  etc.) removed from the enum.
- tests: `s2_profile.rs` (2+7 surface), `search_validation.rs` (production leg =
  current-final; `d10-unique-underpromotion` documented as eval tie-break
  exemption), `m2_4.rs`/`m3_0.rs` node locks re-anchored to rollback profile
  (770/755), `tests/data/search_validation.epd` allows alternate mate-in-2 (f6g6)
  in kqk-mopup. ~44 closed-experiment tests deleted (history in git).
- Verification: fmt clean; clippy 0 errors / 0 new; release lib **394 PASS**;
  integration `cargo test --release --tests` **508 PASS** (22 binaries); end-to-end:
  legacy `--profile current-final-s12 --nnue-model target/release/nnue-s14-datasupply-v5.bin`
  → handshake `profile current-final-s12`/`eval nnue-v2q`/`evalfile nnue-s14-datasupply-v5.bin`,
  blunder position depth1 cp +684 `f7e6`, depth8 `f7e6`; `--profile current-lmr`
  → `startup_error invalid --profile …` listing supported set.
- tool scripts NOT broadly migrated: `tools/**` historical experiment runners may
  reference removed profiles (accepted; git is the record). Active items checked:
  `tools/stage_s14_gui.ps1` (no profile), `tools/s14/*` (legacy s12 name works via
  mapping), `tools/fastchess_profiles.json` (historical data, untouched).

## 2. Architecture after unification (what a fresh agent must know)

- ONE evaluator path: `EvalBackendConfig { evaluation: Evaluation(Classical|Nnue),
  eval_file: String, model: Option<Arc<NnueV2QuantizedModel>> }` in `uci.rs`.
  Startup `--nnue-model <path>` (or legacy NNUE profile names) folds into the same
  config; UCI setoptions mutate it; `select_search_nnue_backend` is the single
  backend builder; `SearchNnueBackend::from_model` derives delivery from metadata.
- Search consumes ONLY `NnueSearchState::evaluate_full_cp_i32_audited` via
  `evaluate_profiled` (search.rs, `negamax` futility site, quiescence stand-pat,
  evasion). Raw exit (`evaluate_raw_cp_i32*`) is for diagnostics/audit only.
- Model metadata API (`nnue_v2q_runtime.rs`): `target_mode()` Cp|MaterialResidual,
  `feature_set()` V2|V2R12 (inputs 22528|23296, max feats/perspective 31|61),
  `head_kind()` LegacyDense|ScreluBuckets, `ft_width()` W128|W256, `l1_backend()`.
  Artifact headers: v1-3=108B, v4=112B (adds feature_set), v5=116B (adds head_kind).
- Material helper: `material_cp_stm(pos)` (nnue_v2q_runtime.rs:314) — STM POV
  (white−black, negated for black), kings 0.
- Synthetic test artifacts (in nnue_v2q_runtime `mod tests`):
  `synthetic_artifact_bytes(fen)` / `_with_mode(fen, mode)` (v3, V2, FT128),
  `synthetic_artifact_bytes_v4_r12(fen, feature_set, inputs)` (v4, MaterialResidual
  hardcoded), NEW `synthetic_zero_output_artifact_bytes(fen, mode)` (final layer
  zeroed → raw≡0). CI cannot see `data/s14/**.bin` (only `.layout.json` is
  tracked) — model-dependent tests must gate on file existence or use synthetics.
- TT: `TranspositionTable::clear()` (tt.rs:276); entries survive `go`/`position`;
  cleared on `ucinewgame`, poison recovery, Hash resize, and NOW evaluator change.

## 3. Frozen decisions / constraints (do NOT violate without user GO)

- No retraining; S14 artifact frozen (`329b7170…`, `data/s14/run/seed-20260908/nnue-s14-datasupply-v5.bin`,
  11,936,916 B — NOT in git). No new SPRT/matches. No production channel switch
  (rollback control-plane gap documented in the promotion closeout). No server
  changes this round; server build predates the refactor.
- No-arg default launch = HCE (classical). NEVER switch the no-arg default to S14
  (user explicit, twice). Full-refresh stays diagnostic-only.
- Naming discipline: canonical names HCE-20260825 / S11-R12 / S12-R0 / S14; bare
  `current-final` defaults to the Arena channel in prose; profiles must carry the
  `engine profile:` prefix. S14 data story: **CP-only** (S12-R0 recipe verbatim,
  data pool 5.0M = 779,590 canonical CP + 4,220,410 Fishtest); result-blend was
  S13 and FAILED — never attribute S14 to dual-signal.
- Production entry convergence is DONE; do not reintroduce per-experiment profile
  branches. If a new experiment needs a different search/eval, put it behind an
  explicit experiment entry/test module, not the production enum.
- TT invalidation on evaluator/model change is a hard requirement for any future
  evaluator-affecting UCI option.

## 4. Gotchas / failed approaches (learned this round, expensive)

- **UCI CLI test harness**: piping `uci/position/go/quit` in one shot sends `quit`
  before the search finishes → engine aborts and returns a fallback bestmove
  (earlier `a7a6` confusion; NOT an engine bug). Hold stdin open (e.g. bash:
  `(printf '…go depth 8\n'; sleep 10; printf 'quit\n') | eureka`) or wait for
  bestmove. The guide's 验证 section and handoff log record this.
- **Windows exe lock**: En Croissant holding `eureka.exe` → `cargo build --release`
  fails "os error 5, access denied" at the final copy. Close the GUI first.
- **WSL /tmp volatility**: the WSL VM restarts between invocations; /tmp content
  (build dirs, binaries) vanishes. Do multi-step work in ONE wsl.exe call or use
  persistent paths; don't background builds that span calls.
- **Build flavors (informational)**: WSL login shell PATH has `~/.local/bin/cc` =
  zig cc (clang 18.1.6/LLD 18) → binaries embed a DT_RUNPATH with the build dir +
  rustup toolchain path + a RANDOM `rustcXXXXXX` token → byte-reproduction is
  impossible beyond a 6-byte runpath delta (same-env rebuild vs server binary
  `dceacfb7…` differed by exactly 6 bytes). Non-login/cloud (gcc + rust-lld 21)
  builds embed no runpath. Do not treat hash mismatch as a defect; check source
  SHA + toolchain + lockfile + UCI identity instead.
- **Clippy debt**: `cargo clippy --all-targets -- -D warnings` fails on PRE-EXISTING
  test-code lints (~61, mostly nnue_v2q_runtime/bench tests). Bar = add no new
  findings; CI/deploy workflow has a documented quality-gate waiver for frozen
  historical commits.
- `git add` on this repo emits CRLF→LF warnings — benign.
- En Croissant engine display name is cached from add-time (may show old
  `+9ef078fd`); cosmetic. Its config model is `path` + `settings` (UCI options),
  no args.

## 5. Exact next steps (in priority order)

1. If continuing verification: run the user's full matrix again on a fresh build —
   HCE / profile-route / option-route on
   `position startpos moves e2e3 e7e6 d1g4 d8e7 g4e6` at depth 6/8/10 (expect
   e7e6 +843 / f7e6 +684 / f7e6 +684 respectively), plus model-switch, NNUE
   off→on, load-failure, and TT-clear checks. All were green at `da5f2d8`.
2. Optional (user-approved ideas, not started): add the blunder position as a
   permanent correctness fixture (needs local gating or a synthetic V2R12
   material-residual artifact); consider teaching `EvalFile` relative paths the
   same exe-dir resolution as `--nnue-model` (small, user floated it).
3. If the user wants S14 on the SERVER: out of scope until the rollback
   control-plane task is done (see promotion closeout); also the server build
   would need the new code (current deployed build predates everything here).
4. Do not resurrect `NnueMode`/NNUE profiles; do not add profile-keyed eval
   semantics back.

## 6. Key references

- Docs: `docs/dev-log/2026-09-10-s14-promotion-closeout.md` (promotion HOLD +
  errata), `docs/dev-log/2026-09-10-s14-promotion-sprt-CHECKPOINT.md` (SPRT
  handoff), `results/s14/s14-en-croissant-guide.md` (GUI guide, current),
  `results/s10/s10-nnue-tryout-guide.md` (S10-era, superseded banner),
  `server-bootstrap-evidence-20260806.md` (byte-reproducibility history),
  `handoff.md` (append-only log, 6 entries for 2026-09-10).
- Code: `src/engine/search.rs` (`evaluate_profiled`, SearchProfile enum :~70),
  `src/engine/nnue_search.rs` (`NnueSearchState`, `for_search`, raw/full evals),
  `src/engine/nnue_v2q_runtime.rs` (`NnueV2QuantizedModel`, metadata, loaders,
  `material_cp_stm`, synthetic builders), `src/uci.rs` (`Evaluation`,
  `EvalBackendConfig`, `SearchNnueBackend::from_model`, `select_search_nnue_backend`,
  `clear_tt_on_evaluator_change`, option emission), `src/engine/tt.rs:276` (clear),
  `tools/stage_s14_gui.ps1` (staging + generated `target\release\EN-CROISSANT-S14.txt`).
- Memory (CodeBuddy project memory): `project_build_identity.md` (build flavors,
  reproducibility, GUI status), `feedback_eureka_version_naming.md` (naming),
  `project_s14_promotion_state.md` (promotion HOLD), `MEMORY.md` index.
