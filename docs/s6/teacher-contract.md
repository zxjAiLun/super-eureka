# S6.0 Teacher contract

Frozen teacher artifact (never "whatever stockfish is installed"):

```text
engine:        Stockfish 18 (UCI id "Stockfish 18")
binary SHA-256: 6b087694916228c905a5e14db74cca8c7e5643602226af1fa5d42353c455b9f9
platform:      linux-x86_64 (avx2)
artifact:      /opt/chessarena/builds/stockfish-18-avx2-linux-x86_64/stockfish
               (Arena registered build; manifest recorded 2026-08-07)
```

## Labeling contract

```text
Threads        = 1
Hash           = 64 MB
MultiPV        = 1
UCI_ShowWDL    = true
Syzygy         = disabled (default; no SyzygyPath set)
labeling mode  = go nodes 16384 (fixed nodes, not time)
per position   = ucinewgame + position fen <fen> + go nodes 16384
score/wdl      = side-to-move perspective (UCI convention)
```

## Determinism / audit

The audit is ALWAYS ON (there is no `--no-audit` bypass) and uses one of two
explicit modes:

```text
fresh-second-pass (no stored labels):
  after ALL records are labeled, the Teacher/Stockfish process is destroyed
  and a NEW process independently re-labels the deterministic first 1,000
  records; every field must match exactly.

vs-stored (labels.jsonl exists):
  the first 1,000 records are re-labeled and compared field-by-field against
  the stored labels (stored labels reject duplicate position_id).

compared fields: position_id, teacher_cp_stm (incl. null), teacher_mate,
  teacher_bestmove, teacher_wdl_stm, nodes, plus teacher identity recorded in
  the manifest.
```

PASS (0 mismatches, sample_count=1000) plus `audit_mode`, the sample
position-id SHA-256, checked, and mismatches are recorded in
teacher_manifest.json. Publishing is fail-closed: labels.jsonl and
teacher_manifest.json are written to temp files first and only published
after full labeling + a passing audit, with the manifest renamed LAST as the
commit point. Any mismatch means TEACHER_PIPELINE_INVALID and nothing is
published.

## Driver quirks (tools/s6/label_teacher.py, empirical, verified)

- the UCI handshake must be requested with exactly ONE `uci` (a second one
  poisons subsequent `go` output under the wsl.exe console bridge);
- stderr must be merged into stdout (DEVNULL deadlocks `go`);
- `isready` must be answered before any `setoption`;
- on teacher process death the driver respawns and retries the position once
  (failures are recorded, never silently dropped).
