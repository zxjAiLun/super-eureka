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

## Determinism

The audit re-labels the first 1,000 dataset positions and requires exact
bestmove / score cp / mate / wdl equality. PASS (0 mismatches) is recorded
in teacher_manifest.json. Any future re-label must re-run the audit; a
mismatch means TEACHER_PIPELINE_INVALID.

## Driver quirks (tools/s6/label_teacher.py, empirical, verified)

- the UCI handshake must be requested with exactly ONE `uci` (a second one
  poisons subsequent `go` output under the wsl.exe console bridge);
- stderr must be merged into stdout (DEVNULL deadlocks `go`);
- `isready` must be answered before any `setoption`;
- on teacher process death the driver respawns and retries the position once
  (failures are recorded, never silently dropped).
