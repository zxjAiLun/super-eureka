# ChessArena v1

Remote-managed cutechess engine match system for the ChessEngineDemo Rust
engine. Runs on the `pearllover.site` server behind HTTPS with no routine SSH
for match management.

- Deploy URL: `https://pearllover.site/chessarena/`
- Admin UI: `https://pearllover.site/chessarena/admin/`
- API: `https://pearllover.site/chessarena/api/v1/`

See `deploy/bootstrap.md` for the one-time server setup and
`deploy/chessarena.env` for the runtime configuration.

## What v1 does

- Registers immutable engine builds (read-only `engine` + `manifest.json`).
- Registers validated opening sets (EPD, unique non-terminal positions).
- Runs one cutechess pair at a time: each opening plays exactly two games
  with strict color swap (`-rounds 2 -repeat 2` on a single-position file).
- Fixed time controls only: `bullet_1_0` (60), `blitz_3_2` (180+2),
  `rapid_5_3` (300+3). Hash 32 MB, concurrency 1, 1 thread.
- Pause (after the current pair), resume, cancel, force-cancel (requires
  `confirm=true`).
- Independent verifier parses `match.pgn`, replays every move for legality,
  enforces strict color swap / identical opening / matching TimeControl,
  cross-checks the cutechess score line, whitelists stderr, and pins engine /
  opening / command provenance. A failed pair fails the whole tournament and
  never counts toward the score.
- Crash-safe recovery: on worker restart a RUNNING pair is either verified as
  complete or re-run from scratch (old attempt directories are preserved,
  missing single games are never patched in).
- Every completed tournament produces `combined.pgn`, `summary.json` and
  `artifact-manifest.json` (SHA-256 of every result file), all downloadable.
- The `events` table records every lifecycle transition from day one so a v2
  live-spectating layer (SSE, UCI telemetry) can be added without schema
  changes.

## What v1 deliberately does NOT do

Live board / depth / score / PV streaming, WebSocket, multi-worker or
distributed scheduling, arbitrary cutechess parameters, binary uploads
through the public API, Docker/Kubernetes/Redis/Celery, React. The admin UI
is Jinja2 + HTMX.

## Local development

```bash
cd arena
python -m pip install -r requirements-dev.txt
python -m pytest tests/ -q
```

Tests use a fake `cutechess-cli` fixture
(`tests/fixtures/fake_cutechess.py`) — no real cutechess or engine needed.
Engine binaries in the test builds are dummy files whose SHA-256 is recorded
and re-checked, exercising the real provenance path.

### Running the API locally

```bash
export ARENA_DB_URL=sqlite:////tmp/arena.db
export ARENA_RUN_ROOT=/tmp/arena-runs
export ARENA_BUILD_ROOT=/tmp/arena-builds
export ARENA_OPENING_ROOT=/tmp/arena-openings
export ARENA_CUTECHESS=/usr/bin/cutechess-cli
export ARENA_BASE_PATH=/chessarena
python -m alembic upgrade head
python -m uvicorn chessarena.main:app --host 127.0.0.1 --port 8787
```

In another terminal:

```bash
python -m chessarena.worker
```

Then open `http://127.0.0.1:8787/chessarena/admin/`.

## Deployment

- `deploy/chessarena-api.service`, `deploy/chessarena-worker.service`:
  systemd units (API `MemoryMax=400M`, worker `MemoryMax=1700M`).
- `deploy/nginx-chessarena.conf`: reverse-proxy fragment; whole
  `/chessarena/` behind Basic Auth in v1.
- `.github/workflows/deploy-arena.yml`: test + deploy + migrate + atomic
  release switch + health check with rollback.
- `.github/workflows/deploy-engine-build.yml`: build a pinned git ref through
  the full Rust gate, probe UCI identity, verify SHA server-side, install the
  immutable build, register it in the database.

## Management commands

```bash
python -m chessarena.admin disk-usage
python -m chessarena.admin archive-tournament <tournament_id>   # tar.zst
```
