# Formal opening suites

Formal engine A/B runs use a fixed PGN or EPD suite selected by
`books/manifest.json`. The repository does not vendor a large binary book.
The manifest pins the source repository, source commit, format, expected
content hash, and expected position/depth metadata. `books/cache/` is ignored.

The current default is the official Stockfish `8moves_v3.pgn` suite. It is a
normal 16-ply PGN suite, not a Polyglot runtime book and not the 32-line
protocol smoke fixture. The source is distributed under CC0-1.0 by the
official Stockfish books repository.

Download and verify it through the Fastchess wrapper:

```text
python tools/run_fastchess.py --help
python tools/run_fastchess.py \
  --fastchess path/to/fastchess \
  --engine-a target/release/chess-engine-demo.exe \
  --engine-b target/release/chess-engine-demo.exe \
  --sha-a <git-sha> --sha-b <git-sha> \
  --download-book --dry-run
```

Do not give the two engines independent opening books. Fastchess selects one
opening per round and repeats it with colors swapped; the runner records the
book path and verified content hash in the run manifest.
