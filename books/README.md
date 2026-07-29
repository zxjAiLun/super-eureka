# Formal opening suites

Formal engine A/B runs use a fixed PGN or EPD suite selected by
`books/manifest.json`. The repository does not vendor a large binary book.
The manifest pins the source repository, source commit, format, expected
content hash, and expected position/depth metadata. `books/cache/` is ignored.

The current default is the official Stockfish `8moves_v3.pgn` suite. It is a
normal 16-ply PGN suite, not a Polyglot runtime book and not the 32-line
protocol smoke fixture. The source is distributed under CC0-1.0 by the
official Stockfish books repository.

The preflight EPD suites are also pinned in the manifest:

* `UHO_4060_v4.epd` for broad opening positions;
* `closedpos.epd` for closed positions;
* `endgames.epd` and `endgames_cdb95105.epd` for endgame positions;
* `stalemates_200d30_v1.epd` for stalemate-sensitive positions.

Acquire and verify all selected suites into the ignored local cache with:

```text
python tools/prepare_books.py --update-manifest
```

The manifest records the SHA-384 of the raw extracted content. The
`upstream_normalized_sri` field records the official Stockfish value after its
line-ending normalization; it is not a hash of the ZIP archive.

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
