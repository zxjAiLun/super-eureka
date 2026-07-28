# S2 Tournament-Selectable Candidate Profile

Status: `INTERFACE ONLY — no Elo or production decision`

The UCI executable now accepts one fixed cumulative search profile at process
startup. The default remains the approved production `Current` profile:

```text
target/release/chess-engine-demo.exe
target/release/chess-engine-demo.exe --profile current
```

Candidate launches use the same binary with an explicit argument:

```text
target/release/chess-engine-demo.exe --profile current-aspiration
target/release/chess-engine-demo.exe --profile current-aspiration-lmr
target/release/chess-engine-demo.exe --profile current-aspiration-lmr-futility
target/release/chess-engine-demo.exe --profile current-aspiration-lmr-futility-see
```

The selected profile is fixed for the lifetime of the process. This keeps
the UCI stream unchanged while allowing fastchess/OpenBench to launch a
baseline and a candidate from the same executable. The null probe and all
non-cumulative experimental profiles are intentionally not selectable here.

Invalid or duplicate startup arguments terminate before the UCI handshake
with a non-zero exit status. `--help` lists the accepted profiles. This
interface does not enable any candidate in `Current`, and it does not turn
bench observations into Elo or SPRT decisions.
