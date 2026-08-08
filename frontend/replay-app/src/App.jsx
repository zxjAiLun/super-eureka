import { useEffect, useMemo, useState } from "react";
import { Chessboard } from "react-chessboard";
import { Chess } from "chess.js";

const START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

// Tournaments start from a non-initial FEN when an opening book is used, and
// chess.js loadPgn() honors that [FEN] header.  Navigation must therefore
// replay from the same start position, not from the standard array.
function pgnStartFen(pgn) {
  const m = pgn.match(/^\[FEN "([^"]+)"\]/m);
  return m ? m[1] : null;
}

// Friendly time-control labels for the badge (API returns the config key).
const TC_LABELS = {
  bullet_1_0: "1+0",
  blitz_3_2: "3+2",
  rapid_5_3: "5+3",
};

function useReplay({ gameId, tournamentId, basePath }) {
  const [status, setStatus] = useState("loading");
  const [error, setError] = useState(null);
  const [moves, setMoves] = useState([]);
  const [meta, setMeta] = useState(null);
  const [pgn, setPgn] = useState("");
  const [startFen, setStartFen] = useState(START_FEN);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [matchRes, pgnRes] = await Promise.all([
          fetch(`${basePath}/public-api/v1/matches/${tournamentId}`),
          fetch(`${basePath}/public-api/v1/games/${gameId}/pgn`),
        ]);
        if (!matchRes.ok || !pgnRes.ok) {
          throw new Error("failed to load game data");
        }
        const match = await matchRes.json();
        const pgnText = await pgnRes.text();
        const game = match.games.find((g) => g.id === gameId);
        if (!game) {
          throw new Error("game not found in match");
        }
        const chess = new Chess();
        chess.loadPgn(pgnText);
        const ms = chess.history({ verbose: true });
        if (cancelled) return;
        setMoves(ms);
        setPgn(pgnText);
        setStartFen(pgnStartFen(pgnText) || START_FEN);
        setMeta({ game, timeControl: match.time_control, matchName: match.name });
        setStatus("ready");
      } catch (e) {
        if (!cancelled) {
          setStatus("error");
          setError(e.message);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [gameId, tournamentId, basePath]);

  return { status, error, moves, meta, pgn, startFen };
}

export default function App({ gameId, tournamentId, basePath, pairIndex }) {
  const { status, error, moves, meta, pgn, startFen } = useReplay({
    gameId,
    tournamentId,
    basePath,
  });
  const [ply, setPly] = useState(0);

  useEffect(() => {
    setPly(0);
  }, [status]);

  const fen = useMemo(() => {
    if (moves.length === 0) return startFen;
    const c = new Chess(startFen);
    for (let i = 0; i < ply; i++) {
      // Apply by source/target squares (verbose object), not by SAN:
      // re-parsing SAN can hit disambiguation errors on real tournament
      // PGNs ("Invalid move: Bxc6").
      const m = moves[i];
      c.move({ from: m.from, to: m.to, promotion: m.promotion });
    }
    return c.fen();
  }, [moves, ply, startFen]);

  useEffect(() => {
    if (status !== "ready") return undefined;
    const handler = (e) => {
      if (e.key === "ArrowRight") {
        setPly((p) => Math.min(moves.length, p + 1));
      } else if (e.key === "ArrowLeft") {
        setPly((p) => Math.max(0, p - 1));
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [status, moves.length]);

  // Keep the active move visible while navigating a long game.
  useEffect(() => {
    if (status !== "ready") return undefined;
    const active = document.querySelector(".move.active");
    if (active) {
      active.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
    return undefined;
  }, [ply, status]);

  if (status === "loading") {
    return <div className="demo-message">Loading game…</div>;
  }
  if (status === "error") {
    return (
      <div className="demo-message demo-error">
        Failed to load replay: {error}
      </div>
    );
  }

  const { game, timeControl, matchName } = meta;
  const rows = [];
  for (let i = 0; i < moves.length; i += 2) {
    rows.push({ n: i / 2 + 1, white: moves[i], black: moves[i + 1] });
  }

  return (
    <div className="replay">
      <div className="replay-board-col">
        {/* White at the bottom: react-chessboard's default white orientation
            places White at the bottom, so Black card sits above the board. */}
        <div className="player-card top">
          <span className="color-dot black" />
          <span className="player-name">{game.black_engine}</span>
        </div>
        <div className="board-wrap" data-fen={fen}>
          <Chessboard options={{ position: fen, allowDragging: false }} />
        </div>
        <div className="player-card bottom">
          <span className="player-name">{game.white_engine}</span>
          <span className="color-dot white" />
        </div>
        <div className="controls">
          <button
            type="button"
            onClick={() => setPly(0)}
            disabled={ply === 0}
          >
            first
          </button>
          <button
            type="button"
            aria-label="Previous move"
            onClick={() => setPly((p) => Math.max(0, p - 1))}
            disabled={ply === 0}
          >
            ←
          </button>
          <span className="ply-indicator">
            {ply}/{moves.length}
          </span>
          <button
            type="button"
            aria-label="Next move"
            onClick={() => setPly((p) => Math.min(moves.length, p + 1))}
            disabled={ply === moves.length}
          >
            →
          </button>
          <button
            type="button"
            onClick={() => setPly(moves.length)}
            disabled={ply === moves.length}
          >
            last
          </button>
        </div>
      </div>

      <div className="replay-side-col">
        <div className="badges">
          <span className="badge">Game {game.game_number}</span>
          <span className="badge">Pair {pairIndex + 1}</span>
          <span className="badge">{TC_LABELS[timeControl] || timeControl}</span>
          <span className="badge badge-result">
            {game.result || "?"}
            {game.termination ? ` · ${game.termination}` : ""}
          </span>
        </div>

        <div className="moves-list">
          {rows.map((r) => (
            <div className="move-row" key={r.n}>
              <span className="move-n">{r.n}.</span>
              <button
                type="button"
                className={"move" + (ply === r.n * 2 - 1 ? " active" : "")}
                onClick={() => setPly(r.n * 2 - 1)}
              >
                {r.white.san}
              </button>
              {r.black && (
                <button
                  type="button"
                  className={"move" + (ply === r.n * 2 ? " active" : "")}
                  onClick={() => setPly(r.n * 2)}
                >
                  {r.black.san}
                </button>
              )}
            </div>
          ))}
        </div>

        <div className="actions">
          <a
            href={`${basePath}/public-api/v1/games/${gameId}/pgn`}
            className="action-link"
          >
            Download PGN
          </a>
          <a
            href={`https://lichess.org/paste?pgn=${encodeURIComponent(pgn)}`}
            target="_blank"
            rel="noreferrer"
            className="action-link"
          >
            Import into Lichess
          </a>
        </div>

        <div className="demo-note">{matchName}</div>
      </div>
    </div>
  );
}
