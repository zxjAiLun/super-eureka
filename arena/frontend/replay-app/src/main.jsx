import { createRoot } from "react-dom/client";
import App from "./App.jsx";
import "./styles.css";

const rootEl = document.getElementById("replay-root");
if (!rootEl) {
  throw new Error("replay-root element not found");
}

const props = {
  gameId: rootEl.dataset.gameId,
  tournamentId: rootEl.dataset.tournamentId,
  basePath: rootEl.dataset.basePath,
  pairIndex: parseInt(rootEl.dataset.pairIndex || "0", 10),
};

if (!props.gameId || !props.tournamentId) {
  rootEl.innerHTML =
    '<p class="demo-message demo-error">Replay app missing game configuration.</p>';
} else {
  createRoot(rootEl).render(<App {...props} />);
}
