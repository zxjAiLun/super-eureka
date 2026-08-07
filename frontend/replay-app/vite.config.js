import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { fileURLToPath, URL } from "node:url";

// The built island is served by FastAPI from {base_path}/static/replay-app/.
// Production and tests both use base_path=/chessarena.
const staticOut = fileURLToPath(
  new URL("../../chessarena/static/replay-app", import.meta.url)
);

export default defineConfig({
  base: "/chessarena/static/replay-app/",
  plugins: [react()],
  build: {
    outDir: staticOut,
    emptyOutDir: true,
  },
});
