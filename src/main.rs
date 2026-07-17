use chess_engine_demo::chess::fen;
use chess_engine_demo::chess::types::START_FEN;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // `cargo run -- perft <depth> [<fen>]` runs a Perft node count
    // (Phase-2 verification). Example: `cargo run -- perft 4`.
    if args.len() >= 2 && args[1] == "perft" {
        let depth: u32 = args.get(2).and_then(|s| s.parse::<u32>().ok()).unwrap_or(4);
        let fen_str = args
            .get(3)
            .cloned()
            .unwrap_or_else(|| START_FEN.to_string());

        let mut pos = match fen::parse_fen(&fen_str) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("invalid FEN: {}", e);
                std::process::exit(1);
            }
        };

        let start = std::time::Instant::now();
        let nodes = pos.perft(depth);
        let elapsed = start.elapsed();
        println!("perft({}) = {}  ({:?})", depth, nodes, elapsed);
        if elapsed.as_secs_f64() > 0.0 {
            println!(
                "nps = {:.0}",
                nodes as f64 / elapsed.as_secs_f64().max(1e-9)
            );
        }
        return;
    }

    // Otherwise, run the UCI protocol loop.
    chess_engine_demo::uci::run();
}
