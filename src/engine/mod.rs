pub mod eval;
pub mod search;
pub mod tt;
pub mod time;

pub use eval::evaluate;
pub use search::{search_best_move, negamax, MATE};
