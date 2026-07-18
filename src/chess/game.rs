//! GameState: a `Position` plus its real, chronological UCI history
//! of Zobrist keys (M3.0).
//!
//! `key_history` holds the FEN/startpos root key followed by the key
//! after each applied move. It is the source of truth for M3.1
//! repetition and M3.2 transposition-table keys. `position` and
//! `key_history` are crate-private: callers may only read them through
//! the getters and may only advance them through `apply_legal_move`
//! (verified) or the crate-private `push_known_legal_move` (already
//! verified by the caller). They can never drift apart.

use crate::chess::movegen::generate_legal_moves;
use crate::chess::position::Position;
use crate::chess::types::*;
use crate::chess::zobrist::{recompute_zobrist, ZobristKey};

#[derive(Clone)]
pub struct GameState {
    position: Position,
    key_history: Vec<ZobristKey>,
}

/// Error returned when a caller tries to apply a move that is not in the
/// current position's legal-move list.
#[derive(Clone, Copy, Debug)]
pub enum GameStateError {
    /// The supplied `Move` is not legal in the current position.
    MoveNotLegal(Move),
}

impl GameState {
    /// Fresh game from the standard start position.
    pub fn startpos() -> Self {
        let position = Position::startpos();
        debug_assert_eq!(position.zobrist_key(), recompute_zobrist(&position));
        GameState {
            position,
            key_history: vec![position.zobrist_key()],
        }
    }

    /// Wrap an already-constructed `Position`. The `Position` must have a
    /// consistent `zobrist_key` (verified by `debug_assert`).
    pub fn from_position(position: Position) -> Self {
        debug_assert_eq!(position.zobrist_key(), recompute_zobrist(&position));
        GameState {
            position,
            key_history: vec![position.zobrist_key()],
        }
    }

    /// Read-only access to the current position.
    pub fn position(&self) -> &Position {
        &self.position
    }

    /// The full key history, oldest first, current last.
    pub fn key_history(&self) -> &[ZobristKey] {
        &self.key_history
    }

    /// The Zobrist key of the current position. Equals
    /// `key_history.last()`.
    pub fn current_key(&self) -> ZobristKey {
        self.position.zobrist_key()
    }

    /// Legal moves of the current position. Generated on a *copy* of the
    /// position, so no `&mut Position` is ever exposed.
    pub fn legal_moves(&self) -> Vec<Move> {
        let mut p = self.position;
        generate_legal_moves(&mut p)
    }

    /// Apply `mv` only if it is legal; otherwise leave `self` fully
    /// unchanged.
    pub fn apply_legal_move(&mut self, mv: Move) -> Result<(), GameStateError> {
        if !self.legal_moves().contains(&mv) {
            return Err(GameStateError::MoveNotLegal(mv));
        }
        self.push_known_legal_move(mv);
        Ok(())
    }

    /// Append a move the caller has already verified legal. Updates both
    /// `position` and `key_history` together.
    pub(crate) fn push_known_legal_move(&mut self, mv: Move) {
        self.position.make_move(mv);
        self.key_history.push(self.position.zobrist_key());
    }

    /// Hand the search a `(position, history)` pair. `history` is moved
    /// out so the search gets an owned copy it can extend with its own
    /// `SearchPath` without touching this `GameState`.
    ///
    /// Intentionally unused in the chess-state commit; it is the hand-off
    /// point for the deferred `feat(search)` commit (SearchPath threading),
    /// so it is kept (and allowed as dead code) rather than re-added later.
    #[allow(dead_code)]
    pub(crate) fn into_search_parts(self) -> (Position, Vec<ZobristKey>) {
        (self.position, self.key_history)
    }
}
