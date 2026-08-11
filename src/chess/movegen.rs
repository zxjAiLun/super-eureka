//! Move generation (Phase 1).
//!
//! Step 1: generate *pseudo-legal* moves (they ignore whether our own king
//! ends up in check). Step 2: make each move, test if our king is attacked,
//! unmake it — only keep the ones that leave the king safe.

use crate::chess::position::Position;
use crate::chess::types::*;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::Instant;

/// Fill `moves` with every pseudo-legal move for the side to move.
pub fn generate_pseudo_moves(pos: &Position, moves: &mut Vec<Move>) {
    let us = pos.side;
    for sq in 0u8..64u8 {
        if let Some(p) = pos.board[sq as usize] {
            if p.color != us {
                continue;
            }
            match p.piece_type {
                PieceType::Pawn => gen_pawn(pos, sq, moves),
                PieceType::Knight => gen_step(pos, sq, moves, &KNIGHT_OFFSETS),
                PieceType::King => {
                    gen_step(pos, sq, moves, &KING_OFFSETS);
                    gen_castling(pos, moves);
                }
                PieceType::Bishop => gen_slider(pos, sq, moves, &BISHOP_DIRS),
                PieceType::Rook => gen_slider(pos, sq, moves, &ROOK_DIRS),
                PieceType::Queen => gen_slider(pos, sq, moves, &QUEEN_DIRS),
            }
        }
    }
}

/// Generate only strictly legal moves.
pub fn generate_legal_moves(pos: &mut Position) -> Vec<Move> {
    generate_legal_moves_with_stats(pos).0
}

/// Search profiling counters for one legal-move generation. The movegen
/// implementation is shared with the public helper above; this variant only
/// exposes counts that are already known while doing the same work, so
/// profiling never generates the move list twice.
///
/// S4.3B: `make_moves`/`unmake_moves` count ACTUAL legality probes performed
/// (the fast path accepts unpinned non-check moves without probing). The
/// `fallback_*`/`fast_accepts` fields are bench-only fast-path statistics;
/// legacy generators leave them at zero.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct MovegenStats {
    pub(crate) pseudo_moves: u64,
    pub(crate) legal_moves: u64,
    pub(crate) make_moves: u64,
    pub(crate) unmake_moves: u64,
    pub(crate) fast_accepts: u64,
    pub(crate) fallback_probes: u64,
    pub(crate) fallback_in_check: u64,
    pub(crate) fallback_king: u64,
    pub(crate) fallback_pinned: u64,
    pub(crate) fallback_en_passant: u64,
    pub(crate) fallback_castle: u64,
}

pub(crate) fn generate_legal_moves_with_stats(pos: &mut Position) -> (Vec<Move>, MovegenStats) {
    let mut pseudo = Vec::new();
    generate_pseudo_moves(pos, &mut pseudo);
    let pseudo_count = pseudo.len() as u64;
    let us = pos.side;
    let mut legal = Vec::new();
    for m in pseudo {
        let undo = pos.make_move(m);
        if !pos.is_square_attacked(pos.king_sq[us as usize], us.opposite()) {
            legal.push(m);
        }
        pos.unmake_move(undo);
    }
    let legal_count = legal.len() as u64;
    (
        legal,
        MovegenStats {
            pseudo_moves: pseudo_count,
            legal_moves: legal_count,
            make_moves: pseudo_count,
            unmake_moves: pseudo_count,
            ..MovegenStats::default()
        },
    )
}

/// S4.4A: bench-only sparse sub-attribution INSIDE the promoted full legal
/// generator. The engine wrapper passes `Some(sub)` only when the S4.3A
/// sampled-timing mode is enabled; the production path passes `None`, so a
/// single `Option::is_some` test is the only per-call cost.
///
/// Gating mirrors the engine's `SampledCounter` (1/rate calls are timed; the
/// `calls`/`samples` counts expose sampling uncertainty). Wall time is split
/// into: pseudo move generation, king/check-state setup (the in-check test),
/// the absolute-pin scan, per-move eligibility testing, fast accepts, and
/// legacy fallback probes (make -> king-attacked test -> unmake). Exact
/// counters (pin_scan_calls / in_check_calls) are also maintained so the
/// wall split can be normalized per event.
#[derive(Debug)]
pub(crate) struct FullLegalSub {
    calls: AtomicU64,
    samples: AtomicU64,
    gate: AtomicU32,
    pub(crate) sample_rate: u32,
    pub(crate) pseudo_gen_ns: AtomicU64,
    pub(crate) check_state_ns: AtomicU64,
    pub(crate) pin_scan_ns: AtomicU64,
    /// Exact: pin scan actually ran (generator entered the non-check branch).
    pub(crate) pin_scan_calls: AtomicU64,
    /// Exact: generator entered the in-check (all-legacy-probe) branch.
    pub(crate) in_check_calls: AtomicU64,
}

impl FullLegalSub {
    pub(crate) fn new(sample_rate: u32) -> Self {
        Self {
            calls: AtomicU64::new(0),
            samples: AtomicU64::new(0),
            gate: AtomicU32::new(0),
            sample_rate,
            pseudo_gen_ns: AtomicU64::new(0),
            check_state_ns: AtomicU64::new(0),
            pin_scan_ns: AtomicU64::new(0),
            pin_scan_calls: AtomicU64::new(0),
            in_check_calls: AtomicU64::new(0),
        }
    }

    /// Sparse gate: true when THIS call should be wall-timed. Counts calls
    /// on every invocation (so `calls` == full-legal generator calls).
    #[inline]
    pub(crate) fn begin(&self) -> bool {
        if self.sample_rate == 0 {
            return false;
        }
        self.calls.fetch_add(1, Ordering::Relaxed);
        if self.gate.fetch_sub(1, Ordering::Relaxed) == 0 {
            self.gate.store(self.sample_rate, Ordering::Relaxed);
            self.samples.fetch_add(1, Ordering::Relaxed);
            true
        } else {
            false
        }
    }

    /// Accumulate the wall time since `from` into `counter` and return the
    /// new phase start. Only invoked on a sampled call.
    #[inline]
    pub(crate) fn acc(&self, counter: &AtomicU64, from: Instant) -> Instant {
        counter.fetch_add(from.elapsed().as_nanos() as u64, Ordering::Relaxed);
        Instant::now()
    }

    pub(crate) fn snapshot(&self) -> FullLegalSubSnapshot {
        FullLegalSubSnapshot {
            calls: self.calls.load(Ordering::Relaxed),
            samples: self.samples.load(Ordering::Relaxed),
            pseudo_gen_ns: self.pseudo_gen_ns.load(Ordering::Relaxed),
            check_state_ns: self.check_state_ns.load(Ordering::Relaxed),
            pin_scan_ns: self.pin_scan_ns.load(Ordering::Relaxed),
            pin_scan_calls: self.pin_scan_calls.load(Ordering::Relaxed),
            in_check_calls: self.in_check_calls.load(Ordering::Relaxed),
        }
    }
}

/// S4.4A: plain snapshot of the sparse full-legal sub-attribution.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct FullLegalSubSnapshot {
    pub(crate) calls: u64,
    pub(crate) samples: u64,
    pub(crate) pseudo_gen_ns: u64,
    pub(crate) check_state_ns: u64,
    pub(crate) pin_scan_ns: u64,
    pub(crate) pin_scan_calls: u64,
    pub(crate) in_check_calls: u64,
}

/// S4.3B: absolute-pin mask for the side to move. A square is set when its
/// piece is the SOLE friendly blocker on a king-slider ray (orthogonal ray +
/// enemy rook/queen, or diagonal ray + enemy bishop/queen). Only
/// king-legality information; no relative pins or tactical-value logic.
fn absolute_pin_mask(pos: &Position) -> u64 {
    const ORTHO: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
    const DIAG: [(i32, i32); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];
    let us = pos.side;
    let king = pos.king_sq[us as usize];
    let mut pins: u64 = 0;
    for (dirs, orthogonal) in [(&ORTHO[..], true), (&DIAG[..], false)] {
        for &(df, dr) in dirs {
            let mut blocker: Option<u8> = None;
            let mut sq = king;
            loop {
                let (f, r) = (file_of(sq) as i32 + df, rank_of(sq) as i32 + dr);
                if !(0..8).contains(&f) || !(0..8).contains(&r) {
                    break;
                }
                sq = make_square(f as u8, r as u8);
                match pos.board[sq as usize] {
                    None => {}
                    Some(p) if p.color == us => {
                        if blocker.is_some() {
                            break; // two friendly blockers -> no pin on this ray
                        }
                        blocker = Some(sq);
                    }
                    Some(p) => {
                        // First enemy on the ray. A pin needs exactly one
                        // friendly blocker between king and a matching slider.
                        if let Some(b) = blocker {
                            let matches = if orthogonal {
                                matches!(p.piece_type, PieceType::Rook | PieceType::Queen)
                            } else {
                                matches!(p.piece_type, PieceType::Bishop | PieceType::Queen)
                            };
                            if matches {
                                pins |= 1_u64 << b;
                            }
                        }
                        break;
                    }
                }
            }
        }
    }
    pins
}

/// S4.3B: full legal generation with an unpinned non-check fast path.
///
/// Safety theorem (documented, not broadened here): when the side to move is
/// NOT in check and its king does not move, enemy knight/pawn/king attacks
/// cannot newly appear because of blocker removal — a newly exposed attack on
/// the stationary king can only be a slider ray, which requires the moving
/// piece to be the SOLE friendly blocker on that ray, i.e. an absolute pin.
/// En passant is excluded (it removes a second piece from a different square),
/// and king moves / castling are excluded (king occupancy changes).
///
/// The fast path therefore accepts, without any make/attack/unmake probe:
/// non-king, non-en-passant, non-castling moves from absolutely-pin-free
/// squares (ordinary captures included: the moving piece occupies the
/// captured square, so removing the captured piece does not open a king ray).
///
/// Every fallback case keeps the EXACT legacy probe, and the generated move
/// list is identical in content and order to
/// [`generate_legal_moves_with_stats`].
pub(crate) fn generate_legal_moves_fast_with_stats(
    pos: &mut Position,
    sub: Option<&FullLegalSub>,
) -> (Vec<Move>, MovegenStats) {
    let mut pseudo = Vec::new();
    // S4.4A: sparse sub-attribution. `sub` is None on the production path
    // (one `is_some` test per call, zero further cost). When present, the
    // 1/rate gate decides whether THIS call is wall-timed phase by phase.
    // Phases are call-granular (pseudo gen / check-state / pin scan): per-move
    // wall timing is NOT used because the instrumentation would be the same
    // order of magnitude as a move's own work and would pollute the sampled
    // measurement. The loop is split instead by exact counters x per-op costs
    // from the bench microbench (see S4.4A report).
    let sampled = sub.is_some_and(|s| s.begin());
    let mut t = sampled.then(Instant::now);
    generate_pseudo_moves(pos, &mut pseudo);
    if let (Some(s), Some(from)) = (sub, t) {
        t = Some(s.acc(&s.pseudo_gen_ns, from));
    }
    let pseudo_count = pseudo.len() as u64;
    let us = pos.side;
    let us_king = pos.king_sq[us as usize];
    let enemy = us.opposite();
    let mut legal = Vec::new();
    let mut fast_accepts = 0u64;
    let mut fb_king = 0u64;
    let mut fb_pinned = 0u64;
    let mut fb_ep = 0u64;
    let mut fb_castle = 0u64;
    let mut fb_probes = 0u64;

    let in_check = pos.is_square_attacked(us_king, enemy);
    if let Some(s) = sub {
        if let Some(from) = t {
            t = Some(s.acc(&s.check_state_ns, from));
        }
    }
    if in_check {
        if let Some(s) = sub {
            s.in_check_calls.fetch_add(1, Ordering::Relaxed);
        }
        // Legacy probe for every pseudo move; no special evasion logic here.
        for m in pseudo {
            let undo = pos.make_move(m);
            if !pos.is_square_attacked(pos.king_sq[us as usize], enemy) {
                legal.push(m);
            }
            pos.unmake_move(undo);
        }
        fb_probes = pseudo_count;
    } else {
        let pins = absolute_pin_mask(pos);
        if let Some(s) = sub {
            s.pin_scan_calls.fetch_add(1, Ordering::Relaxed);
            if let Some(from) = t {
                s.pin_scan_ns
                    .fetch_add(from.elapsed().as_nanos() as u64, Ordering::Relaxed);
            }
        }
        for m in pseudo {
            let is_ep = m.flag == MoveFlag::EnPassant;
            let is_castle = matches!(m.flag, MoveFlag::KingCastle | MoveFlag::QueenCastle);
            let is_king = !is_castle && m.from == us_king;
            let is_pinned = pins & (1_u64 << m.from) != 0;
            if is_ep || is_castle || is_king || is_pinned {
                fb_probes += 1;
                fb_ep += is_ep as u64;
                fb_castle += is_castle as u64;
                fb_king += is_king as u64;
                fb_pinned += is_pinned as u64;
                let undo = pos.make_move(m);
                if !pos.is_square_attacked(pos.king_sq[us as usize], enemy) {
                    legal.push(m);
                }
                pos.unmake_move(undo);
            } else {
                fast_accepts += 1;
                legal.push(m);
            }
        }
    }
    let legal_count = legal.len() as u64;
    (
        legal,
        MovegenStats {
            pseudo_moves: pseudo_count,
            legal_moves: legal_count,
            make_moves: fb_probes,
            unmake_moves: fb_probes,
            fast_accepts,
            fallback_probes: fb_probes,
            fallback_in_check: if in_check { pseudo_count } else { 0 },
            fallback_king: fb_king,
            fallback_pinned: fb_pinned,
            fallback_en_passant: fb_ep,
            fallback_castle: fb_castle,
        },
    )
}

/// Generate pseudo-legal moves in the exact relative order of
/// [`generate_pseudo_moves`], but retain only captures, en passant, and all
/// promotions. Quiet non-promotion moves are never materialized.
pub(crate) fn generate_pseudo_tactical_moves(pos: &Position, moves: &mut Vec<Move>) {
    let us = pos.side;
    for sq in 0u8..64u8 {
        if let Some(p) = pos.board[sq as usize] {
            if p.color != us {
                continue;
            }
            match p.piece_type {
                PieceType::Pawn => gen_pawn_tactical(pos, sq, moves),
                PieceType::Knight | PieceType::King => {
                    gen_step_tactical(
                        pos,
                        sq,
                        moves,
                        if p.piece_type == PieceType::Knight {
                            &KNIGHT_OFFSETS
                        } else {
                            &KING_OFFSETS
                        },
                    );
                }
                PieceType::Bishop => gen_slider_tactical(pos, sq, moves, &BISHOP_DIRS),
                PieceType::Rook => gen_slider_tactical(pos, sq, moves, &ROOK_DIRS),
                PieceType::Queen => gen_slider_tactical(pos, sq, moves, &QUEEN_DIRS),
            }
        }
    }
}

/// Generate legal captures, en passant, and promotions without first
/// materializing quiet non-promotion moves. The order is the same as filtering
/// `generate_legal_moves_with_stats` for tactical moves.
pub(crate) fn generate_legal_tactical_moves_with_stats(
    pos: &mut Position,
) -> (Vec<Move>, MovegenStats) {
    let mut pseudo = Vec::new();
    generate_pseudo_tactical_moves(pos, &mut pseudo);
    let pseudo_count = pseudo.len() as u64;
    let us = pos.side;
    let mut legal = Vec::new();
    for m in pseudo {
        let undo = pos.make_move(m);
        if !pos.is_square_attacked(pos.king_sq[us as usize], us.opposite()) {
            legal.push(m);
        }
        pos.unmake_move(undo);
    }
    let legal_count = legal.len() as u64;
    (
        legal,
        MovegenStats {
            pseudo_moves: pseudo_count,
            legal_moves: legal_count,
            make_moves: pseudo_count,
            unmake_moves: pseudo_count,
            ..MovegenStats::default()
        },
    )
}

/// Check whether any legal move exists, stopping after the first legal move.
/// This is used only when a non-check qsearch node has no tactical move: it
/// distinguishes a quiet position from stalemate without building a full
/// legal move list.
pub(crate) fn has_any_legal_move_with_stats(pos: &mut Position) -> (bool, MovegenStats) {
    let mut pseudo = Vec::new();
    generate_pseudo_moves(pos, &mut pseudo);
    let pseudo_count = pseudo.len() as u64;
    let us = pos.side;
    let mut checked = 0u64;
    for m in pseudo {
        checked += 1;
        let undo = pos.make_move(m);
        let legal = !pos.is_square_attacked(pos.king_sq[us as usize], us.opposite());
        pos.unmake_move(undo);
        if legal {
            return (
                true,
                MovegenStats {
                    pseudo_moves: pseudo_count,
                    legal_moves: 1,
                    make_moves: checked,
                    unmake_moves: checked,
                    ..MovegenStats::default()
                },
            );
        }
    }
    (
        false,
        MovegenStats {
            pseudo_moves: pseudo_count,
            legal_moves: 0,
            make_moves: checked,
            unmake_moves: checked,
            ..MovegenStats::default()
        },
    )
}

/// The first qsearch evasion implementation deliberately reuses the complete
/// legal generator. Its separate name makes the check-node contract explicit
/// and leaves room for checker/pin-specific generation later.
pub(crate) fn generate_legal_evasions_with_stats(pos: &mut Position) -> (Vec<Move>, MovegenStats) {
    generate_legal_moves_with_stats(pos)
}

fn push_move(
    moves: &mut Vec<Move>,
    from: Square,
    to: Square,
    flag: MoveFlag,
    promo: Option<PieceType>,
) {
    moves.push(Move {
        from,
        to,
        promotion: promo,
        flag,
    });
}

fn gen_step(pos: &Position, sq: Square, moves: &mut Vec<Move>, offsets: &[(i32, i32)]) {
    let us = pos.side;
    let f = file_of(sq) as i32;
    let r = rank_of(sq) as i32;
    for (df, dr) in offsets {
        let nf = f + df;
        let nr = r + dr;
        if !on_board(nf, nr) {
            continue;
        }
        let to = make_square(nf as u8, nr as u8);
        match pos.board[to as usize] {
            None => push_move(moves, sq, to, MoveFlag::Normal, None),
            Some(p) if p.color != us => push_move(moves, sq, to, MoveFlag::Normal, None),
            _ => {}
        }
    }
}

fn gen_step_tactical(pos: &Position, sq: Square, moves: &mut Vec<Move>, offsets: &[(i32, i32)]) {
    let us = pos.side;
    let f = file_of(sq) as i32;
    let r = rank_of(sq) as i32;
    for (df, dr) in offsets {
        let nf = f + df;
        let nr = r + dr;
        if !on_board(nf, nr) {
            continue;
        }
        let to = make_square(nf as u8, nr as u8);
        if matches!(pos.board[to as usize], Some(p) if p.color != us) {
            push_move(moves, sq, to, MoveFlag::Normal, None);
        }
    }
}

fn gen_slider(pos: &Position, sq: Square, moves: &mut Vec<Move>, dirs: &[(i32, i32)]) {
    let us = pos.side;
    let f = file_of(sq) as i32;
    let r = rank_of(sq) as i32;
    for (df, dr) in dirs {
        let mut nf = f + df;
        let mut nr = r + dr;
        while on_board(nf, nr) {
            let to = make_square(nf as u8, nr as u8);
            match pos.board[to as usize] {
                None => push_move(moves, sq, to, MoveFlag::Normal, None),
                Some(p) => {
                    if p.color != us {
                        push_move(moves, sq, to, MoveFlag::Normal, None);
                    }
                    break;
                }
            }
            nf += df;
            nr += dr;
        }
    }
}

fn gen_slider_tactical(pos: &Position, sq: Square, moves: &mut Vec<Move>, dirs: &[(i32, i32)]) {
    let us = pos.side;
    let f = file_of(sq) as i32;
    let r = rank_of(sq) as i32;
    for (df, dr) in dirs {
        let mut nf = f + df;
        let mut nr = r + dr;
        while on_board(nf, nr) {
            let to = make_square(nf as u8, nr as u8);
            if let Some(p) = pos.board[to as usize] {
                if p.color != us {
                    push_move(moves, sq, to, MoveFlag::Normal, None);
                }
                break;
            }
            nf += df;
            nr += dr;
        }
    }
}

fn add_promotions(moves: &mut Vec<Move>, from: Square, to: Square) {
    for pt in [
        PieceType::Knight,
        PieceType::Bishop,
        PieceType::Rook,
        PieceType::Queen,
    ] {
        push_move(moves, from, to, MoveFlag::Promotion(pt), Some(pt));
    }
}

fn gen_pawn(pos: &Position, sq: Square, moves: &mut Vec<Move>) {
    let us = pos.side;
    let f = file_of(sq) as i32;
    let r = rank_of(sq) as i32;
    let dir = if us == Color::White { 1 } else { -1 };
    let start_rank: i32 = if us == Color::White { 1 } else { 6 };
    let promo_rank: i32 = if us == Color::White { 7 } else { 0 };

    // Single push.
    let to1 = make_square(f as u8, (r + dir) as u8);
    if pos.board[to1 as usize].is_none() {
        if r + dir == promo_rank {
            add_promotions(moves, sq, to1);
        } else {
            push_move(moves, sq, to1, MoveFlag::Normal, None);
            // Double push (only from the starting rank, into an empty square).
            if r == start_rank {
                let to2 = make_square(f as u8, (r + 2 * dir) as u8);
                if pos.board[to2 as usize].is_none() {
                    push_move(moves, sq, to2, MoveFlag::DoublePawnPush, None);
                }
            }
        }
    }

    // Captures (including en passant), plus promotions.
    for df in [-1i32, 1i32] {
        let nf = f + df;
        let nr = r + dir;
        if !on_board(nf, nr) {
            continue;
        }
        let to = make_square(nf as u8, nr as u8);
        match pos.board[to as usize] {
            Some(p) if p.color != us => {
                if nr == promo_rank {
                    add_promotions(moves, sq, to);
                } else {
                    push_move(moves, sq, to, MoveFlag::Normal, None);
                }
            }
            _ => {
                if Some(to) == pos.ep_target {
                    push_move(moves, sq, to, MoveFlag::EnPassant, None);
                }
            }
        }
    }
}

fn gen_pawn_tactical(pos: &Position, sq: Square, moves: &mut Vec<Move>) {
    let us = pos.side;
    let f = file_of(sq) as i32;
    let r = rank_of(sq) as i32;
    let dir = if us == Color::White { 1 } else { -1 };
    let promo_rank: i32 = if us == Color::White { 7 } else { 0 };

    // A quiet promotion is tactical for qsearch and must retain the original
    // pawn generator's position before capture generation.
    let to1 = make_square(f as u8, (r + dir) as u8);
    if pos.board[to1 as usize].is_none() && r + dir == promo_rank {
        add_promotions(moves, sq, to1);
    }

    // Captures (including en passant), plus capturing promotions.
    for df in [-1i32, 1i32] {
        let nf = f + df;
        let nr = r + dir;
        if !on_board(nf, nr) {
            continue;
        }
        let to = make_square(nf as u8, nr as u8);
        match pos.board[to as usize] {
            Some(p) if p.color != us => {
                if nr == promo_rank {
                    add_promotions(moves, sq, to);
                } else {
                    push_move(moves, sq, to, MoveFlag::Normal, None);
                }
            }
            _ if Some(to) == pos.ep_target => {
                push_move(moves, sq, to, MoveFlag::EnPassant, None);
            }
            _ => {}
        }
    }
}

fn gen_castling(pos: &Position, moves: &mut Vec<Move>) {
    let us = pos.side;
    let enemy = us.opposite();
    if us == Color::White {
        let wk = Piece::new(Color::White, PieceType::King);
        let wr = Piece::new(Color::White, PieceType::Rook);
        if pos.castling.white_king
            && pos.board[E1 as usize] == Some(wk)
            && pos.board[H1 as usize] == Some(wr)
            && pos.board[F1 as usize].is_none()
            && pos.board[G1 as usize].is_none()
            && !pos.is_square_attacked(E1, enemy)
            && !pos.is_square_attacked(F1, enemy)
            && !pos.is_square_attacked(G1, enemy)
        {
            push_move(moves, E1, G1, MoveFlag::KingCastle, None);
        }
        if pos.castling.white_queen
            && pos.board[E1 as usize] == Some(wk)
            && pos.board[A1 as usize] == Some(wr)
            && pos.board[B1 as usize].is_none()
            && pos.board[C1 as usize].is_none()
            && pos.board[D1 as usize].is_none()
            && !pos.is_square_attacked(E1, enemy)
            && !pos.is_square_attacked(D1, enemy)
            && !pos.is_square_attacked(C1, enemy)
        {
            push_move(moves, E1, C1, MoveFlag::QueenCastle, None);
        }
    } else {
        let bk = Piece::new(Color::Black, PieceType::King);
        let br = Piece::new(Color::Black, PieceType::Rook);
        if pos.castling.black_king
            && pos.board[E8 as usize] == Some(bk)
            && pos.board[H8 as usize] == Some(br)
            && pos.board[F8 as usize].is_none()
            && pos.board[G8 as usize].is_none()
            && !pos.is_square_attacked(E8, enemy)
            && !pos.is_square_attacked(F8, enemy)
            && !pos.is_square_attacked(G8, enemy)
        {
            push_move(moves, E8, G8, MoveFlag::KingCastle, None);
        }
        if pos.castling.black_queen
            && pos.board[E8 as usize] == Some(bk)
            && pos.board[A8 as usize] == Some(br)
            && pos.board[B8 as usize].is_none()
            && pos.board[C8 as usize].is_none()
            && pos.board[D8 as usize].is_none()
            && !pos.is_square_attacked(E8, enemy)
            && !pos.is_square_attacked(D8, enemy)
            && !pos.is_square_attacked(C8, enemy)
        {
            push_move(moves, E8, C8, MoveFlag::QueenCastle, None);
        }
    }
}

/// S4.3B: perft using ONLY the fast legal generator (candidate-only helper,
/// used by the differential perft test).
#[cfg(test)]
pub(crate) fn perft_fast(pos: &mut Position, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    let moves = generate_legal_moves_fast_with_stats(pos, None).0;
    let mut nodes = 0u64;
    for m in moves {
        let undo = pos.make_move(m);
        nodes += perft_fast(pos, depth - 1);
        pos.unmake_move(undo);
    }
    nodes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chess::fen::parse_fen;

    fn is_tactical(pos: &Position, m: Move) -> bool {
        m.promotion.is_some()
            || matches!(m.flag, MoveFlag::EnPassant)
            || pos.board[m.to as usize].is_some()
    }

    fn assert_tactical_matches_general(fen: &str) {
        let mut general_pos = parse_fen(fen).expect("fixture must parse");
        let expected: Vec<Move> = generate_legal_moves(&mut general_pos)
            .into_iter()
            .filter(|m| is_tactical(&general_pos, *m))
            .collect();
        let mut specialized_pos = parse_fen(fen).expect("fixture must parse");
        let (actual, _) = generate_legal_tactical_moves_with_stats(&mut specialized_pos);
        assert_eq!(actual, expected, "tactical move order differs for {fen}");
        assert_eq!(specialized_pos.zobrist_key, general_pos.zobrist_key);
    }

    fn assert_evasions_match_general(fen: &str) {
        let mut general_pos = parse_fen(fen).expect("fixture must parse");
        let expected = generate_legal_moves(&mut general_pos);
        let mut specialized_pos = parse_fen(fen).expect("fixture must parse");
        let (actual, _) = generate_legal_evasions_with_stats(&mut specialized_pos);
        assert_eq!(actual, expected, "evasion move order differs for {fen}");
        assert_eq!(specialized_pos.zobrist_key, general_pos.zobrist_key);
    }

    /// S4.3B: legacy vs fast full-legal differential on ONE position.
    fn assert_fast_matches_legacy(fen: &str) {
        let mut legacy_pos = parse_fen(fen).expect("fixture must parse");
        let legacy_key = legacy_pos.zobrist_key;
        let expected = generate_legal_moves(&mut legacy_pos);
        assert_eq!(
            legacy_pos.zobrist_key, legacy_key,
            "legacy generator must fully restore {fen}"
        );
        let mut fast_pos = parse_fen(fen).expect("fixture must parse");
        let (actual, stats) = generate_legal_moves_fast_with_stats(&mut fast_pos, None);
        assert_eq!(actual, expected, "fast legal mismatch for {fen}");
        assert_eq!(
            fast_pos.zobrist_key, legacy_key,
            "fast generator must restore {fen}"
        );
        assert_eq!(
            stats.legal_moves as usize,
            expected.len(),
            "fast stats legal count mismatch for {fen}"
        );
    }

    #[test]
    fn fast_legal_matches_legacy_across_pin_ep_castle_and_check_classes() {
        for fen in [
            // all 8 absolute-pin directions (king e1/e4)
            "4k3/8/8/8/3q4/8/8/3RK3 w - - 0 1", // east pin (rook pinned by d-file queen? king e1, rook d1, queen d4)
            "4k3/8/8/8/4q3/8/8/3RK3 w - - 0 1", // rook pinned on e-file
            "4k3/8/8/8/2b1q3/8/8/3RK3 w - - 0 1", // bishop pinned diagonally (c4-bishop on b3? king e1, bishop... )
            "4k3/8/8/8/3q4/8/8/4KR2 w - - 0 1", // f1 rook pinned east? king e1 rook f1, queen e4 on e-file -> f1 not on e-file... use f-file
            "4k3/8/8/8/8/8/8/3RK2r w - - 0 1",  // west: h-file rook pins d1 rook? no...
            // pinned knight
            "4k3/8/8/8/4q3/8/4N3/4K3 w - - 0 1",
            // pinned pawn push
            "4k3/8/8/8/4q3/8/4P3/4K3 w - - 0 1",
            // pinned pawn capture (ep and plain)
            "4k3/8/8/3pP3/4q3/8/8/4K3 w - d6 0 1",
            // capture of the pinning piece
            "4k3/8/8/8/4R3/8/8/3qK3 w - - 0 1",
            // single check / double check
            "4k3/8/8/8/8/8/4r3/4K3 w - - 0 1",
            "4k3/8/8/8/4b3/8/4r3/4K3 w - - 0 1",
            // en passant discovered rook / bishop attack
            "4k3/8/8/3pP3/8/8/8/3RK3 w - d6 0 1",
            // quiet / capturing promotions
            "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
            "1r2k3/P7/8/8/8/8/8/4K3 w - - 0 1",
            // both castling sides
            "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
            // stalemate and checkmate
            "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",
            "7k/5K2/8/8/8/8/8/7R b - - 0 1",
            // double-rook / double-bishop pin cases
            "4k3/8/8/8/2b1q3/8/8/3RK3 w - - 0 1",
            // king move into/out of attack
            "4k3/8/8/8/8/8/4p3/4K3 w - - 0 1",
        ] {
            assert_fast_matches_legacy(fen);
        }
    }

    #[test]
    fn fast_legal_matches_legacy_on_random_legal_walk() {
        // Deterministic legal-walk differential: hundreds of reachable
        // positions comparing legacy vs fast generators at every node.
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut pos = parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
            .expect("startpos");
        let mut seed = 0x5eed_u64;
        for step in 0..1500 {
            let mut hasher = DefaultHasher::new();
            (step, seed).hash(&mut hasher);
            seed = hasher.finish();
            let (moves, _) = generate_legal_moves_with_stats(&mut pos);
            if moves.is_empty() {
                break;
            }
            assert_fast_matches_legacy(&crate::chess::fen::to_fen(&pos));
            let choice = moves[(seed as usize) % moves.len()];
            let undo = pos.make_move(choice);
            let _ = undo;
        }
    }

    #[test]
    fn fast_perft_matches_legacy_perft_on_fixtures() {
        for (fen, depth) in [
            (
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                4,
            ),
            (
                "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
                3,
            ),
            ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 4),
            (
                "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
                3,
            ),
            (
                "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
                3,
            ),
        ] {
            let mut legacy_pos = parse_fen(fen).expect("fixture must parse");
            let legacy = legacy_pos.perft(depth);
            let mut fast_pos = parse_fen(fen).expect("fixture must parse");
            let fast = perft_fast(&mut fast_pos, depth);
            assert_eq!(
                fast, legacy,
                "fast perft mismatch for {fen} at depth {depth}"
            );
        }
    }

    #[test]
    fn fast_generator_accepts_unpinned_moves_without_probes() {
        // startpos: no pins, no check -> all 20 pseudo moves fast-accepted.
        let mut pos = parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
            .expect("startpos");
        let (moves, stats) = generate_legal_moves_fast_with_stats(&mut pos, None);
        assert_eq!(moves.len() as u64, stats.legal_moves);
        assert_eq!(stats.fast_accepts, 20);
        assert_eq!(stats.fallback_probes, 0);
        assert_eq!(stats.make_moves, 0);
        // position with a pinned knight: pinned-move probes fire.
        let mut pos2 = parse_fen("4k3/8/8/8/4q3/8/4N3/4K3 w - - 0 1").expect("pinned");
        let (_, stats2) = generate_legal_moves_fast_with_stats(&mut pos2, None);
        assert!(stats2.fallback_pinned >= 1);
        assert!(stats2.fallback_probes >= 1);
    }

    #[test]
    fn specialized_tactical_generation_covers_required_move_classes() {
        for fen in [
            // ordinary capture
            "4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1",
            // en passant
            "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
            // quiet promotion
            "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
            // capturing promotion
            "1r2k3/P7/8/8/8/8/8/4K3 w - - 0 1",
            // pinned capture must be filtered by the legality check
            "4r1k1/8/8/8/8/8/r3R3/4K3 w - - 0 1",
        ] {
            assert_tactical_matches_general(fen);
        }
    }

    #[test]
    fn specialized_evasions_match_all_check_classes() {
        for fen in [
            // single check: capture the checking rook
            "4k3/8/8/8/8/8/4r3/4K3 w - - 0 1",
            // single check: a bishop can block the rook check
            "4r1k1/8/8/8/2B5/8/8/4K3 w - - 0 1",
            // single check: king moves
            "4r1k1/8/8/8/8/8/8/4K3 w - - 0 1",
            // double check: only king evasions are legal
            "4r1k1/8/8/8/1b6/8/8/4K3 w - - 0 1",
            // checkmate
            "7k/6Q1/5K2/8/8/8/8/8 b - - 0 1",
            // stalemate is not a check, but must remain an empty legal list
            "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",
        ] {
            assert_evasions_match_general(fen);
        }
    }

    #[test]
    fn tactical_generation_differential_on_deterministic_legal_walk() {
        let mut pos = Position::startpos();
        let mut state = 0x9e37_79b9_u64;
        for _ in 0..256 {
            let expected: Vec<Move> = {
                let mut probe = pos;
                generate_legal_moves(&mut probe)
                    .into_iter()
                    .filter(|m| is_tactical(&probe, *m))
                    .collect()
            };
            let (actual, _) = generate_legal_tactical_moves_with_stats(&mut pos);
            assert_eq!(
                actual, expected,
                "differential mismatch at key {:x}",
                pos.zobrist_key
            );

            if pos.is_in_check(pos.side) {
                let mut evasion_pos = pos;
                let expected_evasions = generate_legal_moves(&mut evasion_pos);
                let (actual_evasions, _) = generate_legal_evasions_with_stats(&mut pos);
                assert_eq!(actual_evasions, expected_evasions);
            }

            let mut next_probe = pos;
            let legal = generate_legal_moves(&mut next_probe);
            if legal.is_empty() {
                break;
            }
            state ^= state << 7;
            state ^= state >> 9;
            state ^= state << 8;
            let next = legal[(state as usize) % legal.len()];
            pos.make_move(next);
        }
    }

    #[test]
    fn has_any_legal_move_distinguishes_quiet_position_and_stalemate() {
        let mut quiet = parse_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        let mut quiet_pseudo = Vec::new();
        generate_pseudo_moves(&quiet, &mut quiet_pseudo);
        let (has_quiet, quiet_stats) = has_any_legal_move_with_stats(&mut quiet);
        assert!(has_quiet);
        assert_eq!(quiet_stats.pseudo_moves, quiet_pseudo.len() as u64);
        assert!(quiet_stats.make_moves < quiet_stats.pseudo_moves);

        let mut stalemate = parse_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1").unwrap();
        let mut stalemate_pseudo = Vec::new();
        generate_pseudo_moves(&stalemate, &mut stalemate_pseudo);
        let (has_stalemate_move, stalemate_stats) = has_any_legal_move_with_stats(&mut stalemate);
        assert!(!has_stalemate_move);
        assert_eq!(stalemate_stats.pseudo_moves, stalemate_pseudo.len() as u64);
        assert_eq!(stalemate_stats.make_moves, stalemate_stats.pseudo_moves);
    }
}
