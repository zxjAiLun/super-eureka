//! Transposition table — core types, allocation, probe, store, and mate codec.
//!
//! This is the first M3.2 commit. The table is NOT yet wired into the search;
//! it is exercised only by unit tests. See the spec for the exact contract.

use std::fmt;
use std::mem::size_of;

use crate::chess::types::Move;
use crate::chess::zobrist::ZobristKey;
use crate::engine::search::MATE;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

pub const MATE_THRESHOLD: i32 = MATE - 1000; // 999_000
pub const MAX_MATE_PLY: u32 = 999;

// ---------------------------------------------------------------------------
// Mate score codec
// ---------------------------------------------------------------------------

/// Encode a raw search score for TT storage by adding `ply` to mate scores.
/// Returns `None` if `ply` exceeds `MAX_MATE_PLY` or if the arithmetic would
/// overflow `i32`.
pub fn score_to_tt(score: i32, ply: u32) -> Option<i32> {
    if ply > MAX_MATE_PLY {
        return None;
    }

    if score > MATE_THRESHOLD {
        score.checked_add_unsigned(ply)
    } else if score < -MATE_THRESHOLD {
        score.checked_sub_unsigned(ply)
    } else {
        Some(score)
    }
}

/// Decode a stored TT score back to a raw search score by subtracting `ply`
/// from mate scores. Returns `None` if `ply` exceeds `MAX_MATE_PLY` or if
/// the arithmetic would overflow `i32`.
pub fn score_from_tt(stored: i32, ply: u32) -> Option<i32> {
    if ply > MAX_MATE_PLY {
        return None;
    }

    if stored > MATE_THRESHOLD {
        stored.checked_sub_unsigned(ply)
    } else if stored < -MATE_THRESHOLD {
        stored.checked_add_unsigned(ply)
    } else {
        Some(stored)
    }
}

// ---------------------------------------------------------------------------
// TtKey
// ---------------------------------------------------------------------------

/// Key used to index and verify a TT entry. Must capture all context that
/// distinguishes otherwise-identical board positions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TtKey {
    pub position: ZobristKey,
    pub halfmove_clock: u32,
    pub repetition_signature: u64,
}

impl TtKey {
    pub const fn new(position: ZobristKey, halfmove_clock: u32, repetition_signature: u64) -> Self {
        TtKey {
            position,
            halfmove_clock,
            repetition_signature,
        }
    }
}

// ---------------------------------------------------------------------------
// Bound
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bound {
    Exact,
    Lower,
    Upper,
}

// ---------------------------------------------------------------------------
// TTEntry
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TTEntry {
    pub key: TtKey,
    pub depth: u32,
    pub score: i32,
    pub bound: Bound,
    pub best_move: Option<Move>,
}

// ---------------------------------------------------------------------------
// TtAllocError
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TtAllocError {
    InvalidSize,
    ArithmeticOverflow,
    AllocationFailed,
}

impl fmt::Display for TtAllocError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TtAllocError::InvalidSize => write!(f, "invalid TT size (must be >= 1 MB)"),
            TtAllocError::ArithmeticOverflow => write!(f, "TT size arithmetic overflow"),
            TtAllocError::AllocationFailed => write!(f, "TT memory allocation failed"),
        }
    }
}

impl std::error::Error for TtAllocError {}

// ---------------------------------------------------------------------------
// Index mixing
// ---------------------------------------------------------------------------

/// Fixed deterministic 64-bit mixer that reads all three fields of `TtKey`.
/// Based on SplitMix64 over a single u64 assembled from the three fields,
/// then folded to the table capacity.
fn tt_index(key: TtKey, capacity: usize) -> usize {
    if capacity == 0 {
        return 0;
    }

    // Mix position, halfmove_clock, and repetition_signature into one u64.
    let h = key.position;
    let c = key.halfmove_clock as u64;
    let r = key.repetition_signature;

    // w = h ^ c ^ r  (XOR combine, then a SplitMix64 finalisation pass).
    let mut z = h ^ c ^ r;
    z ^= z >> 30;
    z = z.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z ^= z >> 27;
    z = z.wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^= z >> 31;

    z as usize % capacity
}

// ---------------------------------------------------------------------------
// TranspositionTable
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct TranspositionTable {
    slots: Vec<Option<TTEntry>>,
    size_mb: usize,
}

impl Default for TranspositionTable {
    fn default() -> Self {
        TranspositionTable::disabled()
    }
}

impl TranspositionTable {
    /// Create a disabled (zero-capacity) table. Probes always miss; stores
    /// are no-ops.
    pub fn disabled() -> Self {
        TranspositionTable {
            slots: Vec::new(),
            size_mb: 0,
        }
    }

    /// Allocate a new table with capacity derived from `mb` megabytes.
    pub fn new_mb(mb: usize) -> Result<Self, TtAllocError> {
        if mb == 0 {
            return Err(TtAllocError::InvalidSize);
        }

        let bytes = mb
            .checked_mul(1024)
            .and_then(|n| n.checked_mul(1024))
            .ok_or(TtAllocError::ArithmeticOverflow)?;

        let entries = bytes / size_of::<Option<TTEntry>>();
        if entries == 0 {
            return Err(TtAllocError::InvalidSize);
        }

        let mut slots = Vec::new();
        slots
            .try_reserve_exact(entries)
            .map_err(|_| TtAllocError::AllocationFailed)?;
        slots.resize(entries, None);

        Ok(TranspositionTable { slots, size_mb: mb })
    }

    /// Resize the table to a new capacity derived from `mb` megabytes.
    /// On failure the old table is fully preserved.
    pub fn resize_mb(&mut self, mb: usize) -> Result<(), TtAllocError> {
        if mb == 0 {
            return Err(TtAllocError::InvalidSize);
        }

        let bytes = mb
            .checked_mul(1024)
            .and_then(|n| n.checked_mul(1024))
            .ok_or(TtAllocError::ArithmeticOverflow)?;

        let entries = bytes / size_of::<Option<TTEntry>>();
        if entries == 0 {
            return Err(TtAllocError::InvalidSize);
        }

        let mut new_slots = Vec::new();
        new_slots
            .try_reserve_exact(entries)
            .map_err(|_| TtAllocError::AllocationFailed)?;
        new_slots.resize(entries, None);

        // Success: atomically publish the new table.
        self.slots = new_slots;
        self.size_mb = mb;

        Ok(())
    }

    /// Clear all entries. Capacity is preserved.
    pub fn clear(&mut self) {
        for slot in &mut self.slots {
            *slot = None;
        }
    }

    /// Number of entries the table can hold.
    pub fn capacity_entries(&self) -> usize {
        self.slots.len()
    }

    /// Configured size in megabytes (not a byte-rounded inference).
    pub fn size_mb(&self) -> usize {
        self.size_mb
    }

    /// Probe the table. Returns a copy of the matching entry, or `None` on
    /// miss (disabled table, empty slot, or key mismatch).
    pub fn probe(&self, key: TtKey) -> Option<TTEntry> {
        if self.slots.is_empty() {
            return None;
        }

        let idx = tt_index(key, self.slots.len());
        match &self.slots[idx] {
            Some(entry) if entry.key == key => Some(*entry),
            _ => None,
        }
    }

    /// Store an entry according to the replacement rules.
    ///
    /// Empty slot: write directly.
    /// Same full key: replace if deeper, or same-depth Exact over non-Exact,
    ///   or same-depth same-quality Some(move) over None.
    /// Different key, same index (collision): replace only if `new.depth >= old.depth`.
    pub fn store(&mut self, entry: TTEntry) {
        if self.slots.is_empty() {
            return; // disabled: no-op
        }

        let idx = tt_index(entry.key, self.slots.len());

        match &self.slots[idx] {
            None => {
                // Empty slot: write directly.
                self.slots[idx] = Some(entry);
            }
            Some(old) if old.key == entry.key => {
                // Same full key: apply quality rules.
                if should_replace_same_key(old, &entry) {
                    self.slots[idx] = Some(entry);
                }
            }
            Some(old) => {
                // Collision (different key, same index).
                if (entry.depth as i32) >= (old.depth as i32) {
                    self.slots[idx] = Some(entry);
                }
            }
        }
    }
}

/// Decide whether `new` should replace `old` when they share the same `TtKey`.
fn should_replace_same_key(old: &TTEntry, new: &TTEntry) -> bool {
    debug_assert_eq!(old.key, new.key);

    // Deeper entry always replaces shallower.
    if new.depth > old.depth {
        return true;
    }

    if new.depth == old.depth {
        // Same-depth Exact replaces non-Exact.
        if new.bound == Bound::Exact && old.bound != Bound::Exact {
            return true;
        }

        // Same-depth same-quality: prefer entry with a best_move.
        let same_quality = bound_quality(new.bound) == bound_quality(old.bound);
        if same_quality && new.best_move.is_some() && old.best_move.is_none() {
            return true;
        }
    }

    false
}

/// Classify bound quality: Exact is higher quality than Lower/Upper.
fn bound_quality(b: Bound) -> u8 {
    match b {
        Bound::Exact => 2,
        Bound::Lower | Bound::Upper => 1,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chess::fen::parse_fen;
    use crate::chess::movegen::generate_legal_moves;
    use crate::chess::types::Move;

    // ---- Mate codec ----

    #[test]
    fn ordinary_cp_unchanged() {
        assert_eq!(score_to_tt(0, 0), Some(0));
        assert_eq!(score_to_tt(999_000, 0), Some(999_000));
        assert_eq!(score_to_tt(-999_000, 0), Some(-999_000));
        assert_eq!(score_from_tt(0, 0), Some(0));
        assert_eq!(score_from_tt(999_000, 0), Some(999_000));
        assert_eq!(score_from_tt(-999_000, 0), Some(-999_000));
    }

    #[test]
    fn mate_boundary_encode() {
        // +999001 is a mate score, +999000 is ordinary cp.
        assert_eq!(score_to_tt(MATE_THRESHOLD + 1, MAX_MATE_PLY), Some(MATE));
        assert_eq!(
            score_to_tt(-(MATE_THRESHOLD + 1), MAX_MATE_PLY),
            Some(-MATE)
        );
        // Threshold itself is ordinary cp.
        assert_eq!(score_to_tt(MATE_THRESHOLD, 0), Some(MATE_THRESHOLD));
        assert_eq!(score_to_tt(-MATE_THRESHOLD, 0), Some(-MATE_THRESHOLD));
    }

    #[test]
    fn mate_ply_exceeds_max() {
        assert_eq!(score_to_tt(MATE_THRESHOLD + 1, MAX_MATE_PLY + 1), None);
        assert_eq!(score_from_tt(MATE, MAX_MATE_PLY + 1), None);
    }

    #[test]
    fn mate_overflow_stress() {
        // i32::MAX + anything overflows
        assert_eq!(score_to_tt(i32::MAX, 1), None);
        assert_eq!(score_to_tt(i32::MIN, 1), None);

        // Exact boundary at MAX_MATE_PLY
        assert_eq!(score_to_tt(i32::MAX - 999, MAX_MATE_PLY), Some(i32::MAX));
        assert_eq!(score_to_tt(i32::MAX - 998, MAX_MATE_PLY), None);

        assert_eq!(score_to_tt(i32::MIN + 999, MAX_MATE_PLY), Some(i32::MIN));
        assert_eq!(score_to_tt(i32::MIN + 998, MAX_MATE_PLY), None);
    }

    #[test]
    fn positive_mate_same_ply_roundtrip() {
        // MATE - 11 = mate at distance 11 plies (root-relative).
        let raw = MATE - 11;
        let stored = score_to_tt(raw, 7).unwrap();
        let decoded = score_from_tt(stored, 7).unwrap();
        assert_eq!(decoded, raw);
    }

    #[test]
    fn negative_mate_same_ply_roundtrip() {
        let raw = -(MATE - 11);
        let stored = score_to_tt(raw, 7).unwrap();
        let decoded = score_from_tt(stored, 7).unwrap();
        assert_eq!(decoded, raw);
    }

    #[test]
    fn different_ply_decode_positive() {
        // Node at ply 7, mate in 4 more plies: root-relative MATE - 11.
        let score_at_ply_7 = MATE - 11;
        let stored = score_to_tt(score_at_ply_7, 7).unwrap();
        // stored == MATE - 4 node-relative.

        // Decode at ply 9: root is now 2 plies deeper.
        assert_eq!(score_from_tt(stored, 9), Some(MATE - 13));
    }

    #[test]
    fn different_ply_decode_negative() {
        let score_at_ply_7 = -(MATE - 11);
        let stored = score_to_tt(score_at_ply_7, 7).unwrap();

        assert_eq!(score_from_tt(stored, 9), Some(-(MATE - 13)));
    }

    #[test]
    fn cp_unchanged_through_roundtrip() {
        for cp in &[-5000, -100, 0, 100, 5000, 999_000, -999_000] {
            assert_eq!(score_to_tt(*cp, 0), Some(*cp));
            assert_eq!(score_from_tt(*cp, 0), Some(*cp));
        }
    }

    // ---- TtKey ----

    #[test]
    fn tt_key_new() {
        let k = TtKey::new(42, 0, 0xdead);
        assert_eq!(k.position, 42);
        assert_eq!(k.halfmove_clock, 0);
        assert_eq!(k.repetition_signature, 0xdead);
    }

    // ---- Index mixing smoke ----

    #[test]
    fn index_mixes_all_fields() {
        let cap = 1_000;
        let k1 = TtKey::new(100, 0, 0);
        let k2 = TtKey::new(100, 1, 0);
        let k3 = TtKey::new(100, 0, 1);
        // At least one differs from k1 (modulo collision is acceptable, but
        // changing fields must have a chance to change the result).
        let i1 = tt_index(k1, cap);
        let any_diff = (tt_index(k2, cap) != i1) || (tt_index(k3, cap) != i1);
        assert!(
            any_diff,
            "index must depend on halfmove_clock and repetition_signature"
        );
    }

    // ---- Disabled / Default ----

    #[test]
    fn default_is_disabled() {
        let tt: TranspositionTable = Default::default();
        assert_eq!(tt.capacity_entries(), 0);
        assert_eq!(tt.size_mb(), 0);
        assert!(tt.probe(TtKey::new(0, 0, 0)).is_none());
    }

    #[test]
    fn disabled_probe_always_miss() {
        let tt = TranspositionTable::disabled();
        assert!(tt.probe(TtKey::new(42, 0, 0)).is_none());
    }

    #[test]
    fn disabled_store_no_op() {
        let mut tt = TranspositionTable::disabled();
        let entry = TTEntry {
            key: TtKey::new(42, 0, 0),
            depth: 5,
            score: 100,
            bound: Bound::Exact,
            best_move: None,
        };
        tt.store(entry);
        assert!(tt.probe(TtKey::new(42, 0, 0)).is_none());
    }

    #[test]
    fn disabled_clear_no_op() {
        let mut tt = TranspositionTable::disabled();
        tt.clear(); // must not panic
        assert_eq!(tt.capacity_entries(), 0);
    }

    // ---- Allocation ----

    #[test]
    fn new_mb_zero_is_invalid() {
        assert_eq!(
            TranspositionTable::new_mb(0).unwrap_err(),
            TtAllocError::InvalidSize
        );
    }

    #[test]
    fn new_mb_one_succeeds() {
        let tt = TranspositionTable::new_mb(1).expect("1 MB");
        assert!(tt.capacity_entries() >= 1);
        assert_eq!(tt.size_mb(), 1);
    }

    #[test]
    fn new_mb_overflow() {
        assert_eq!(
            TranspositionTable::new_mb(usize::MAX).unwrap_err(),
            TtAllocError::ArithmeticOverflow
        );
    }

    // ---- Resize ----

    #[test]
    fn resize_zero_errors_and_preserves_old() {
        let mut tt = TranspositionTable::new_mb(1).unwrap();
        let old_cap = tt.capacity_entries();
        let old_sz = tt.size_mb();

        assert_eq!(tt.resize_mb(0).unwrap_err(), TtAllocError::InvalidSize);
        assert_eq!(tt.capacity_entries(), old_cap);
        assert_eq!(tt.size_mb(), old_sz);
    }

    #[test]
    fn resize_success_updates_and_clears() {
        let mut tt = TranspositionTable::new_mb(1).unwrap();
        // Store something
        let entry = TTEntry {
            key: TtKey::new(1, 0, 0),
            depth: 1,
            score: 10,
            bound: Bound::Exact,
            best_move: None,
        };
        tt.store(entry);
        assert_eq!(tt.probe(TtKey::new(1, 0, 0)), Some(entry));

        // Resize
        tt.resize_mb(2).unwrap();
        assert_eq!(tt.size_mb(), 2);
        assert!(tt.capacity_entries() > 0);
        // Entries are cleared
        assert!(tt.probe(TtKey::new(1, 0, 0)).is_none());
    }

    #[test]
    fn resize_overflow_preserves_old() {
        let mut tt = TranspositionTable::new_mb(1).unwrap();
        let old_cap = tt.capacity_entries();
        let old_sz = tt.size_mb();

        assert_eq!(
            tt.resize_mb(usize::MAX).unwrap_err(),
            TtAllocError::ArithmeticOverflow
        );
        assert_eq!(tt.capacity_entries(), old_cap);
        assert_eq!(tt.size_mb(), old_sz);
    }

    // ---- Full-key hit / miss ----

    #[test]
    fn exact_key_hit() {
        let mut tt = TranspositionTable::new_mb(1).unwrap();
        let k = TtKey::new(42, 3, 0xabc);
        let entry = TTEntry {
            key: k,
            depth: 5,
            score: 100,
            bound: Bound::Exact,
            best_move: None,
        };
        tt.store(entry);
        assert_eq!(tt.probe(k), Some(entry));
    }

    #[test]
    fn miss_on_position_diff() {
        let mut tt = TranspositionTable::new_mb(1).unwrap();
        let k1 = TtKey::new(42, 3, 0xabc);
        let k2 = TtKey::new(99, 3, 0xabc);
        tt.store(TTEntry {
            key: k1,
            depth: 5,
            score: 100,
            bound: Bound::Exact,
            best_move: None,
        });
        assert!(tt.probe(k2).is_none());
    }

    #[test]
    fn miss_on_halfmove_clock_diff() {
        let mut tt = TranspositionTable::new_mb(1).unwrap();
        let k1 = TtKey::new(42, 3, 0xabc);
        let k2 = TtKey::new(42, 7, 0xabc);
        tt.store(TTEntry {
            key: k1,
            depth: 5,
            score: 100,
            bound: Bound::Exact,
            best_move: None,
        });
        assert!(tt.probe(k2).is_none());
    }

    #[test]
    fn miss_on_repetition_signature_diff() {
        let mut tt = TranspositionTable::new_mb(1).unwrap();
        let k1 = TtKey::new(42, 3, 0xabc);
        let k2 = TtKey::new(42, 3, 0xdef);
        tt.store(TTEntry {
            key: k1,
            depth: 5,
            score: 100,
            bound: Bound::Exact,
            best_move: None,
        });
        assert!(tt.probe(k2).is_none());
    }

    // ---- Same-key replacement ----

    #[test]
    fn deeper_replaces_shallower() {
        let mut tt = TranspositionTable::new_mb(1).unwrap();
        let k = TtKey::new(1, 0, 0);
        let shallow = TTEntry {
            key: k,
            depth: 3,
            score: 50,
            bound: Bound::Exact,
            best_move: None,
        };
        let deep = TTEntry {
            key: k,
            depth: 6,
            score: 80,
            bound: Bound::Exact,
            best_move: None,
        };
        tt.store(shallow);
        tt.store(deep);
        assert_eq!(tt.probe(k).unwrap().depth, 6);
    }

    #[test]
    fn shallower_does_not_replace_deeper() {
        let mut tt = TranspositionTable::new_mb(1).unwrap();
        let k = TtKey::new(1, 0, 0);
        let deep = TTEntry {
            key: k,
            depth: 6,
            score: 80,
            bound: Bound::Exact,
            best_move: None,
        };
        let shallow = TTEntry {
            key: k,
            depth: 3,
            score: 50,
            bound: Bound::Exact,
            best_move: None,
        };
        tt.store(deep);
        tt.store(shallow);
        assert_eq!(tt.probe(k).unwrap().depth, 6);
    }

    #[test]
    fn same_depth_exact_replaces_lower() {
        let mut tt = TranspositionTable::new_mb(1).unwrap();
        let k = TtKey::new(1, 0, 0);
        let lower = TTEntry {
            key: k,
            depth: 5,
            score: 50,
            bound: Bound::Lower,
            best_move: None,
        };
        let exact = TTEntry {
            key: k,
            depth: 5,
            score: 80,
            bound: Bound::Exact,
            best_move: None,
        };
        tt.store(lower);
        tt.store(exact);
        assert_eq!(tt.probe(k).unwrap().bound, Bound::Exact);
    }

    #[test]
    fn same_depth_exact_replaces_upper() {
        let mut tt = TranspositionTable::new_mb(1).unwrap();
        let k = TtKey::new(1, 0, 0);
        let upper = TTEntry {
            key: k,
            depth: 5,
            score: 50,
            bound: Bound::Upper,
            best_move: None,
        };
        let exact = TTEntry {
            key: k,
            depth: 5,
            score: 80,
            bound: Bound::Exact,
            best_move: None,
        };
        tt.store(upper);
        tt.store(exact);
        assert_eq!(tt.probe(k).unwrap().bound, Bound::Exact);
    }

    #[test]
    fn same_depth_non_exact_does_not_replace_exact() {
        let mut tt = TranspositionTable::new_mb(1).unwrap();
        let k = TtKey::new(1, 0, 0);
        let exact = TTEntry {
            key: k,
            depth: 5,
            score: 80,
            bound: Bound::Exact,
            best_move: None,
        };
        let lower = TTEntry {
            key: k,
            depth: 5,
            score: 50,
            bound: Bound::Lower,
            best_move: None,
        };
        tt.store(exact);
        tt.store(lower);
        assert_eq!(tt.probe(k).unwrap().bound, Bound::Exact);
    }

    #[test]
    fn same_quality_some_move_replaces_none() {
        let mut tt = TranspositionTable::new_mb(1).unwrap();
        let k = TtKey::new(1, 0, 0);
        let mv = find_first_move();
        let no_move = TTEntry {
            key: k,
            depth: 5,
            score: 80,
            bound: Bound::Exact,
            best_move: None,
        };
        let with_move = TTEntry {
            key: k,
            depth: 5,
            score: 80,
            bound: Bound::Exact,
            best_move: Some(mv),
        };
        tt.store(no_move);
        tt.store(with_move);
        assert_eq!(tt.probe(k).unwrap().best_move, Some(mv));
    }

    #[test]
    fn same_quality_none_does_not_replace_some() {
        let mut tt = TranspositionTable::new_mb(1).unwrap();
        let k = TtKey::new(1, 0, 0);
        let mv = find_first_move();
        let with_move = TTEntry {
            key: k,
            depth: 5,
            score: 80,
            bound: Bound::Exact,
            best_move: Some(mv),
        };
        let no_move = TTEntry {
            key: k,
            depth: 5,
            score: 80,
            bound: Bound::Exact,
            best_move: None,
        };
        tt.store(with_move);
        tt.store(no_move);
        assert_eq!(tt.probe(k).unwrap().best_move, Some(mv));
    }

    fn find_first_move() -> Move {
        let pos = parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        generate_legal_moves(&mut pos.clone())[0]
    }

    // ---- Collision ----

    /// Collision: different keys at same index. Deeper replaces shallower.
    #[test]
    fn collision_deeper_replaces_shallower() {
        let mut tt = TranspositionTable::disabled();
        tt.slots = vec![None];
        tt.size_mb = 0;

        let k1 = TtKey::new(1, 0, 0);
        let k2 = TtKey::new(2, 0, 0);

        let shallow = TTEntry {
            key: k1,
            depth: 3,
            score: 50,
            bound: Bound::Exact,
            best_move: None,
        };
        let deep = TTEntry {
            key: k2,
            depth: 6,
            score: 80,
            bound: Bound::Exact,
            best_move: None,
        };
        tt.store(shallow); // k1 stored at slot 0
        tt.store(deep); // k2 depth (6) >= k1 depth (3) -> replaces
                        // k2 is now at slot 0
        assert_eq!(tt.probe(k2), Some(deep));
        // k1 is gone
        assert!(tt.probe(k1).is_none());
    }

    /// Collision: shallower does not replace deeper.
    #[test]
    fn collision_shallower_does_not_replace_deeper() {
        let mut tt = TranspositionTable::disabled();
        tt.slots = vec![None];
        tt.size_mb = 0;

        let k1 = TtKey::new(1, 0, 0);
        let k2 = TtKey::new(2, 0, 0);

        let deep = TTEntry {
            key: k1,
            depth: 6,
            score: 80,
            bound: Bound::Exact,
            best_move: None,
        };
        let shallow = TTEntry {
            key: k2,
            depth: 3,
            score: 50,
            bound: Bound::Exact,
            best_move: None,
        };
        tt.store(deep); // k1 (depth 6) stored at slot 0
        tt.store(shallow); // k2 depth (3) < k1 depth (6) -> NOT replaced
                           // k1 still at slot 0
        assert_eq!(tt.probe(k1), Some(deep));
        // k2 was never stored
        assert!(tt.probe(k2).is_none());
    }

    // ---- Clear ----

    #[test]
    fn clear_empties_all_slots() {
        let mut tt = TranspositionTable::new_mb(1).unwrap();
        let k = TtKey::new(1, 0, 0);
        tt.store(TTEntry {
            key: k,
            depth: 5,
            score: 100,
            bound: Bound::Exact,
            best_move: None,
        });
        assert!(tt.probe(k).is_some());
        tt.clear();
        assert!(tt.probe(k).is_none());
        // capacity preserved
        assert!(tt.capacity_entries() > 0);
    }

    // ---- best_move None preservation ----

    #[test]
    fn best_move_none_preserved() {
        let mut tt = TranspositionTable::new_mb(1).unwrap();
        let k = TtKey::new(1, 0, 0);
        let entry = TTEntry {
            key: k,
            depth: 5,
            score: 100,
            bound: Bound::Exact,
            best_move: None,
        };
        tt.store(entry);
        let retrieved = tt.probe(k).unwrap();
        assert!(retrieved.best_move.is_none());
    }
}
