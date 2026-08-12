//! S6.1A — FeatureSetV1: deterministic sparse chess feature extraction.
//!
//! ONE Rust implementation is the single source of truth for feature
//! semantics. It feeds BOTH the future runtime learned evaluator and the
//! sparse exporter consumed by the Python trainer. Python never re-implements
//! chess feature definitions (S6 contract).
//!
//! Contract:
//! - every feature is `white_count - black_count` (black squares mirrored to
//!   White-relative squares first);
//! - phase: N/B = 1, R = 2, Q = 4, max = 24 (same as the production eval);
//! - tempo: +1 white to move, -1 black to move;
//! - feature IDs are FROZEN once the schema is committed (s6-feature-v1);
//! - symmetry invariant: mirror board vertically + swap colors + swap side to
//!   move => feature vector = -P, phase identical, tempo negated;
//! - extraction is deterministic (same FEN => same byte-for-byte export).

use crate::chess::position::Position;
use crate::chess::types::*;

/// Feature family base offsets (stable, part of the frozen schema).
pub(crate) const F_MATERIAL: u16 = 0; // 5
pub(crate) const F_PSQT: u16 = 5; // 6 * 32 = 192
pub(crate) const F_BISHOP_PAIR: u16 = 197; // 1
pub(crate) const F_PAWN: u16 = 198; // 16
pub(crate) const F_MOBILITY: u16 = 214; // 4
pub(crate) const F_OUTPOST: u16 = 218; // 1
pub(crate) const F_ROOK: u16 = 219; // 3
pub(crate) const F_KING: u16 = 222; // 3
pub(crate) const F_SPACE: u16 = 225; // 1
pub(crate) const F_TEMPO: u16 = 226; // 1
pub(crate) const FEATURE_COUNT: u16 = 227;

/// One extracted feature value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FeatureValue {
    pub(crate) id: u16,
    pub(crate) value: i16,
}

/// Sparse deterministic feature extraction for one position.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct FeatureSetV1 {
    /// Non-pawn material phase 0..=24 (N/B=1, R=2, Q=4, clamped).
    pub(crate) phase: u8,
    /// All `FEATURE_COUNT` values (id == index). Zeros kept dense in the
    /// struct; the exporter drops them for the sparse JSONL.
    pub(crate) values: Vec<i16>,
}

impl FeatureSetV1 {
    pub(crate) fn sparse(&self) -> Vec<FeatureValue> {
        self.values
            .iter()
            .enumerate()
            .filter(|(_, v)| **v != 0)
            .map(|(i, v)| FeatureValue {
                id: i as u16,
                value: *v,
            })
            .collect()
    }
}

/// White-relative square: black ranks mirrored so both colors face "up".
/// Returns (white_relative_rank, file_symmetric_file).
#[inline]
fn white_relative(sq: Square, color: Color) -> (u8, u8) {
    let r = rank_of(sq);
    let f = file_of(sq);
    let rr = if color == Color::White { r } else { 7 - r };
    let ff = if f < 4 { f } else { 7 - f };
    (rr, ff)
}

#[inline]
fn piece_type_index(pt: PieceType) -> u16 {
    match pt {
        PieceType::Pawn => 0,
        PieceType::Knight => 1,
        PieceType::Bishop => 2,
        PieceType::Rook => 3,
        PieceType::Queen => 4,
        PieceType::King => 5,
    }
}

/// Own-pawn attack mask on the file grid (files 0..=7) for one side, plus
/// per-square helpers used by the pawn/structure/king features.
struct PawnView {
    /// files with at least one pawn of the side
    pawn_files: [bool; 8],
    /// count of pawns per file
    pawns_per_file: [u8; 8],
    /// highest BOARD rank of the side's pawns per file (0-based)
    max_rank: [u8; 8],
    /// lowest BOARD rank of the side's pawns per file (0-based)
    min_rank: [u8; 8],
    /// files attacked by pawns of the side
    attack_files: [bool; 8],
    /// per-square: is this square attacked by a pawn of the side
    pawn_attacks: u64,
}

impl PawnView {
    fn highest_rank(&self, file: usize) -> i32 {
        self.max_rank[file] as i32
    }
    fn lowest_rank(&self, file: usize) -> i32 {
        self.min_rank[file] as i32
    }
}

fn pawn_view(pos: &Position, side: Color) -> PawnView {
    let mut v = PawnView {
        pawn_files: [false; 8],
        pawns_per_file: [0; 8],
        max_rank: [0; 8],
        min_rank: [7; 8],
        attack_files: [false; 8],
        pawn_attacks: 0,
    };
    let dir: i32 = if side == Color::White { 1 } else { -1 };
    for sq in 0u8..64u8 {
        if let Some(p) = pos.board[sq as usize] {
            if p.color == side && p.piece_type == PieceType::Pawn {
                let f = file_of(sq) as usize;
                let r = rank_of(sq);
                v.pawn_files[f] = true;
                v.pawns_per_file[f] += 1;
                v.max_rank[f] = v.max_rank[f].max(r);
                v.min_rank[f] = v.min_rank[f].min(r);
                for df in [-1i32, 1i32] {
                    let nf = f as i32 + df;
                    let nr = r as i32 + dir;
                    if (0..8).contains(&nf) && (0..8).contains(&nr) {
                        v.attack_files[nf as usize] = true;
                        v.pawn_attacks |= 1_u64 << make_square(nf as u8, nr as u8);
                    }
                }
            }
        }
    }
    v
}

/// Pawn-structure feature values for one side (already sign-scaled by the
/// caller: white +1, black -1).
fn pawn_features(pos: &Position, side: Color, v: &mut [i16]) {
    let view = pawn_view(pos, side);
    let sign: i16 = if side == Color::White { 1 } else { -1 };
    // isolated / connected / islands / doubled
    let mut isolated = 0i16;
    let mut connected = 0i16;
    let mut islands = 0i16;
    let mut doubled = 0i16;
    for f in 0usize..8 {
        if view.pawn_files[f] {
            let left = f > 0 && view.pawn_files[f - 1];
            let right = f < 7 && view.pawn_files[f + 1];
            if !left && !right {
                isolated += 1;
            }
            if view.pawns_per_file[f] > 1 {
                doubled += view.pawns_per_file[f] as i16 - 1;
            }
        }
    }
    // connected (LOCAL semantics, decided for the frozen schema): a pawn is
    // connected when a friendly pawn on an adjacent file stands at |rank
    // difference| <= 1 (same rank, or one rank ahead/behind). NOT mere
    // adjacent-file presence.
    for sq in 0u8..64u8 {
        if let Some(p) = pos.board[sq as usize] {
            if p.color == side && p.piece_type == PieceType::Pawn {
                let f = file_of(sq) as i32;
                let r = rank_of(sq) as i32;
                let mut conn = false;
                for df in [-1i32, 1i32] {
                    let nf = f + df;
                    if !(0..8).contains(&nf) {
                        continue;
                    }
                    for nr in (r - 1).max(0)..=(r + 1).min(7) {
                        if let Some(q) = pos.board[square_on(nr as u8, nf as u8) as usize] {
                            if q.color == side && q.piece_type == PieceType::Pawn {
                                conn = true;
                            }
                        }
                    }
                }
                if conn {
                    connected += 1;
                }
            }
        }
    }
    let mut run = 0u8;
    for f in 0usize..8 {
        if view.pawn_files[f] {
            run += 1;
        } else if run > 0 {
            islands += 1;
            run = 0;
        }
    }
    if run > 0 {
        islands += 1;
    }
    // passed / protected-passed by advancement rank bucket 0..=5.
    // PASSED uses BOARD-DIRECTION geometry (never cross-color relative
    // advancement): an enemy pawn on the same/adjacent file strictly AHEAD
    // in BOARD rank blocks the passer; enemy pawns behind do not.
    let mut passed = [0i16; 6];
    let mut protected_passed = [0i16; 6];
    let enemy = side.opposite();
    let ev = pawn_view(pos, enemy);
    for sq in 0u8..64u8 {
        if let Some(p) = pos.board[sq as usize] {
            if p.color == side && p.piece_type == PieceType::Pawn {
                let f = file_of(sq) as i32;
                let r = rank_of(sq) as i32;
                let rr = white_relative(sq, side).0 as i32;
                let adv = (rr - 1).max(0) as usize;
                let bucket = adv.min(5);
                let mut is_passed = true;
                for df in -1i32..=1 {
                    let nf = f + df;
                    if !(0..8).contains(&nf) {
                        continue;
                    }
                    let ahead = if side == Color::White {
                        ev.highest_rank(nf as usize) > r
                    } else {
                        ev.lowest_rank(nf as usize) < r
                    };
                    if ahead {
                        is_passed = false;
                    }
                }
                if is_passed {
                    passed[bucket] += 1;
                    if view.pawn_attacks & (1_u64 << sq) != 0 {
                        protected_passed[bucket] += 1;
                    }
                }
            }
        }
    }
    v[F_PAWN as usize] += sign * isolated;
    v[F_PAWN as usize + 1] += sign * doubled;
    v[F_PAWN as usize + 2] += sign * connected;
    for i in 0..6 {
        v[F_PAWN as usize + 3 + i] += sign * passed[i];
        v[F_PAWN as usize + 9 + i] += sign * protected_passed[i];
    }
    v[F_PAWN as usize + 15] += sign * islands;
}

/// Material, bishop pair, PSQT, mobility, minor/rook/king/space features
/// computed side by side (each feature accumulates white - black).
pub(crate) fn extract_features_v1(pos: &Position) -> FeatureSetV1 {
    let mut values = vec![0i16; FEATURE_COUNT as usize];

    // tempo
    values[F_TEMPO as usize] = if pos.side == Color::White { 1 } else { -1 };

    let mut material = [0i16; 5];
    let mut bishop_pairs = [0i16; 2];
    let mut psqt = vec![0i16; 192];
    let mut outposts = [0i16; 2];
    let mut rook_open = [0i16; 2];
    let mut rook_semi = [0i16; 2];
    let mut rook_seventh = [0i16; 2];
    let mut shelter = [0i16; 2];
    let mut adj_open = [0i16; 2];
    let mut storm = [0i16; 2];
    let mut space = [0i16; 2];

    let white_view = pawn_view(pos, Color::White);
    let black_view = pawn_view(pos, Color::Black);

    for sq in 0u8..64u8 {
        if let Some(p) = pos.board[sq as usize] {
            let pi = piece_type_index(p.piece_type) as usize;
            if p.piece_type != PieceType::King {
                material[pi.min(4)] += if p.color == Color::White { 1 } else { -1 };
            }
            let (rr, ff) = white_relative(sq, p.color);
            psqt[pi * 32 + rr as usize * 4 + ff as usize] +=
                if p.color == Color::White { 1 } else { -1 };
            if p.piece_type == PieceType::Bishop {
                let bi = if p.color == Color::White { 0 } else { 1 };
                bishop_pairs[bi] += 1;
            }
            match p.piece_type {
                PieceType::Knight => {
                    // outpost: rel rank 4..=5, not attacked by enemy pawns,
                    // protected by own pawn
                    if (4..=5).contains(&rr) {
                        let own = if p.color == Color::White {
                            &white_view
                        } else {
                            &black_view
                        };
                        let enemy_v = if p.color == Color::White {
                            &black_view
                        } else {
                            &white_view
                        };
                        let idx = if p.color == Color::White { 0 } else { 1 };
                        if enemy_v.pawn_attacks & (1_u64 << sq) == 0
                            && own.pawn_attacks & (1_u64 << sq) != 0
                        {
                            outposts[idx] += 1;
                        }
                    }
                }
                PieceType::Rook => {
                    let idx = if p.color == Color::White { 0 } else { 1 };
                    let f = file_of(sq) as usize;
                    let own = if p.color == Color::White {
                        &white_view
                    } else {
                        &black_view
                    };
                    let any = white_view.pawn_files[f] || black_view.pawn_files[f];
                    if !any {
                        rook_open[idx] += 1;
                    } else if !own.pawn_files[f] {
                        rook_semi[idx] += 1;
                    }
                    if rr == 6 {
                        rook_seventh[idx] += 1;
                    }
                }
                PieceType::King => {
                    let idx = if p.color == Color::White { 0 } else { 1 };
                    let f = file_of(sq) as i32;
                    let own = if p.color == Color::White {
                        &white_view
                    } else {
                        &black_view
                    };
                    // shelter: own pawns on king file +-1, rel ranks 1..=3
                    let mut sh = 0i16;
                    for df in -1i32..=1 {
                        let nf = f + df;
                        if !(0..8).contains(&nf) {
                            continue;
                        }
                        for rr2 in 1..=3u8 {
                            let sq2 = square_on(
                                if p.color == Color::White {
                                    rr2
                                } else {
                                    7 - rr2
                                },
                                nf as u8,
                            );
                            if let Some(q) = pos.board[sq2 as usize] {
                                if q.color == p.color && q.piece_type == PieceType::Pawn {
                                    sh += 1;
                                }
                            }
                        }
                        // open/semi-open adjacent file (no own pawns)
                        if !own.pawn_files[nf as usize] {
                            adj_open[idx] += 1;
                        }
                    }
                    shelter[idx] += sh;
                    // storm: enemy pawns on king file +-1, rel ranks 5..=7
                    let mut st = 0i16;
                    for df in -1i32..=1 {
                        let nf = f + df;
                        if !(0..8).contains(&nf) {
                            continue;
                        }
                        for rr2 in 4..=6u8 {
                            let sq2 = square_on(
                                if p.color == Color::White {
                                    rr2
                                } else {
                                    7 - rr2
                                },
                                nf as u8,
                            );
                            if let Some(q) = pos.board[sq2 as usize] {
                                if q.color != p.color && q.piece_type == PieceType::Pawn {
                                    st += 1;
                                }
                            }
                        }
                    }
                    storm[idx] += st;
                }
                _ => {}
            }
            // space: own non-pawn pieces on relative ranks 3..=5
            if p.piece_type != PieceType::Pawn
                && p.piece_type != PieceType::King
                && (3..=5).contains(&rr)
            {
                let idx = if p.color == Color::White { 0 } else { 1 };
                space[idx] += 1;
            }
        }
    }
    // bishop pair = 2+ bishops per side
    for b in &mut bishop_pairs {
        *b = (*b >= 2) as i16;
    }

    // mobility: pseudo moves of BOTH sides. The sign follows COLOR (white
    // +1, black -1), never the side to move: under mirror+swap the feature
    // must negate, which requires color-fixed signs.
    let mut mobility = [0i16; 4];
    for side in [Color::White, Color::Black] {
        let sign: i16 = if side == Color::White { 1 } else { -1 };
        let mut side_pos = *pos;
        side_pos.side = side;
        let mut side_pseudo: Vec<Move> = Vec::new();
        #[allow(clippy::unnecessary_mut_passed, clippy::ptr_arg)]
        crate::chess::movegen::generate_pseudo_moves(&mut side_pos, &mut side_pseudo);
        for m in &side_pseudo {
            if let Some(p) = pos.board[m.from as usize] {
                let mi = match p.piece_type {
                    PieceType::Knight => 0,
                    PieceType::Bishop => 1,
                    PieceType::Rook => 2,
                    PieceType::Queen => 3,
                    _ => continue,
                };
                mobility[mi] += sign;
            }
        }
    }

    // assemble dense vector
    for i in 0..5 {
        values[F_MATERIAL as usize + i] = material[i];
    }
    for i in 0..192 {
        values[F_PSQT as usize + i] = psqt[i];
    }
    values[F_BISHOP_PAIR as usize] = bishop_pairs[0] - bishop_pairs[1];
    pawn_features(pos, Color::White, &mut values);
    pawn_features(pos, Color::Black, &mut values);
    for i in 0..4 {
        values[F_MOBILITY as usize + i] = mobility[i];
    }
    values[F_OUTPOST as usize] = outposts[0] - outposts[1];
    values[F_ROOK as usize] = rook_open[0] - rook_open[1];
    values[F_ROOK as usize + 1] = rook_semi[0] - rook_semi[1];
    values[F_ROOK as usize + 2] = rook_seventh[0] - rook_seventh[1];
    values[F_KING as usize] = shelter[0] - shelter[1];
    values[F_KING as usize + 1] = adj_open[0] - adj_open[1];
    values[F_KING as usize + 2] = storm[0] - storm[1];
    values[F_SPACE as usize] = space[0] - space[1];

    FeatureSetV1 {
        phase: crate::engine::eval::game_phase(pos) as u8,
        values,
    }
}

#[inline]
fn square_on(rank: u8, file: u8) -> u8 {
    rank * 8 + file
}

/// S6.1A feature name table (frozen with the committed schema).
pub(crate) fn feature_name(id: u16) -> String {
    let pt_names = ["pawn", "knight", "bishop", "rook", "queen", "king"];
    if (F_MATERIAL..=F_MATERIAL + 4).contains(&id) {
        return format!("material.{}", pt_names[(id - F_MATERIAL) as usize]);
    }
    if (F_PSQT..=F_PSQT + 191).contains(&id) {
        let off = id - F_PSQT;
        let pt = pt_names[(off / 32) as usize];
        let sq = off % 32;
        let rank = sq / 4 + 1;
        let file = sq % 4; // canonical representative file: a,b,c,d
        return format!("psqt.{}.{}{}", pt, char::from(b'a' + file as u8), rank);
    }
    if id == F_BISHOP_PAIR {
        return "bishop.pair".to_string();
    }
    if (F_PAWN..=F_PAWN + 15).contains(&id) {
        let off = id - F_PAWN;
        return match off {
            0 => "pawn.isolated".to_string(),
            1 => "pawn.doubled".to_string(),
            2 => "pawn.connected".to_string(),
            3..=8 => format!("pawn.passed.r{}", off as u8 - 1),
            9..=14 => format!("pawn.protected_passed.r{}", off as u8 - 7),
            _ => "pawn.islands".to_string(),
        };
    }
    if (F_MOBILITY..=F_MOBILITY + 3).contains(&id) {
        return format!(
            "mobility.{}",
            ["knight", "bishop", "rook", "queen"][(id - F_MOBILITY) as usize]
        );
    }
    if id == F_OUTPOST {
        return "minor.outpost".to_string();
    }
    if (F_ROOK..=F_ROOK + 2).contains(&id) {
        return ["rook.open_file", "rook.semi_open_file", "rook.seventh_rank"]
            [(id - F_ROOK) as usize]
            .to_string();
    }
    if (F_KING..=F_KING + 2).contains(&id) {
        return [
            "king.pawn_shelter",
            "king.open_semi_adjacent",
            "king.enemy_pawn_storm",
        ][(id - F_KING) as usize]
            .to_string();
    }
    if id == F_SPACE {
        return "space".to_string();
    }
    if id == F_TEMPO {
        return "tempo".to_string();
    }
    format!("unknown.{id}")
}

/// SHA-256 of the canonical feature schema (frozen by
/// tools/s6/freeze_schema.py; the engine never hashes at runtime - model
/// artifacts declare their schema SHA and the runtime compares it against
/// this compiled constant). Real 64-hex, not a fingerprint.
#[allow(dead_code)] // consumed by S6.2 model-artifact loading
pub(crate) const FEATURE_SCHEMA_SHA256: &str =
    "8c5c51ac1e9c33e2796d52b01b3002e9c09d714c8517f4d4f0da44d0f4e70d7e";

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chess::fen::parse_fen;

    /// Vertical mirror + color swap + side-to-move swap on a FEN.
    fn mirror_swap(fen: &str) -> String {
        let parts: Vec<&str> = fen.split(' ').collect();
        let mut rows: Vec<&str> = parts[0].split('/').collect();
        rows.reverse();
        let mut board = String::new();
        for (i, row) in rows.iter().enumerate() {
            let mut out = String::new();
            for c in row.chars() {
                match c {
                    'P' => out.push('p'),
                    'N' => out.push('n'),
                    'B' => out.push('b'),
                    'R' => out.push('r'),
                    'Q' => out.push('q'),
                    'K' => out.push('k'),
                    'p' => out.push('P'),
                    'n' => out.push('N'),
                    'b' => out.push('B'),
                    'r' => out.push('R'),
                    'q' => out.push('Q'),
                    'k' => out.push('K'),
                    d => out.push(d),
                }
            }
            board.push_str(&out);
            if i < 7 {
                board.push('/');
            }
        }
        let side = if parts[1] == "w" { "b" } else { "w" };
        format!(
            "{} {} {} {} {} {}",
            board, side, parts[2], parts[3], parts[4], parts[5]
        )
    }

    #[test]
    fn feature_count_and_names_are_stable() {
        assert_eq!(FEATURE_COUNT, 227);
        assert_eq!(FEATURE_SCHEMA_SHA256.len(), 64);
        assert!(
            FEATURE_SCHEMA_SHA256
                .chars()
                .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
            "FEATURE_SCHEMA_SHA256 must be 64 lowercase hex"
        );
        assert_eq!(feature_name(0), "material.pawn");
        assert_eq!(feature_name(4), "material.queen");
        // PSQT file-symmetric buckets use canonical names a,b,c,d
        assert_eq!(feature_name(5), "psqt.pawn.a1");
        assert_eq!(feature_name(6), "psqt.pawn.b1");
        assert_eq!(feature_name(7), "psqt.pawn.c1");
        assert_eq!(feature_name(8), "psqt.pawn.d1");
        assert_eq!(feature_name(193), "psqt.king.a8"); // rr7, file 0
        assert_eq!(feature_name(196), "psqt.king.d8"); // rr7, file 3
        assert_eq!(feature_name(197), "bishop.pair");
        assert_eq!(feature_name(198), "pawn.isolated");
        assert_eq!(feature_name(226), "tempo");
        // every id has a name
        for id in 0..FEATURE_COUNT {
            assert!(!feature_name(id).is_empty());
        }
    }

    #[test]
    fn symmetry_mirror_swap_negates_vector() {
        let fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r1bq1rk1/pp2ppbp/2np1np1/8/2PNP3/2N1B3/PP3PPP/R1BQK2R w KQ - 3 9",
            "4k3/8/8/8/4q3/8/4N3/4K3 w - - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        ];
        for fen in fens {
            let pos = parse_fen(fen).expect("fen");
            let mirrored = parse_fen(&mirror_swap(fen)).expect("mirror");
            let a = extract_features_v1(&pos);
            let b = extract_features_v1(&mirrored);
            assert_eq!(a.phase, b.phase, "phase identical for {fen}");
            assert_eq!(a.values.len(), b.values.len());
            for (i, (x, y)) in a.values.iter().zip(b.values.iter()).enumerate() {
                assert_eq!(*x, -(*y), "feature {i} must negate for {fen}");
            }
            assert_eq!(a.values[F_TEMPO as usize], -b.values[F_TEMPO as usize]);
        }
    }

    #[test]
    fn determinism_same_fen_same_vector() {
        let fen = "r1bq1rk1/pp2ppbp/2np1np1/8/2PNP3/2N1B3/PP3PPP/R1BQK2R w KQ - 3 9";
        let pos = parse_fen(fen).expect("fen");
        assert_eq!(extract_features_v1(&pos), extract_features_v1(&pos));
    }

    #[test]
    fn passed_pawn_board_direction_geometry() {
        // S6.1A Repair 1: passed must use board-direction geometry, NOT
        // cross-color relative advancement. White pawn e5:
        //  - black e6 (same file ahead)   -> NOT passed
        //  - black d6 / f6 (adjacent ahead) -> NOT passed
        //  - black d4 (behind)            -> passed
        let e5_id = F_PAWN + 3 + 3; // adv 3 -> bucket 3
        let cases: [(&str, bool); 4] = [
            ("k7/8/4p3/4P3/8/8/8/7K w - - 0 1", false), // e6 ahead
            ("k7/8/3p4/4P3/8/8/8/7K w - - 0 1", false), // d6 ahead
            ("k7/8/5p2/4P3/8/8/8/7K w - - 0 1", false), // f6 ahead
            ("k7/8/8/4P3/8/3p4/8/7K w - - 0 1", true),  // d4 behind
        ];
        for (fen, expect_passed) in cases {
            let pos = parse_fen(fen).unwrap_or_else(|e| panic!("{fen}: {e}"));
            let f = extract_features_v1(&pos);
            let v = f.values[e5_id as usize];
            assert_eq!(
                v == 1,
                expect_passed,
                "e5 passed={} for {fen} (got {v})",
                expect_passed
            );
            // black mirror must negate AND the mirrored black pawn must have
            // the SAME passed status
            let mir = parse_fen(&mirror_swap(fen)).expect("mirror");
            let g = extract_features_v1(&mir);
            assert_eq!(
                g.values[e5_id as usize], -v,
                "mirror of {fen} must negate e5 passer"
            );
        }
        // black e5 (0-based rank 4 -> rr 3 -> adv 2 -> bucket 2): white e4
        // ahead for black -> NOT passed; white e6 behind for black -> passed.
        let e5b_id = F_PAWN + 3 + 2;
        let black_cases: [(&str, bool); 4] = [
            ("k7/8/8/4p3/4P3/8/8/7K b - - 0 1", false), // white e4 ahead
            ("k7/8/8/4p3/3P4/8/8/7K b - - 0 1", false), // white d4 ahead
            ("k7/8/8/4p3/5P2/8/8/7K b - - 0 1", false), // white f4 ahead
            ("k7/8/4P3/4p3/8/8/8/7K b - - 0 1", true),  // white e6 behind
                                                        // e6 = 1-based rank 6 = index 2: already "4P3" at index 2 above ✓
        ];
        for (fen, expect_passed) in black_cases {
            let pos = parse_fen(fen).unwrap_or_else(|e| panic!("{fen}: {e}"));
            let f = extract_features_v1(&pos);
            // black pawn passes are negative
            let v = f.values[e5b_id as usize];
            assert_eq!(
                v == -1,
                expect_passed,
                "black e5 passed={} for {fen} (got {v})",
                expect_passed
            );
        }
    }

    #[test]
    fn connected_pawn_local_semantics() {
        // connected = friendly pawn on adjacent file at same rank or one
        // rank behind. White c4 + d3 (d3 protects c4) -> both connected.
        let pos = parse_fen("k7/8/8/8/2P5/3P4/8/7K w - - 0 1").expect("fen");
        let f = extract_features_v1(&pos);
        assert_eq!(f.values[F_PAWN as usize + 2], 2, "both pawns connected");
        // White c4 + d6 (far apart) -> NOT connected.
        let pos2 = parse_fen("k7/8/3P4/8/2P5/8/8/7K w - - 0 1").expect("fen");
        let f2 = extract_features_v1(&pos2);
        assert_eq!(
            f2.values[F_PAWN as usize + 2],
            0,
            "far apart: not connected"
        );
    }

    #[test]
    fn snapshot_fens_cover_all_families() {
        // Each family must be nonzero on its dedicated snapshot.
        let cases: Vec<(&str, &[u16])> = vec![
            // isolated pawns (white a2 isolated, black d5/d6 chain)
            ("7k/2p5/3p4/8/8/8/P7/7K w - - 0 1", &[198]),
            // doubled pawns
            ("7k/8/8/8/8/2p5/2p5/7K w - - 0 1", &[199]),
            // connected pawns
            ("7k/8/8/8/8/4p3/3p4/7K w - - 0 1", &[200]),
            // passed pawn on relative rank 7
            ("8/P6k/8/8/8/8/8/7K w - - 0 1", &[201 + 5]),
            // protected passed pawn (white pawns a7+b6+a5; a5 protects b6)
            ("8/P7/1P6/P7/8/8/8/k6K w - - 0 1", &[201 + 4, 207 + 4]),
            // bishop pair (white)
            ("1b1k4/8/8/8/8/8/8/B1B1K3 w - - 0 1", &[197]),
            // knight outpost (white knight on c5, protected by b4 pawn, no
            // black pawns attacking c5)
            ("8/8/8/2N5/1P6/8/8/k1K5 w - - 0 1", &[218]),
            // rook open file (white rook on a-file, no pawns)
            ("k7/8/8/8/8/8/8/R1K5 w - - 0 1", &[219]),
            // rook semi-open (white rook a1, own pawn b2, black pawn a7)
            ("k7/p7/8/8/8/8/1P6/R1K5 w - - 0 1", &[220]),
            // rook seventh (white rook a7)
            ("k7/R7/8/8/8/8/8/2K5 w - - 0 1", &[221]),
            // king shelter (white king g1 with f2/g2/h2 pawns)
            ("6k1/8/8/8/8/8/5PPP/6K1 w - - 0 1", &[222]),
            // enemy pawn storm (black pawns f5,g5,h5 vs white king g1)
            ("6k1/8/8/5ppp/8/8/8/6K1 w - - 0 1", &[224]),
            // space (white pieces on ranks 3-5)
            ("k7/8/8/4N3/8/8/8/K7 w - - 0 1", &[225]),
            // mobility: queens on open board
            ("k7/8/8/8/8/8/8/KQ6 w - - 0 1", &[217]),
            // tempo: white to move
            ("4k3/8/8/8/8/8/8/4K3 w - - 0 1", &[226]),
            // rook ending (white rook a1 open file; black rook e3 semi-open)
            ("4k3/8/8/8/8/4r3/4P3/R3K3 w - - 0 1", &[219, 220]),
            // pawn ending (white pawn e5 isolated + passed)
            ("4k3/8/8/4P3/8/8/8/4K3 w - - 0 1", &[198, 201 + 3]),
            // middlegame (queen mobility, king shelter; c4/e4 are NOT passed:
            // black b7/d6/f6 are ahead on adjacent files)
            (
                "r1bq1rk1/pp2ppbp/2np1np1/8/2PNP3/2N1B3/PP3PPP/R1BQK2R w KQ - 3 9",
                &[217, 222],
            ),
        ];
        for (fen, ids) in cases {
            let pos = parse_fen(fen).unwrap_or_else(|e| panic!("{fen}: {e}"));
            let f = extract_features_v1(&pos);
            for id in ids {
                assert_ne!(
                    f.values[*id as usize],
                    0,
                    "feature {id} ({}) must be nonzero for {fen}",
                    feature_name(*id)
                );
            }
        }
    }

    #[test]
    fn startpos_features_sanity() {
        let pos = parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
            .expect("startpos");
        let f = extract_features_v1(&pos);
        assert_eq!(f.phase, 24);
        assert_eq!(f.values[F_TEMPO as usize], 1);
        assert_eq!(f.values[F_MATERIAL as usize], 0); // 8 vs 8 pawns
        assert_eq!(f.values[F_BISHOP_PAIR as usize], 0); // 1 vs 1 pair
        assert_eq!(f.values[F_MOBILITY as usize], 0); // 4 vs 4 knights
        assert_eq!(f.values[F_ROOK as usize], 0);
        assert_eq!(f.values[F_MATERIAL as usize + 1], 0); // 4 vs 4 knights
                                                          // mirrored white/black knights share a psqt id -> net 0 at startpos
        let knight_b1 = F_PSQT + 32 + 1; // white b1 -> rr0, file 1
        assert_eq!(f.values[knight_b1 as usize], 0);
        // asymmetric knight: only white knight b1
        let pos2 = parse_fen("k7/8/8/8/8/8/8/1N5K w - - 0 1").expect("knight");
        let f2 = extract_features_v1(&pos2);
        assert_eq!(f2.values[knight_b1 as usize], 1);
    }
}
