"""S10-J2-0: FROZEN phase routing contract.

phase_score(pos) = sum over BOTH sides of piece weights
    N = 1, B = 1, R = 2, Q = 4 (P = K = 0)
    — IDENTICAL to the production HCE `phase_weight` in
      src/engine/eval.rs (EvalContext.phase) and to the classifier
      used by every H0 artifact (h0_a1/h0_c_shift/h0_e).

buckets (the H0 thresholds, unchanged):
    high : phase >= 18
    mid  : 8 <= phase < 18
    low  : 1 <= phase < 8
    zero : phase == 0

Counts (verified before training, frozen):
    train       high 200,000 / mid 360,000 / low 160,000 / zero 80,000
    validation  high  25,000 / mid  45,000 / low  20,000 / zero 10,000
    H0-E lockbox 64 / 64 / 64 / 64 (by construction)

Runtime: computable in the SAME board scan as material_cp_stm.
"""

PHASE_WEIGHTS = {"n": 1, "b": 1, "r": 2, "q": 4}

BUCKET_THRESHOLDS = {"high": 18, "mid": 8, "low": 1}
BUCKETS = ("high", "mid", "low", "zero")


def phase_score(piece_map) -> int:
    total = 0
    for pc in piece_map.values():
        total += PHASE_WEIGHTS.get(pc.symbol().lower(), 0)
    return total


def bucket_of_phase(phase: int) -> str:
    if phase >= 18:
        return "high"
    if phase >= 8:
        return "mid"
    if phase >= 1:
        return "low"
    return "zero"
