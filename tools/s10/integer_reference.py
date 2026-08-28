#!/usr/bin/env python3
"""S10-B5-C: Python INTEGER reference for the EUNN2Q01 scheme (v2).

True integer arithmetic mirroring the future Rust runtime exactly:
integer FT accumulate, integer clipping, integer dense input requantization
((a+16)>>5), integer i16 dense MAC, arithmetic shift with round-half-away,
final fixed-point -> cp conversion (single float multiply at the end).

This module is the semantic single source of truth for quantized inference;
the Rust runtime must match it BIT-EXACTLY on the raw integer output
(Gate 1).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.s10.export_quantized import (
    FT_SHIFT, DENSE_W_SHIFT, DENSE_Z_SHIFT, QA,
    build_quantized_arrays, load_frozen_sd,
)


class IntegerNnueV2:
    """Integer reference model; inference is pure integer arithmetic.

    Scheme v3: dense MACs consume accumulator-precision activations
    directly (no input requantization); weights i16 in 2^12 scale; bias in
    2^24 scale; z >> 12 back to A units.
    """

    def __init__(self, q: dict):
        self.ft_w = q["ft_w"]          # int16 [22528, 128]
        self.ft_b = q["ft_b"]          # int32 [128]
        self.l1_w = q["l1_w"]          # int16 [32, 256]
        self.l1_b = q["l1_b"]          # int32 [32]
        self.l2_w = q["l2_w"]          # int16 [32, 32]
        self.l2_b = q["l2_b"]          # int32 [32]
        self.out_w = q["out_w"]        # int16 [1, 32]
        self.out_b = q["out_b"]        # int32 [1]

    @classmethod
    def from_frozen_checkpoint(cls) -> "IntegerNnueV2":
        sd, _ = load_frozen_sd()
        return cls(build_quantized_arrays(sd))

    def accumulate(self, indices) -> np.ndarray:
        """acc = q_bias + sum(q_w rows); returns int64 [128] (A units)."""
        acc = self.ft_b.astype(np.int64)
        for idx in indices:
            acc += self.ft_w[idx].astype(np.int64)
        return acc

    @staticmethod
    def _shift_round(x: int, shift: int) -> int:
        """Arithmetic right shift, round half away from zero."""
        if x >= 0:
            return (x + (1 << (shift - 1))) >> shift
        return -((-x + (1 << (shift - 1))) >> shift)

    def evaluate_raw(self, stm_indices, nstm_indices) -> int:
        """Full integer forward pass; returns raw output in A units."""
        stm_acc = self.accumulate(stm_indices)
        nstm_acc = self.accumulate(nstm_indices)

        # ClippedReLU(0, QA) over [stm, nstm] -> 256 activations (A units).
        acts0 = [int(v) for v in np.concatenate([
            np.clip(stm_acc, 0, QA), np.clip(nstm_acc, 0, QA)])]

        def dense(w: np.ndarray, b: np.ndarray, acts: list[int]) -> list[int]:
            # MAC at accumulator precision (no input requantization).
            outs = []
            for o in range(w.shape[0]):
                z = int(b[o])
                row = w[o]
                for i, av in enumerate(acts):
                    z += int(row[i]) * av
                outs.append(self._shift_round(z, DENSE_Z_SHIFT))
            return outs

        a1 = [min(max(v, 0), QA) for v in dense(self.l1_w, self.l1_b, acts0)]
        a2 = [min(max(v, 0), QA) for v in dense(self.l2_w, self.l2_b, a1)]
        z_out = int(self.out_b[0]) + sum(
            int(self.out_w[0][i]) * a2[i] for i in range(32))
        return self._shift_round(z_out, DENSE_Z_SHIFT)

    def evaluate_cp(self, stm_indices, nstm_indices) -> float:
        """cp = raw / 2^FT_SHIFT * 1000 (single float conversion)."""
        raw = self.evaluate_raw(stm_indices, nstm_indices)
        return (raw / (1 << FT_SHIFT)) * 1000.0
