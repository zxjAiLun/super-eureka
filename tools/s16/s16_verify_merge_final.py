"""S16 item 1 (final): factorised vs merged float eval on the TRAINED checkpoint.

The in-training check runs before the loop, when the shared table is all-zero,
so it cannot certify the merge for a non-zero shared table. This loads the
saved best checkpoint and redoes the comparison on real pool rows.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, r"tools\s14")
sys.path.insert(0, r"tools\s15")
sys.path.insert(0, r"tools\s16")
import s14_train  # noqa: E402
import nnue_cache  # noqa: E402
import s16_train as S  # noqa: E402

CKPT = Path(sys.argv[1])
CACHE = sys.argv[2] if len(sys.argv) > 2 else r"data\s15\cache\cache-000.ekc"
N = int(sys.argv[3]) if len(sys.argv) > 3 else 4096

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
sd = ck["model_state_dict"]
model = S.S16Model(num_inputs=S.NNUE_INPUTS_V2R12, ft_width=256)
model.load_state_dict(sd)
model.eval()

shared = sd["ft_shared"]
print(f"checkpoint : {CKPT}")
print(f"shared rows: {tuple(shared.shape)}")
print(f"sentinel 704 absmax = {float(shared[S.SENTINEL].abs().max()):.3e}")
print(f"real rows 0..703 absmax = {float(shared[:704].abs().max()):.6f}")
print(f"shared is non-zero on real rows: "
      f"{bool((shared[:704] != 0).any())}")
print()

pool = nnue_cache.load_pool([CACHE], max_index=S.NNUE_INPUTS_V2R12,
                            device="cpu")
n = min(N, pool.n)
idx = torch.arange(n)
s_i, s_o = s14_train._gather_bag(pool.stm_flat, pool.stm_off, idx)
n_i, n_o = s14_train._gather_bag(pool.nstm_flat, pool.nstm_off, idx)
b = pool.bucket[idx]

with torch.no_grad():
    a = model(s_i, s_o, n_i, n_o, b)
    c = model.merged_forward(s_i, s_o, n_i, n_o, b)
    d = (a - c).abs()
    mx = float(d.max())
    mw = (a - c).abs() / torch.clamp(a.abs(), min=1e-6)

r12_rows = int((s_i >= S.V2_BASE).sum() + (n_i >= S.V2_BASE).sum())
print(f"rows compared        : {n}")
print(f"R12 feature rows seen: {r12_rows}")
print(f"max |factorised-merged|     = {mx:.6e}")
print(f"max relative diff (>1e-6)   = {float(mw.max()):.6e}")
print()
ok = mx < 1e-5
print(f"RESULT: {'PASS' if ok else 'FAIL'} (tolerance 1e-5)")
sys.exit(0 if ok else 1)
