"""S10-I1-A: sibling-ranking auxiliary corpus + loss (trainer module).

Loaded by train_nnue.py only when --rank-corpus is given. Without it the
trainer is byte-identical to the legacy path.

Semantics (frozen):
  model score for a parent's move = -(material(child) + residual(child))
    (the child's side-to-move is the opponent)
  valid pair: |teacher_i - teacher_j| >= 20cp (parent POV, cp only)
  pair weight w = min(|gap|, 400) / 400
  pair loss = w * softplus(-sign(gap) * (s_i - s_j) / 100cp)
  parent loss = mean over its valid pairs
  corpus loss = mean over parents in the batch (move-count invariance)
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

RANK_MIN_GAP_CP = 20.0
RANK_TEMPERATURE_CP = 100.0
RANK_MAX_WEIGHT_GAP_CP = 400.0


class RankCorpus:
    """Precomputes child features and parent pair tensors."""

    def __init__(self, path: Path, engine_bin: Path, device,
                 exported_cache: dict | None = None):
        import torch

        parents = [json.loads(l) for l in
                   Path(path).read_text(encoding="utf-8").splitlines()
                   if l.strip()]
        if not parents:
            raise SystemExit("FAIL CLOSED: empty rank corpus")

        # child features via the Rust exporter (representation truth)
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location(
            "_tn", str(Path(__file__).parent / "train_nnue.py"))
        _tn = _ilu.module_from_spec(_spec)
        import sys as _sys
        _sys.modules["_tn"] = _tn
        _spec.loader.exec_module(_tn)
        EncodedSplit = _tn.EncodedSplit
        export_features_from_engine = _tn.export_features_from_engine
        material_cp_stm_python = _tn.material_cp_stm_python
        child_records = []
        meta = []  # (parent_idx, child_idx_within_parent, teacher_cp)
        for pi, p in enumerate(parents):
            for si, sib in enumerate(p["siblings"]):
                child_records.append({
                    "position_id": f"r{pi}_{si}",
                    "fen": sib["child_fen"],
                })
                meta.append((pi, si, sib.get("cp")))
        exported = export_features_from_engine(
            Path(engine_bin), child_records, "v2")

        items = []
        for (pi, si, tcp), rec in zip(meta, child_records):
            exp = exported[rec["position_id"]]
            stm_white = rec["fen"].split()[1] == "w"
            items.append({
                "pi": pi, "si": si, "teacher_cp": tcp,
                "stm": exp["white"] if stm_white else exp["black"],
                "nstm": exp["black"] if stm_white else exp["white"],
                "material": float(material_cp_stm_python(rec["fen"])),
            })
        enc = EncodedSplit([
            {**it, "target_scaled": 0.0, "target_cp": 0.0} for it in items])

        self.device = device
        self.stm_ind = enc.stm_indices.to(device)
        self.stm_off = enc.stm_offsets.to(device)
        self.nstm_ind = enc.nstm_indices.to(device)
        self.nstm_off = enc.nstm_offsets.to(device)
        self.material = torch.tensor(
            [it["material"] for it in items], dtype=torch.float32,
            device=device)

        # flat child index -> (pi, si); si is the child's index within parent
        self.pi_of = [it["pi"] for it in items]
        self.si_of = [it["si"] for it in items]
        self.teacher_cp = [it["teacher_cp"] for it in items]

        # per-parent child lists (flat indices), cp-comparable only
        by_parent = defaultdict(list)
        for i, it in enumerate(items):
            if it["teacher_cp"] is not None:
                by_parent[it["pi"]].append(i)
        self.parents_with_pairs = []
        self.pair_a = []   # flat child idx
        self.pair_b = []
        self.pair_sign = []
        self.pair_w = []
        self.pair_parent = []
        for pi, children in sorted(by_parent.items()):
            pairs = []
            for x in range(len(children)):
                for y in range(x + 1, len(children)):
                    ia, ib = children[x], children[y]
                    gap = self.teacher_cp[ia] - self.teacher_cp[ib]
                    if abs(gap) < RANK_MIN_GAP_CP:
                        continue
                    pairs.append((ia, ib))
            if not pairs:
                continue
            self.parents_with_pairs.append(pi)
            for ia, ib in pairs:
                gap = self.teacher_cp[ia] - self.teacher_cp[ib]
                self.pair_a.append(ia)
                self.pair_b.append(ib)
                self.pair_sign.append(1.0 if gap > 0 else -1.0)
                self.pair_w.append(
                    min(abs(gap), RANK_MAX_WEIGHT_GAP_CP)
                    / RANK_MAX_WEIGHT_GAP_CP)
                self.pair_parent.append(pi)

        self.pair_a = torch.tensor(self.pair_a, device=device)
        self.pair_b = torch.tensor(self.pair_b, device=device)
        self.pair_sign = torch.tensor(self.pair_sign, dtype=torch.float32,
                                      device=device)
        self.pair_w = torch.tensor(self.pair_w, dtype=torch.float32,
                                   device=device)
        self.pair_parent = torch.tensor(self.pair_parent, device=device)
        # unique parent ids tensor for segment means
        self.parent_ids = torch.tensor(self.parents_with_pairs,
                                       device=device)

        # S10-I1-A R1 perf: CSR parent->pair index (pairs are appended in
        # parent order during construction, so a cumulative count gives the
        # segment boundaries directly). A 32-parent batch then touches only
        # its own ~580 pairs instead of scanning all 177k.
        _pair_parent_list = (self.pair_parent.tolist()
                              if hasattr(self.pair_parent, "tolist")
                              else self.pair_parent)
        pair_counts = [0] * (max(_pair_parent_list) + 1
                             if _pair_parent_list else 1)
        for pp in _pair_parent_list:
            pair_counts[pp] += 1
        self.pair_csr_offsets = [0]
        for c in pair_counts:
            self.pair_csr_offsets.append(self.pair_csr_offsets[-1] + c)
        # CSR parent->children index (children were also appended in parent
        # order)
        child_counts = [0] * (max(self.pi_of) + 1 if self.pi_of else 1)
        for pi in self.pi_of:
            child_counts[pi] += 1
        self.child_csr_offsets = [0]
        for c in child_counts:
            self.child_csr_offsets.append(self.child_csr_offsets[-1] + c)
        self.parent_children = [
            list(range(self.child_csr_offsets[pi],
                       self.child_csr_offsets[pi + 1]))
            for pi in range(len(child_counts))]

        # S10-I1-A R1 performance: precompute per-child (start, len) segment
        # bounds for BOTH perspectives so the rank-batch repack is pure
        # tensor indexing (no Python per-child slicing on the hot path).
        self.stm_starts = self.stm_off.clone()
        self.nstm_starts = self.nstm_off.clone()
        stm_total = self.stm_ind.numel()
        nstm_total = self.nstm_ind.numel()
        n = self.stm_off.numel()
        stm_ends = torch.cat([self.stm_off[1:],
                              torch.tensor([stm_total], device=device)])
        nstm_ends = torch.cat([self.nstm_off[1:],
                               torch.tensor([nstm_total], device=device)])
        self.stm_lens = stm_ends - self.stm_starts
        self.nstm_lens = nstm_ends - self.nstm_starts

        self.n_children = len(items)
        self.n_pairs = len(self.pair_a)
        self.n_parents = len(self.parents_with_pairs)

    def epoch_parent_batch(self, epoch, batch_idx):
        """Deterministic per-epoch parent shuffle; batch `batch_idx` of
        size 32. One full pass per epoch = ceil(n_parents/32) batches."""
        import torch
        import random as _random
        rng = _random.Random(0x11A + epoch)
        order = list(range(self.n_parents))
        rng.shuffle(order)
        per = 32
        start = batch_idx * per
        ids = order[start:start + per]
        if not ids:
            return torch.tensor([], dtype=torch.long, device=self.device)
        return torch.tensor(ids, device=self.device)

    def _children_of(self, parent_ids):
        """Flat child indices whose parent is in parent_ids (CSR lookup)."""
        out = []
        for pid in parent_ids:
            out.extend(self.parent_children[pid])
        return out

    def loss_for_parents(self, model, parent_ids):
        """Ranking loss for the given parent ids (mean over parents).

        REPACKS the children of these parents into a compact batch with
        recomputed embedding offsets (subsetting the flattened index
        tensors without repacking desyncs embedding_bag ranges — the
        `end >= begin` trap). ~32 parents x ~8 children = ~256 children,
        the same magnitude as the scalar batch.
        """
        import torch
        if self.n_pairs == 0:
            return None
        ids = parent_ids.tolist() if hasattr(parent_ids, "tolist") \
            else list(parent_ids)
        if not ids:
            return None
        children = self._children_of(ids)
        if not children:
            return None
        # repack features: gather segments and rebuild offsets
        # Vectorized repack (identical semantics to the loop version):
        # gather child segments via repeat_interleave/arange and rebuild
        # the per-item START offsets for embedding_bag mode='sum'.
        ch = torch.tensor(children, device=self.device)
        stm_len = self.stm_lens[ch]
        nstm_len = self.nstm_lens[ch]
        stm_off = torch.zeros_like(stm_len)
        nstm_off = torch.zeros_like(nstm_len)
        torch.cumsum(stm_len[:-1], dim=0, out=stm_off[1:])
        torch.cumsum(nstm_len[:-1], dim=0, out=nstm_off[1:])
        # index gather: for child k, indices stm_starts[k] .. +len[k]
        total_stm = int(stm_len.sum().item())
        total_nstm = int(nstm_len.sum().item())
        stm_repeat = torch.repeat_interleave(stm_len)
        nstm_repeat = torch.repeat_interleave(nstm_len)
        stm_ranks = torch.arange(total_stm, device=self.device) - \
            torch.repeat_interleave(stm_off, stm_len)
        nstm_ranks = torch.arange(total_nstm, device=self.device) - \
            torch.repeat_interleave(nstm_off, nstm_len)
        stm_ind = self.stm_ind[torch.repeat_interleave(
            self.stm_starts[ch], stm_len) + stm_ranks]
        nstm_ind = self.nstm_ind[torch.repeat_interleave(
            self.nstm_starts[ch], nstm_len) + nstm_ranks]

        preds = model(stm_ind, stm_off, nstm_ind, nstm_off)
        composed = self.material[children] + preds * 1000.0
        score = -composed  # parent POV

        # CSR pair lookup: only the batch parents' own pairs (construction
        # appended pairs in parent order, so slices are contiguous).
        import torch as _t
        pos_lookup = {i: k for k, i in enumerate(children)}
        a_list, b_list, s_list, w_list, p_list = [], [], [], [], []
        for pid in ids:
            lo = self.pair_csr_offsets[pid]
            hi = self.pair_csr_offsets[pid + 1]
            if lo == hi:
                continue
            pa_seg = self.pair_a[lo:hi].tolist()
            pb_seg = self.pair_b[lo:hi].tolist()
            for j in range(hi - lo):
                ia, ib = pa_seg[j], pb_seg[j]
                if ia in pos_lookup and ib in pos_lookup:
                    a_list.append(pos_lookup[ia])
                    b_list.append(pos_lookup[ib])
                    s_list.append(self.pair_sign[lo + j].item())
                    w_list.append(self.pair_w[lo + j].item())
                    p_list.append(self.pair_parent[lo + j].item())
        if not a_list:
            return None
        pa = _t.tensor(a_list, device=self.device)
        pb = _t.tensor(b_list, device=self.device)
        sign_v = _t.tensor(s_list, device=self.device)
        w_v = _t.tensor(w_list, device=self.device)
        parent_v = _t.tensor(p_list, device=self.device)
        pair_loss = w_v * _t.nn.functional.softplus(
            -sign_v * (score[pa] - score[pb]) / RANK_TEMPERATURE_CP)
        uniq = _t.unique(parent_v)
        total = _t.zeros(len(uniq), device=self.device)
        counts = _t.zeros(len(uniq), device=self.device)
        pp = _t.searchsorted(uniq, parent_v)
        total.index_add_(0, pp, pair_loss)
        counts.index_add_(0, pp, _t.ones_like(pair_loss))
        have = counts > 0
        return (total[have] / counts[have]).mean()

    def raw_loss_magnitude(self, model):
        """Mean raw (unweighted-by-rank_scale) pair loss over ALL pairs —
        used once at initialization for the rank_scale calibration."""
        import torch
        with torch.no_grad():
            preds = model(self.stm_ind, self.stm_off,
                          self.nstm_ind, self.nstm_off)
            composed = self.material + preds * 1000.0
            score = -composed
            s_a = score[self.pair_a]
            s_b = score[self.pair_b]
            pair_loss = self.pair_w * torch.nn.functional.softplus(
                -self.pair_sign * (s_a - s_b) / RANK_TEMPERATURE_CP)
            return pair_loss.mean().item()
