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

        self.n_children = len(items)
        self.n_pairs = len(self.pair_a)
        self.n_parents = len(self.parents_with_pairs)

    def batch_parent_ids(self, step, steps_total):
        """Deterministic round-robin of parents across rank steps."""
        import torch
        per = max(1, self.n_parents // max(1, steps_total) + 1)
        start = (step * per) % self.n_parents
        ids = [(start + k) % self.n_parents
               for k in range(min(per, self.n_parents))]
        return torch.tensor(ids, device=self.device)

    def loss_for_parents(self, model, parent_ids):
        """Ranking loss for the given parent ids (mean over parents).

        Evaluates ALL corpus children in ONE embedding_bag forward
        (offsets must stay aligned with the full flattened index list —
        subsetting children would desync the offset ranges, which is
        exactly the EmbeddingBag `end >= begin` trap), then selects the
        pairs whose parent is in parent_ids.
        """
        import torch
        if self.n_pairs == 0:
            return None
        preds = model(self.stm_ind, self.stm_off,
                      self.nstm_ind, self.nstm_off)
        composed = self.material + preds * 1000.0
        score = -composed  # parent POV

        in_set = torch.isin(self.pair_parent, parent_ids)
        if not in_set.any():
            return None
        pa = self.pair_a[in_set]
        pb = self.pair_b[in_set]
        sign_v = self.pair_sign[in_set]
        w_v = self.pair_w[in_set]
        parent_v = self.pair_parent[in_set]
        pair_loss = w_v * torch.nn.functional.softplus(
            -sign_v * (score[pa] - score[pb]) / RANK_TEMPERATURE_CP)
        # mean per parent, then mean over the selected parents
        uniq = torch.unique(parent_v)
        total = torch.zeros(len(uniq), device=self.device)
        counts = torch.zeros(len(uniq), device=self.device)
        pp = torch.searchsorted(uniq, parent_v)
        total.index_add_(0, pp, pair_loss)
        counts.index_add_(0, pp, torch.ones_like(pair_loss))
        return (total / counts).mean()

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
