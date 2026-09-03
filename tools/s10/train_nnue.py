#!/usr/bin/env python3
"""S10-A Production NNUE Foundation Training Harness.

Architecture (Production 128-32-32-1 baseline with ClippedReLU):
    - Sparse Feature Transformer: NNUE_INPUTS -> 128 (with bias)
    - Activation: ClippedReLU(0.0, 1.0)
    - Perspective Accumulators: Concat [stm_acc, nstm_acc] -> 256
    - Dense Hidden 1: Linear(256, 32) -> ClippedReLU(0.0, 1.0)
    - Dense Hidden 2: Linear(32, 32)  -> ClippedReLU(0.0, 1.0)
    - Output Head:    Linear(32, 1)   -> cp prediction (scaled by TARGET_SCALE)

The Rust engine (`bench nnue-features-batch --feature-set v1|v2`) is the SINGLE
encoding source of truth.

Fail-Closed Rules:
    - Dataset manifest SHA256 verification
    - Teacher manifest / labels.jsonl SHA256 verification
    - Position ID uniqueness and exact matching across records, labels, and exporter
    - Missing/extra labels rejection
    - FEN consistency check
    - Feature index bounds check
    - Usable position_id sets must be identical across all training runs for the same dataset
    - Strict deterministic execution: fails closed if any operation is non-deterministic
"""

from __future__ import annotations

import os
import sys

# Ensure CUBLAS deterministic workspace is configured before torch is imported / used
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import hashlib
import json
import math
import subprocess
import tempfile
import time
from pathlib import Path

import torch
import torch.nn as nn

# Enforce deterministic mode globally
torch.use_deterministic_algorithms(True, warn_only=False)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

NNUE_INPUTS_V1 = 40960
NNUE_INPUTS_V2 = 22528

CLIP_CP = 2000.0
TARGET_SCALE = 1000.0
LOSS_BETA = 0.1

# S10-F1: canonical piece values. These MUST match
# `PieceType::value` in src/chess/types.rs exactly; the trainer verifies
# this at run time against `bench material-batch` (fail closed) instead of
# trusting the Python copy alone.
CANONICAL_PIECE_CP = {"p": 100, "n": 320, "b": 330, "r": 500, "q": 900}

TARGET_MODES = ("cp", "material-residual")
DEFAULT_LR = 1e-3
DEFAULT_WD = 1e-5
DEFAULT_BATCH_SIZE = 256
DEFAULT_MAX_EPOCHS = 100
DEFAULT_PATIENCE = 15

# S10-B3/E0 frozen teacher PROVENANCE contract. The teacher *artifact*
# (engine identity, binary SHA, search options, audit mode) is frozen and
# applies to every labeling run; the *dataset scale* (labeled_positions) and
# labels SHA are per-run values: `labeled_positions` must equal the dataset
# manifest's records_total, and `labels_sha256` is verified against the
# labels.jsonl bytes on disk (see load_dataset). This generalization is what
# allows the 1M (and later) data-scale probes to train under the same
# fail-closed provenance without weakening any artifact check.
#
# S10-E2-W0 teacher rotation: the production teacher is now the official
# SF18 Windows x86-64 AVX2 build (exe c86215fa...). The E2-W0
# cross-platform qualification (2048 independent FENs, results/s10/
# s10-e2-w0-report.json) found the Windows and Linux builds
# byte-identical on every field, and the Windows relabel of the full
# nested parent 300k reproduced the old Linux B2 labels.jsonl
# byte-for-byte (SHA bcd49da1... both sides) — same teacher semantics,
# new production binary. The historical Linux SHA stays recorded below
# as the B2-era reference.
FROZEN_TEACHER_CONTRACT = {
    "engine": "Stockfish 18",
    "binary_sha256": (
        "c86215fa1977d53b82ed854540a4c7b025be4cd042276c85ba3de53fb9118911"),
    "nodes": 16384,
    "options": {
        "Threads": "1",
        "Hash": "64",
        "MultiPV": "1",
        "UCI_ShowWDL": "true",
    },
    "audit": {
        "ok": True,
        "mode": "fresh-second-pass",
        "checked": 1000,
        "mismatches": [],
    },
    # Historical reference for the 300k run (S10-B2); NOT enforced as a
    # constant anymore -- the labels SHA is checked against the actual
    # labels.jsonl hash in load_dataset.
    "historical_labels_sha256_300k": (
        "bcd49da1ece75a15591e135d5bcf6d036608b1759d6a00e639f3e344e516116f"
    ),
}


def verify_teacher_contract(teacher_manifest: dict) -> None:
    """Fail closed unless the teacher manifest matches the frozen provenance
    contract (engine identity, binary SHA, options, audit result). The
    dataset-scale fields (labeled_positions, labels_sha256) are per-run and
    are cross-checked against the dataset itself in load_dataset."""
    tm = teacher_manifest
    checks = [
        ("engine", tm.get("engine"), FROZEN_TEACHER_CONTRACT["engine"]),
        ("binary_sha256", tm.get("binary_sha256"),
         FROZEN_TEACHER_CONTRACT["binary_sha256"]),
        ("nodes", tm.get("nodes"), FROZEN_TEACHER_CONTRACT["nodes"]),
    ]
    for field, actual, expected in checks:
        if actual != expected:
            raise SystemExit(
                f"FAIL CLOSED: teacher manifest {field} {actual!r} != frozen "
                f"contract {expected!r}"
            )
    for opt_key, opt_expected in FROZEN_TEACHER_CONTRACT["options"].items():
        opt_actual = tm.get("options", {}).get(opt_key)
        if opt_actual != opt_expected:
            raise SystemExit(
                f"FAIL CLOSED: teacher manifest option {opt_key}="
                f"{opt_actual!r} != frozen contract {opt_expected!r}"
            )
    audit = tm.get("audit", {})
    frozen_audit = FROZEN_TEACHER_CONTRACT["audit"]
    if audit.get("ok") is not frozen_audit["ok"]:
        raise SystemExit("FAIL CLOSED: teacher manifest audit.ok is not true")
    if audit.get("mode") != frozen_audit["mode"]:
        raise SystemExit(
            f"FAIL CLOSED: teacher manifest audit.mode {audit.get('mode')!r} "
            f"!= {frozen_audit['mode']!r}"
        )
    if audit.get("checked") != frozen_audit["checked"]:
        raise SystemExit(
            f"FAIL CLOSED: teacher manifest audit.checked "
            f"{audit.get('checked')!r} != {frozen_audit['checked']!r}"
        )
    if audit.get("mismatches") != frozen_audit["mismatches"]:
        raise SystemExit(
            f"FAIL CLOSED: teacher manifest audit.mismatches "
            f"{audit.get('mismatches')!r} != {frozen_audit['mismatches']!r}"
        )
    # Per-run scale fields must still be PRESENT (load_dataset cross-checks
    # them against the dataset bytes).
    if not isinstance(tm.get("labeled_positions"), int):
        raise SystemExit(
            "FAIL CLOSED: teacher manifest labeled_positions missing/not int"
        )
    if not isinstance(tm.get("labels_sha256"), str):
        raise SystemExit(
            "FAIL CLOSED: teacher manifest labels_sha256 missing/not str"
        )


class ClippedReLU(nn.Module):
    def __init__(self, min_val: float = 0.0, max_val: float = 1.0):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, self.min_val, self.max_val)


class NnueModel(nn.Module):
    """Production NNUE architecture: 128-32-32-1."""

    def __init__(self, num_inputs: int, ft_width: int = 128):
        super().__init__()
        self.num_inputs = num_inputs
        self.ft_width = ft_width

        # Feature Transformer (sparse embedding table + accumulator bias)
        self.ft_weights = nn.Embedding(num_inputs, ft_width)
        self.ft_bias = nn.Parameter(torch.zeros(ft_width))

        # Dense evaluation network
        self.act1 = ClippedReLU(0.0, 1.0)
        self.l1 = nn.Linear(ft_width * 2, 32)
        self.act2 = ClippedReLU(0.0, 1.0)
        self.l2 = nn.Linear(32, 32)
        self.act3 = ClippedReLU(0.0, 1.0)
        self.out = nn.Linear(32, 1)

        self._init_weights()

    def _init_weights(self):
        # Uniform init for embedding table
        nn.init.uniform_(self.ft_weights.weight, -0.01, 0.01)
        nn.init.zeros_(self.ft_bias)
        nn.init.kaiming_uniform_(self.l1.weight, nonlinearity="relu")
        nn.init.zeros_(self.l1.bias)
        nn.init.kaiming_uniform_(self.l2.weight, nonlinearity="relu")
        nn.init.zeros_(self.l2.bias)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(
        self,
        stm_indices: torch.Tensor,
        stm_offsets: torch.Tensor,
        nstm_indices: torch.Tensor,
        nstm_offsets: torch.Tensor,
    ) -> torch.Tensor:
        # Fast vectorized sparse bag-of-features lookup via nn.functional.embedding_bag
        stm_acc = (
            nn.functional.embedding_bag(
                stm_indices, self.ft_weights.weight, stm_offsets, mode="sum"
            )
            + self.ft_bias
        )
        nstm_acc = (
            nn.functional.embedding_bag(
                nstm_indices, self.ft_weights.weight, nstm_offsets, mode="sum"
            )
            + self.ft_bias
        )

        # Apply ClippedReLU to accumulator output [stm_acc, nstm_acc] -> 256
        acc_act = self.act1(torch.cat([stm_acc, nstm_acc], dim=1))

        # Forward dense layers
        h1 = self.act2(self.l1(acc_act))
        h2 = self.act3(self.l2(h1))
        out = self.out(h2)
        return out.view(-1)


# ---------------------------------------------------------------------------
# Dataset Integrity & Fail-Closed Loading
# ---------------------------------------------------------------------------

def compute_dataset_sha(records: list[dict]) -> str:
    canonical = "".join(
        json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_dataset(dataset_dir: Path) -> dict:
    dataset_dir = Path(dataset_dir)
    manifest_path = dataset_dir / "dataset_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"FAIL CLOSED: missing {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    teacher_manifest_path = dataset_dir / "teacher_manifest.json"
    if not teacher_manifest_path.exists():
        raise SystemExit(f"FAIL CLOSED: missing {teacher_manifest_path}")
    teacher_manifest = json.loads(teacher_manifest_path.read_text(encoding="utf-8"))

    records: list[dict] = []
    for shard in sorted(dataset_dir.glob("part-*.jsonl")):
        for line in shard.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))

    ids = [r["position_id"] for r in records]
    if len(set(ids)) != len(ids):
        raise SystemExit("FAIL CLOSED: duplicate position_id in dataset")

    actual_sha = compute_dataset_sha(records)
    if actual_sha != manifest["dataset_sha256"]:
        raise SystemExit(
            f"FAIL CLOSED: dataset_sha256 mismatch {actual_sha} != {manifest['dataset_sha256']}"
        )
    if len(records) != manifest["records_total"]:
        raise SystemExit(
            f"FAIL CLOSED: record count {len(records)} != manifest {manifest['records_total']}"
        )

    labels: dict[str, dict] = {}
    labels_path = dataset_dir / "labels.jsonl"
    if not labels_path.exists():
        raise SystemExit(f"FAIL CLOSED: missing {labels_path}")
    labels_text = labels_path.read_text(encoding="utf-8")
    for line in labels_text.splitlines():
        if line.strip():
            rec = json.loads(line)
            pid = rec["position_id"]
            if pid in labels:
                raise SystemExit(f"FAIL CLOSED: duplicate position_id in labels: {pid}")
            labels[pid] = rec

    missing = set(ids) - set(labels)
    extra = set(labels) - set(ids)
    if missing:
        raise SystemExit(f"FAIL CLOSED: {len(missing)} records without labels")
    if extra:
        raise SystemExit(f"FAIL CLOSED: {len(extra)} labels without records")

    labels_sha = hashlib.sha256(labels_text.encode("utf-8")).hexdigest()
    if labels_sha != teacher_manifest.get("labels_sha256"):
        raise SystemExit(
            f"FAIL CLOSED: labels_sha256 mismatch {labels_sha} != {teacher_manifest.get('labels_sha256')}"
        )

    # Per-run scale field: the teacher must have labeled EXACTLY this
    # dataset (no more, no fewer).
    if teacher_manifest.get("labeled_positions") != manifest["records_total"]:
        raise SystemExit(
            f"FAIL CLOSED: teacher manifest labeled_positions "
            f"{teacher_manifest.get('labeled_positions')!r} != dataset "
            f"records_total {manifest['records_total']!r}"
        )

    # S10-B3/E0: fail closed unless the teacher artifact matches the frozen
    # provenance contract (engine identity, binary SHA, options, audit).
    verify_teacher_contract(teacher_manifest)

    return {
        "records": records,
        "labels": labels,
        "dataset_sha": actual_sha,
        "labels_sha": labels_sha,
        "manifest": manifest,
        "teacher_manifest": teacher_manifest,
    }


def export_features_from_engine(
    engine: Path, records: list[dict], feature_set: str
) -> dict[str, dict]:
    engine = Path(engine).resolve()
    with tempfile.TemporaryDirectory(prefix="s10-export-") as tmp:
        batch_path = Path(tmp) / "batch.txt"
        batch_path.write_text(
            "".join(f"{r['position_id']}|{r['fen']}\n" for r in records),
            encoding="utf-8",
        )
        proc = subprocess.run(
            [
                str(engine),
                "bench",
                "nnue-features-batch",
                "--batch",
                str(batch_path),
                "--feature-set",
                feature_set,
            ],
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if proc.returncode != 0:
            raise SystemExit(
                f"FAIL CLOSED: exporter exit code {proc.returncode}: {proc.stderr[:500]}"
            )

    max_dim = NNUE_INPUTS_V1 if feature_set == "v1" else NNUE_INPUTS_V2
    exported: dict[str, dict] = {}
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        pid = rec["position_id"]
        if pid in exported:
            raise SystemExit(f"FAIL CLOSED: duplicate position_id in export: {pid}")
        for idx in rec["white"] + rec["black"]:
            if idx < 0 or idx >= max_dim:
                raise SystemExit(
                    f"FAIL CLOSED: feature index {idx} out of range [0, {max_dim}) in export"
                )
        exported[pid] = rec

    if len(exported) != len(records):
        raise SystemExit(
            f"FAIL CLOSED: exporter returned {len(exported)} records for "
            f"{len(records)} inputs"
        )
    return exported


def material_cp_stm_python(fen: str) -> int:
    """Raw stm-perspective material balance from the canonical piece values.
    Python twin of the engine's `bench material-batch` formula; equality with
    the Rust integers is verified per run (fail closed)."""
    board = fen.split()[0]
    stm_white = fen.split()[1] == "w"
    w = 0
    b = 0
    for ch in board:
        if ch.isupper():
            v = CANONICAL_PIECE_CP.get(ch.lower())
            if v is not None:
                w += v
        else:
            v = CANONICAL_PIECE_CP.get(ch)
            if v is not None:
                b += v
    return (w - b) if stm_white else (b - w)


def export_material_from_engine(
    engine: Path, records: list[dict]
) -> dict[str, int]:
    """S10-F1: raw stm material balance per position, straight from the
    engine's canonical PieceType::value via `bench material-batch`. The
    Python twin is cross-checked on every record (fail closed on mismatch),
    so trainer and runtime can never silently drift apart."""
    engine = Path(engine).resolve()
    with tempfile.TemporaryDirectory(prefix="s10-material-") as tmp:
        batch_path = Path(tmp) / "batch.txt"
        batch_path.write_text(
            "".join(f"{r['position_id']}|{r['fen']}\n" for r in records),
            encoding="utf-8",
        )
        proc = subprocess.run(
            [
                str(engine),
                "bench",
                "material-batch",
                "--batch",
                str(batch_path),
            ],
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if proc.returncode != 0:
            raise SystemExit(
                f"FAIL CLOSED: material-batch exit code {proc.returncode}: "
                f"{proc.stderr[:500]}"
            )
    out: dict[str, int] = {}
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        out[rec["position_id"]] = rec["material_cp_stm"]
    if len(out) != len(records):
        raise SystemExit(
            f"FAIL CLOSED: material-batch returned {len(out)} records for "
            f"{len(records)} inputs"
        )
    # Exact Python <-> Rust cross-check on every record.
    for r in records:
        py = material_cp_stm_python(r["fen"])
        rv = out[r["position_id"]]
        if py != rv:
            raise SystemExit(
                f"FAIL CLOSED: material mismatch python={py} rust={rv} "
                f"for {r['position_id']} fen={r['fen']}"
            )
    return out


# ---------------------------------------------------------------------------
# Training Pipeline
# ---------------------------------------------------------------------------

class EncodedSplit:
    def __init__(self, items: list[dict]):
        self.items = items
        self.n = len(items)

        # Pre-compute flattened arrays and offsets for fast batched embedding_bag
        stm_indices = []
        stm_offsets = []
        nstm_indices = []
        nstm_offsets = []
        targets = []
        raw_cps = []

        cur_stm_off = 0
        cur_nstm_off = 0

        for it in items:
            stm_offsets.append(cur_stm_off)
            nstm_offsets.append(cur_nstm_off)

            stm_indices.extend(it["stm"])
            nstm_indices.extend(it["nstm"])

            cur_stm_off += len(it["stm"])
            cur_nstm_off += len(it["nstm"])

            targets.append(it["target_scaled"])
            raw_cps.append(it["target_cp"])

        self.stm_indices = torch.tensor(stm_indices, dtype=torch.long)
        self.stm_offsets = torch.tensor(stm_offsets, dtype=torch.long)
        self.nstm_indices = torch.tensor(nstm_indices, dtype=torch.long)
        self.nstm_offsets = torch.tensor(nstm_offsets, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.raw_cps = torch.tensor(raw_cps, dtype=torch.float32)


def train_and_eval(
    dataset_dir: Path,
    engine_bin: Path,
    feature_set: str,
    seed: int,
    output_dir: Path,
    lr: float = DEFAULT_LR,
    weight_decay: float = DEFAULT_WD,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    patience: int = DEFAULT_PATIENCE,
    device_name: str | None = None,
    allow_holdout: bool = False,
    target_mode: str = "cp",
    ft_width: int = 128,
) -> dict:
    if target_mode not in TARGET_MODES:
        raise SystemExit(
            f"FAIL CLOSED: unknown target mode '{target_mode}' "
            f"(expected {'|'.join(TARGET_MODES)})"
        )
    if ft_width not in (128, 256):
        raise SystemExit(
            f"FAIL CLOSED: unsupported ft_width {ft_width} "
            "(expected 128 | 256; the G1 capacity probe is frozen to "
            "these two widths)"
        )
    start_time = time.time()
    num_inputs = NNUE_INPUTS_V1 if feature_set == "v1" else NNUE_INPUTS_V2

    # Determinism / Device setup
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if device_name is None:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    # 1. Load dataset with fail-closed validation
    ds = load_dataset(dataset_dir)
    records = ds["records"]
    labels = ds["labels"]

    # 2. Filter usable records (teacher_cp_stm not None). Mate-only positions
    #    (teacher_cp_stm is None) are excluded per the frozen S10-A recipe.
    usable = []
    mate_only_by_split: dict[str, int] = {}
    for r in records:
        lbl = labels[r["position_id"]]
        cp = lbl.get("teacher_cp_stm")
        if cp is not None:
            usable.append((r, lbl))
        else:
            mate_only_by_split[r["split"]] = (
                mate_only_by_split.get(r["split"], 0) + 1
            )

    if not usable:
        raise SystemExit("FAIL CLOSED: 0 usable records with teacher_cp_stm")

    # S10-B3: Stage 1 (allow_holdout=False) exports features and constructs
    # targets ONLY for train + validation. Holdout is never computed, never
    # output, and participates in no metric during Stage 1.
    export_records = [
        r for r, _ in usable if r["split"] in ("train", "validation")
    ] + ([r for r, _ in usable if r["split"] == "holdout"]
         if allow_holdout else [])

    # 3. Export features from Rust engine
    exported = export_features_from_engine(engine_bin, export_records, feature_set)

    # S10-F1: material export (Rust canonical values, Python cross-checked).
    # Exported for BOTH modes: `cp` mode records it as provenance only.
    material_stm = export_material_from_engine(engine_bin, export_records)

    # 4. Prepare split items
    splits: dict[str, list[dict]] = {"train": [], "validation": [], "holdout": []}
    residual_targets: list[float] = []
    for r, lbl in usable:
        pid = r["position_id"]
        split = r["split"]
        if split == "holdout" and not allow_holdout:
            continue
        exp = exported[pid]
        target_cp = max(-CLIP_CP, min(CLIP_CP, float(lbl["teacher_cp_stm"])))
        if target_mode == "material-residual":
            # R = T - M; NO second clip — a full standard army is ~4000cp and
            # promotion positions go higher. The output head is a plain
            # Linear(32,1) with no clamp, so the architecture can represent
            # |R| > 2000.
            material_cp = float(material_stm[pid])
            residual_cp = target_cp - material_cp
            target_scaled = residual_cp / TARGET_SCALE
            residual_targets.append(residual_cp)
        else:
            target_scaled = target_cp / TARGET_SCALE

        stm_is_white = r["fen"].split()[1] == "w"
        stm_feats = exp["white"] if stm_is_white else exp["black"]
        nstm_feats = exp["black"] if stm_is_white else exp["white"]

        item = {
            "position_id": pid,
            "white": exp["white"],
            "black": exp["black"],
            "stm": stm_feats,
            "nstm": nstm_feats,
            "target_scaled": target_scaled,
            "target_cp": target_cp,
            "material_cp_stm": material_stm[pid],
        }
        splits[split].append(item)

    train_items = splits["train"]
    val_items = splits["validation"]
    holdout_items = splits["holdout"]

    # Feature coverage diagnostics
    train_feat_counts = torch.zeros(num_inputs, dtype=torch.int64)
    for item in train_items:
        for idx in item["white"]:
            train_feat_counts[idx] += 1
        for idx in item["black"]:
            train_feat_counts[idx] += 1

    train_observed_unique = int((train_feat_counts > 0).sum().item())
    singleton_count = int((train_feat_counts == 1).sum().item())
    le5_count = int(((train_feat_counts > 0) & (train_feat_counts <= 5)).sum().item())

    # Validation unseen diagnostics
    val_total_activations = 0
    val_unseen_activations = 0
    val_positions_with_unseen = 0

    for item in val_items:
        pos_has_unseen = False
        for idx in item["white"]:
            val_total_activations += 1
            if train_feat_counts[idx] == 0:
                val_unseen_activations += 1
                pos_has_unseen = True
        for idx in item["black"]:
            val_total_activations += 1
            if train_feat_counts[idx] == 0:
                val_unseen_activations += 1
                pos_has_unseen = True
        if pos_has_unseen:
            val_positions_with_unseen += 1

    val_unseen_activation_rate = (
        (val_unseen_activations / val_total_activations) if val_total_activations > 0 else 0.0
    )
    val_positions_with_unseen_rate = (
        (val_positions_with_unseen / len(val_items)) if len(val_items) > 0 else 0.0
    )

    # Encode splits
    val_encoded = EncodedSplit(val_items)
    val_stm_ind = val_encoded.stm_indices.to(device)
    val_stm_off = val_encoded.stm_offsets.to(device)
    val_nstm_ind = val_encoded.nstm_indices.to(device)
    val_nstm_off = val_encoded.nstm_offsets.to(device)
    val_targets = val_encoded.targets.to(device)
    val_raw_cps = val_encoded.raw_cps.to(device)
    val_materials = torch.tensor(
        [it["material_cp_stm"] for it in val_items],
        dtype=torch.float32, device=device,
    )

    # S10-F1: residual target distribution (train+validation actually
    # encoded). Recorded BEFORE training; never clipped.
    residual_target_stats = None
    if target_mode == "material-residual":
        rs = sorted(residual_targets)
        n_r = len(rs)

        def pct(p):
            return rs[min(n_r - 1, int(round(p / 100.0 * (n_r - 1))))]

        residual_target_stats = {
            "n": n_r,
            "min": rs[0],
            "p01": pct(1),
            "p05": pct(5),
            "median": pct(50),
            "p95": pct(95),
            "p99": pct(99),
            "max": rs[-1],
            "count_abs_gt_2000": sum(1 for x in rs if abs(x) > 2000),
            "count_abs_gt_3000": sum(1 for x in rs if abs(x) > 3000),
            "count_abs_gt_4000": sum(1 for x in rs if abs(x) > 4000),
        }
        print(f"residual target stats: {residual_target_stats}")

    # 5. Initialize Model & Optimizer
    model = NnueModel(num_inputs=num_inputs, ft_width=ft_width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.SmoothL1Loss(beta=LOSS_BETA)

    best_val_loss = float("inf")
    best_val_mae = float("inf")
    best_epoch = -1
    best_state_dict = None
    epochs_no_improve = 0

    g = torch.Generator()
    g.manual_seed(seed)

    history = []
    n_train = len(train_items)

    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(n_train, generator=g).tolist()
        train_loss_accum = 0.0

        for start_idx in range(0, n_train, batch_size):
            end_idx = min(start_idx + batch_size, n_train)
            batch_indices = perm[start_idx:end_idx]
            batch_items = [train_items[i] for i in batch_indices]
            b_enc = EncodedSplit(batch_items)

            stm_ind = b_enc.stm_indices.to(device)
            stm_off = b_enc.stm_offsets.to(device)
            nstm_ind = b_enc.nstm_indices.to(device)
            nstm_off = b_enc.nstm_offsets.to(device)
            targets = b_enc.targets.to(device)

            optimizer.zero_grad()
            preds = model(stm_ind, stm_off, nstm_ind, nstm_off)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()

            train_loss_accum += loss.item() * len(batch_items)

        train_loss = train_loss_accum / n_train

        # Fast vectorized validation pass
        model.eval()
        with torch.no_grad():
            preds = model(val_stm_ind, val_stm_off, val_nstm_ind, val_nstm_off)
            val_loss = criterion(preds, val_targets).item()
            pred_cp = preds * TARGET_SCALE
            if target_mode == "material-residual":
                # Composed view: M + residual prediction vs clipped teacher T.
                # |pred_residual - (T-M)| == |(M+pred_residual) - T| exactly in
                # real arithmetic; the float32 forward values are cast to
                # float64 so the two orderings agree well under 1e-9 (float32
                # rounding of M+pred vs T-M differs at ~2e-4 for |cp| ~ 2000).
                pred64 = pred_cp.double()
                mat64 = val_materials.double()
                tgt64 = val_raw_cps.double()
                composed = mat64 + pred64
                val_mae = torch.mean(
                    torch.abs(composed - tgt64)
                ).item()
                residual_mae = torch.mean(
                    torch.abs(pred64 - (tgt64 - mat64))
                ).item()
                assert abs(residual_mae - val_mae) < 1e-9, (
                    "residual/composed MAE invariance violated: "
                    f"{residual_mae} vs {val_mae}"
                )
            else:
                val_mae = torch.mean(torch.abs(pred_cp - val_raw_cps)).item()

        epoch_rec = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mae": val_mae,
        }
        history.append(epoch_rec)

        # Checkpoint strictly by validation SmoothL1 loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_mae = val_mae
            best_epoch = epoch
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    # Restore best checkpoint
    model.load_state_dict({k: v.to(device) for k, v in best_state_dict.items()})
    model.eval()

    # Re-evaluate validation set with best checkpoint to verify restored loss/MAE parity
    with torch.no_grad():
        preds = model(val_stm_ind, val_stm_off, val_nstm_ind, val_nstm_off)
        restored_val_loss = criterion(preds, val_targets).item()
        pred_cp = preds * TARGET_SCALE
        if target_mode == "material-residual":
            composed = val_materials + pred_cp
            restored_val_mae = torch.mean(
                torch.abs(composed - val_raw_cps)
            ).item()
        else:
            restored_val_mae = torch.mean(torch.abs(pred_cp - val_raw_cps)).item()

    assert abs(restored_val_loss - best_val_loss) < 1e-5, "Restored val loss parity failure"

    # Holdout evaluation (Stage 2 strictly only)
    holdout_metrics = None
    if allow_holdout:
        holdout_encoded = EncodedSplit(holdout_items)
        h_stm_ind = holdout_encoded.stm_indices.to(device)
        h_stm_off = holdout_encoded.stm_offsets.to(device)
        h_nstm_ind = holdout_encoded.nstm_indices.to(device)
        h_nstm_off = holdout_encoded.nstm_offsets.to(device)
        h_targets = holdout_encoded.targets.to(device)
        h_raw_cps = holdout_encoded.raw_cps.to(device)
        h_materials = torch.tensor(
            [it["material_cp_stm"] for it in holdout_items],
            dtype=torch.float32, device=device,
        )

        with torch.no_grad():
            preds = model(h_stm_ind, h_stm_off, h_nstm_ind, h_nstm_off)
            h_loss = criterion(preds, h_targets).item()
            pred_cp = preds * TARGET_SCALE
            if target_mode == "material-residual":
                composed = h_materials + pred_cp
                h_mae = torch.mean(torch.abs(composed - h_raw_cps)).item()
            else:
                h_mae = torch.mean(torch.abs(pred_cp - h_raw_cps)).item()

        holdout_metrics = {
            "records_evaluated": len(holdout_items),
            "holdout_loss": h_loss,
            "holdout_mae": h_mae,
        }

    # Parameter footprint calculation
    l1_in = ft_width * 2
    ft_param_count = num_inputs * ft_width + ft_width
    dense_param_count = (l1_in * 32 + 32) + (32 * 32 + 32) + (32 * 1 + 1)
    total_param_count = ft_param_count + dense_param_count
    ft_fp32_bytes = ft_param_count * 4
    total_fp32_bytes = total_param_count * 4

    elapsed_s = time.time() - start_time

    summary = {
        "feature_set": feature_set,
        "seed": seed,
        "ft_width": ft_width,
        "target_mode": target_mode,
        "material_anchor": {
            "canonical_piece_cp": CANONICAL_PIECE_CP,
            "python_rust_crosscheck": "exact, every record, fail-closed",
            "residual_target_cp": residual_target_stats,
        } if target_mode == "material-residual" else {
            "canonical_piece_cp": CANONICAL_PIECE_CP,
            "python_rust_crosscheck": "exact, every record, fail-closed",
        },
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "dataset_sha256": ds["dataset_sha"],
        "labels_sha256": ds["labels_sha"],
        "records_usable": len(usable),
        "usable_split_counts": {k: len(v) for k, v in splits.items()},
        "mate_only_excluded_by_split": mate_only_by_split,
        "records_with_teacher_cp_stm": len(usable),
        "records_with_teacher_mate": sum(mate_only_by_split.values()),
        "architecture": {
            "num_inputs": num_inputs,
            "ft_width": ft_width,
            "dense_layers": [l1_in, 32, 32, 1],
            "ft_param_count": ft_param_count,
            "dense_param_count": dense_param_count,
            "total_param_count": total_param_count,
            "ft_fp32_bytes": ft_fp32_bytes,
            "total_fp32_bytes": total_fp32_bytes,
            "ft_fp32_mib": ft_fp32_bytes / (1024 * 1024),
            "total_fp32_mib": total_fp32_bytes / (1024 * 1024),
        },
        "coverage_diagnostics": {
            "train_observed_unique": train_observed_unique,
            "train_observed_ratio": train_observed_unique / num_inputs,
            "singleton_count": singleton_count,
            "singleton_ratio": singleton_count / train_observed_unique if train_observed_unique > 0 else 0.0,
            "le5_count": le5_count,
            "le5_ratio": le5_count / train_observed_unique if train_observed_unique > 0 else 0.0,
            "val_unseen_activation_rate": val_unseen_activation_rate,
            "val_positions_with_unseen_rate": val_positions_with_unseen_rate,
        },
        "training": {
            "best_epoch": best_epoch,
            "total_epochs": len(history),
            "best_val_loss": best_val_loss,
            "best_val_mae": best_val_mae,
            "elapsed_seconds": elapsed_s,
        },
        "holdout_observed": holdout_metrics is not None,
        "holdout_metrics": holdout_metrics,
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save summary artifact
    summary_path = output_dir / f"training_summary_{feature_set}_s{seed}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Save best checkpoint model
    checkpoint_path = output_dir / f"checkpoint_{feature_set}_s{seed}.pt"
    torch.save(
        {
            "summary": summary,
            "model_state_dict": best_state_dict,
        },
        checkpoint_path,
    )

    return summary


def run_preflight(dataset_dir: Path, engine_bin: Path, feature_set: str,
                  device_name: str | None = None) -> dict:
    """S10-B3-1: pure data/training preflight. Loads and verifies the frozen
    dataset/labels/teacher provenance, counts usable vs mate-only records,
    and reports the runtime identity. No model is created and no training
    happens."""
    num_inputs = NNUE_INPUTS_V1 if feature_set == "v1" else NNUE_INPUTS_V2

    ds = load_dataset(dataset_dir)
    records = ds["records"]
    labels = ds["labels"]

    n_cp = sum(1 for r in records
               if labels[r["position_id"]].get("teacher_cp_stm") is not None)
    n_mate_only = len(records) - n_cp

    usable_by_split: dict[str, int] = {"train": 0, "validation": 0, "holdout": 0}
    mate_by_split: dict[str, int] = {}
    for r in records:
        if labels[r["position_id"]].get("teacher_cp_stm") is not None:
            usable_by_split[r["split"]] += 1
        else:
            mate_by_split[r["split"]] = mate_by_split.get(r["split"], 0) + 1

    engine_bin = Path(engine_bin).resolve()
    engine_sha = hashlib.sha256(engine_bin.read_bytes()).hexdigest()

    if device_name is None:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"

    preflight = {
        "stage": "s10_b3_preflight",
        "dataset_sha256": ds["dataset_sha"],
        "labels_sha256": ds["labels_sha"],
        "teacher_contract_verified": True,
        "records_total": len(records),
        "records_with_teacher_cp_stm": n_cp,
        "records_with_teacher_mate": n_mate_only,
        "usable_split_counts": usable_by_split,
        "mate_only_excluded_by_split": mate_by_split,
        "feature_set": feature_set,
        "feature_set_inputs": num_inputs,
        "engine_exporter_path": str(engine_bin),
        "engine_exporter_sha256": engine_sha,
        "device": device_name,
        "device_name": (torch.cuda.get_device_name(0)
                        if device_name.startswith("cuda")
                        and torch.cuda.is_available() else device_name),
        "torch_version": torch.__version__,
        "cuda_version": (torch.version.cuda
                         if torch.cuda.is_available() else None),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    return preflight


def main():
    parser = argparse.ArgumentParser(description="S10 Production NNUE Training Harness")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to dataset directory")
    parser.add_argument("--engine", type=Path, required=True, help="Path to eureka engine binary")
    parser.add_argument("--feature-set", choices=["v1", "v2"], required=True, help="Feature set representation")
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WD)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--target-mode", choices=list(TARGET_MODES), default="cp",
                        help="cp: predict clipped teacher cp directly (the "
                             "B3 recipe). material-residual: predict "
                             "T - M (no second clip); runtime composes "
                             "M + residual.")
    parser.add_argument("--ft-width", type=int, choices=[128, 256], default=128,
                        help="feature-transformer width (S10-G1 capacity "
                             "probe; 128 is the frozen production width, "
                             "256 doubles only the FT capacity)")
    parser.add_argument("--allow-holdout", action="store_true", help="Stage 2 only: evaluate holdout split")
    parser.add_argument("--preflight", action="store_true",
                        help="run data/training preflight only (no training)")

    args = parser.parse_args()

    if args.preflight:
        preflight = run_preflight(
            dataset_dir=args.dataset,
            engine_bin=args.engine,
            feature_set=args.feature_set,
            device_name=args.device,
        )
        print(json.dumps(preflight, indent=2))
        return

    summary = train_and_eval(
        dataset_dir=args.dataset,
        engine_bin=args.engine,
        feature_set=args.feature_set,
        seed=args.seed,
        output_dir=args.output,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        device_name=args.device,
        allow_holdout=args.allow_holdout,
        target_mode=args.target_mode,
        ft_width=args.ft_width,
    )
    print(
        f"Training completed for {args.feature_set} seed {args.seed} "
        f"ft_width {summary['ft_width']}: "
        f"best_epoch={summary['training']['best_epoch']} "
        f"best_val_mae={summary['training']['best_val_mae']:.3f} cp "
        f"(elapsed: {summary['training']['elapsed_seconds']:.1f}s)"
    )


if __name__ == "__main__":
    main()
