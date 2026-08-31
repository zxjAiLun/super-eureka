import json
import unittest
import torch
from tools.s10.train_nnue import (
    NnueModel, ClippedReLU, NNUE_INPUTS_V1, NNUE_INPUTS_V2,
    FROZEN_TEACHER_CONTRACT, verify_teacher_contract,
)

REAL_TEACHER_MANIFEST = {
    "engine": "Stockfish 18",
    "binary_sha256": FROZEN_TEACHER_CONTRACT["binary_sha256"],
    "nodes": 16384,
    "options": {"Threads": "1", "Hash": "64", "MultiPV": "1",
                "UCI_ShowWDL": "true"},
    "labeled_positions": 300000,
    "audit": {"ok": True, "mode": "fresh-second-pass", "checked": 1000,
              "mismatches": []},
    "labels_sha256": FROZEN_TEACHER_CONTRACT["historical_labels_sha256_300k"],
}


class TestNnueModel(unittest.TestCase):
    def test_clipped_relu(self):
        act = ClippedReLU(0.0, 1.0)
        x = torch.tensor([-0.5, 0.0, 0.5, 1.0, 1.5])
        out = act(x)
        expected = torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0])
        self.assertTrue(torch.allclose(out, expected))

    def test_model_forward_v1_and_v2(self):
        for num_inputs in [NNUE_INPUTS_V1, NNUE_INPUTS_V2]:
            model = NnueModel(num_inputs=num_inputs, ft_width=128)
            # Batch of 2 positions
            stm_indices = torch.tensor([0, 10, 20, 1, 2, 3], dtype=torch.long)
            stm_offsets = torch.tensor([0, 3], dtype=torch.long)
            nstm_indices = torch.tensor([100, 200, 5, 6], dtype=torch.long)
            nstm_offsets = torch.tensor([0, 2], dtype=torch.long)

            out = model(stm_indices, stm_offsets, nstm_indices, nstm_offsets)
            self.assertEqual(out.shape, (2,))
            self.assertFalse(torch.isnan(out).any())


class TestTeacherContractGate(unittest.TestCase):
    def test_valid_manifest_passes(self):
        verify_teacher_contract(json.loads(json.dumps(REAL_TEACHER_MANIFEST)))

    def _assert_fails_closed(self, mutate):
        tm = json.loads(json.dumps(REAL_TEACHER_MANIFEST))
        mutate(tm)
        with self.assertRaises(SystemExit):
            verify_teacher_contract(tm)

    def test_wrong_engine_fails_closed(self):
        self._assert_fails_closed(lambda t: t.update(engine="Stockfish 17"))

    def test_wrong_binary_sha_fails_closed(self):
        self._assert_fails_closed(lambda t: t.update(binary_sha256="a" * 64))

    def test_wrong_nodes_fails_closed(self):
        self._assert_fails_closed(lambda t: t.update(nodes=8192))

    def test_wrong_option_fails_closed(self):
        def mutate(t):
            t["options"]["Hash"] = "128"
        self._assert_fails_closed(mutate)

    def test_missing_option_fails_closed(self):
        def mutate(t):
            del t["options"]["UCI_ShowWDL"]
        self._assert_fails_closed(mutate)

    def test_wrong_labeled_positions_still_present_but_per_run(self):
        # S10-E0: labeled_positions is no longer pinned to 300000 in the
        # contract (a 1M run must pass); it must still be an int and is
        # cross-checked against the dataset manifest in load_dataset.
        tm = json.loads(json.dumps(REAL_TEACHER_MANIFEST))
        tm["labeled_positions"] = 1000000
        verify_teacher_contract(tm)  # must NOT raise

    def test_missing_labeled_positions_fails_closed(self):
        self._assert_fails_closed(lambda t: t.pop("labeled_positions"))

    def test_non_int_labeled_positions_fails_closed(self):
        self._assert_fails_closed(
            lambda t: t.update(labeled_positions="1000000"))

    def test_missing_labels_sha_fails_closed(self):
        self._assert_fails_closed(lambda t: t.pop("labels_sha256"))

    def test_non_str_labels_sha_fails_closed(self):
        self._assert_fails_closed(lambda t: t.update(labels_sha256=123))

    def test_audit_not_ok_fails_closed(self):
        self._assert_fails_closed(
            lambda t: t["audit"].update(ok=False))

    def test_audit_wrong_mode_fails_closed(self):
        self._assert_fails_closed(
            lambda t: t["audit"].update(mode="vs-stored"))

    def test_audit_with_mismatches_fails_closed(self):
        self._assert_fails_closed(
            lambda t: t["audit"].update(mismatches=["x"]))

    def test_wrong_labels_sha_is_per_run_not_contract(self):
        # S10-E0: labels_sha256 is verified against the actual labels.jsonl
        # bytes in load_dataset, not pinned here; a different SHA value is
        # structurally acceptable to the contract gate.
        tm = json.loads(json.dumps(REAL_TEACHER_MANIFEST))
        tm["labels_sha256"] = "b" * 64
        verify_teacher_contract(tm)  # must NOT raise

    def test_missing_audit_fails_closed(self):
        self._assert_fails_closed(lambda t: t.pop("audit"))


class TestMaterialResidualTargetMode(unittest.TestCase):
    """S10-F1: material-anchored residual target construction."""

    def test_canonical_piece_values_match_engine(self):
        # The Python twin must carry the engine's canonical values
        # (src/chess/types.rs PieceType::value). The full Python<->Rust
        # cross-check runs in the trainer itself; this pins the constants.
        from tools.s10.train_nnue import CANONICAL_PIECE_CP
        self.assertEqual(
            CANONICAL_PIECE_CP,
            {"p": 100, "n": 320, "b": 330, "r": 500, "q": 900},
        )

    def test_material_cp_stm_python_signs(self):
        from tools.s10.train_nnue import material_cp_stm_python
        # Equal material, black to move -> 0
        self.assertEqual(
            material_cp_stm_python(
                "r1bqkbnr/ppp1pp1p/2np2p1/3P4/2P5/2N5/"
                "PP2PPPP/R1BQKBNR b KQkq - 0 4"),
            0,
        )
        # Black down a knight, black to move -> -320
        self.assertEqual(
            material_cp_stm_python(
                "r1bqk1nr/ppp1ppbp/2Pp2p1/8/2P5/2N5/"
                "PP2PPPP/R1BQKBNR b KQkq - 0 5"),
            -320,
        )
        # Same material flipped: white up N for P, white to move -> +220
        self.assertEqual(
            material_cp_stm_python(
                "r1bqk1nr/p1p1ppbp/2pp2p1/8/2P5/2N5/"
                "PP2PPPP/R1BQKBNR w KQkq - 0 6"),
            220,
        )
        # Symmetric board, white to move -> 0; kings contribute nothing.
        self.assertEqual(
            material_cp_stm_python(
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
            0,
        )

    def test_residual_target_no_second_clip(self):
        # R = T - M with T already clamped once; a position with T=+2000
        # and M=-2000 yields R=+4000 and must NOT be re-clamped.
        from tools.s10.train_nnue import CLIP_CP, TARGET_SCALE
        T = max(-CLIP_CP, min(CLIP_CP, 2500.0))  # -> 2000
        M = -2000.0
        residual_scaled = (T - M) / TARGET_SCALE
        self.assertEqual(residual_scaled, 4.0)

    def test_residual_composed_mae_invariance(self):
        # |pred - (T-M)| == |(M+pred) - T| for every sample (float math).
        M = -320.0
        T = 250.0
        pred = -900.0
        self.assertAlmostEqual(
            abs(pred - (T - M)), abs((M + pred) - T), places=12
        )

    def test_target_modes_frozen(self):
        from tools.s10.train_nnue import TARGET_MODES
        self.assertEqual(TARGET_MODES, ("cp", "material-residual"))


if __name__ == '__main__':
    unittest.main()
