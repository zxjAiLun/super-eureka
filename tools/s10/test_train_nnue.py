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
    "labels_sha256": FROZEN_TEACHER_CONTRACT["labels_sha256"],
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

    def test_wrong_labeled_positions_fails_closed(self):
        self._assert_fails_closed(lambda t: t.update(labeled_positions=299999))

    def test_audit_not_ok_fails_closed(self):
        self._assert_fails_closed(
            lambda t: t["audit"].update(ok=False))

    def test_audit_wrong_mode_fails_closed(self):
        self._assert_fails_closed(
            lambda t: t["audit"].update(mode="vs-stored"))

    def test_audit_with_mismatches_fails_closed(self):
        self._assert_fails_closed(
            lambda t: t["audit"].update(mismatches=["x"]))

    def test_wrong_labels_sha_fails_closed(self):
        self._assert_fails_closed(lambda t: t.update(labels_sha256="a" * 64))

    def test_missing_audit_fails_closed(self):
        self._assert_fails_closed(lambda t: t.pop("audit"))


if __name__ == '__main__':
    unittest.main()
