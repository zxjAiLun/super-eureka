import unittest
import torch
from tools.s10.train_nnue import NnueModel, ClippedReLU, NNUE_INPUTS_V1, NNUE_INPUTS_V2

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

if __name__ == '__main__':
    unittest.main()
