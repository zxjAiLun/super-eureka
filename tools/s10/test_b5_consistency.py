"""S10-B5 Repair 1 consistency tests.

Enforce that the committed calibration artifact, the committed quantized
layout artifact, and the live exporter constants describe the SAME frozen
v3 scheme. Any future drift between any two of them fails here.
"""

import json
import unittest
from pathlib import Path

from tools.s10 import export_quantized as eq

RESULTS = Path(__file__).resolve().parents[2] / "results" / "s10"


class TestSchemeV3Consistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calib = json.loads(
            (RESULTS / "s10-b5-calibration.json").read_text(encoding="utf-8"))
        cls.layout = json.loads(
            (RESULTS / "s10-b5-quantized-layout.json").read_text(
                encoding="utf-8"))

    def test_calibration_declares_scheme_v3(self):
        self.assertEqual(self.calib["scheme_version"], 3)
        self.assertFalse(self.calib["holdout_used"])

    def test_shifts_match_exporter_constants(self):
        s = self.calib["scheme"]
        self.assertEqual(s["ft"]["shift"], eq.FT_SHIFT)
        self.assertEqual(s["ft"]["container"], "i16")
        self.assertEqual(s["dense"]["container"], "i16")
        self.assertEqual(s["dense"]["weight_shift"], eq.DENSE_W_SHIFT)
        self.assertFalse(s["dense"]["input_requantization"])
        self.assertEqual(s["bias_shift"], eq.DENSE_W_SHIFT + eq.FT_SHIFT)
        self.assertEqual(s["dense_z_shift"], eq.DENSE_Z_SHIFT)
        self.assertEqual(s["qa"], eq.QA)

    def test_overflow_bounds_match_layout(self):
        c = self.calib["proven_overflow_bounds"]
        l = self.layout["proven_overflow_bounds"]
        for key in ("ft_accumulator_bound", "l1_mac_bound", "l2_mac_bound",
                    "out_mac_bound", "i32_max"):
            self.assertEqual(c[key], l[key], key)
        for key in ("ft_overflow", "l1_overflow", "l2_overflow",
                    "out_overflow"):
            self.assertFalse(c[key], key)
            self.assertFalse(l[key], key)

    def test_layout_shifts_match_exporter(self):
        self.assertEqual(self.layout["shifts"]["ft"], eq.FT_SHIFT)
        self.assertEqual(self.layout["shifts"]["dense_w"], eq.DENSE_W_SHIFT)
        self.assertEqual(self.layout["shifts"]["dense_z"], eq.DENSE_Z_SHIFT)
        self.assertEqual(self.layout["qa"], eq.QA)

    def test_no_stale_scheme_fields(self):
        """Rejected scheme remnants must not appear in committed artifacts."""
        blob = json.dumps(self.calib) + json.dumps(self.layout)
        for stale in ("qact", "dense_in", "s_ft", "s_l1", "DENSE_IN_SHIFT"):
            self.assertNotIn(stale, blob, f"stale field {stale} present")

    def test_frozen_sources_match_exporter(self):
        self.assertEqual(
            self.layout["source_checkpoint_sha256"],
            eq.FROZEN_CHECKPOINT_SHA)
        self.assertEqual(
            self.layout["source_fp32_artifact_sha256"],
            eq.FROZEN_FP32_ARTIFACT_SHA)


if __name__ == "__main__":
    unittest.main()
