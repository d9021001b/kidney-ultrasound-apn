"""Tests for the development-only DINOv3 single-encoder gate."""
from __future__ import annotations

import unittest

import pandas as pd

import renal_dinov3_single_encoder as D


class Dinov3SingleEncoderTests(unittest.TestCase):
    def metrics(self, candidate_auroc, candidate_ap):
        rows = []
        for seed in range(10):
            rows.append({"seed": seed, "group": D.BASELINE, "auroc": 0.70, "average_precision": 0.75})
            rows.append({"seed": seed, "group": D.CANDIDATE, "auroc": candidate_auroc, "average_precision": candidate_ap})
        return pd.DataFrame(rows)

    def test_gate_allows_a_candidate_within_both_margins(self):
        result = D.advancement_decision(self.metrics(0.68, 0.73))
        self.assertTrue(result["advance_to_fixed_vote"])

    def test_gate_blocks_a_candidate_below_either_margin(self):
        result = D.advancement_decision(self.metrics(0.679, 0.75))
        self.assertFalse(result["advance_to_fixed_vote"])

    def test_candidate_identity_is_explicit(self):
        self.assertEqual(D.CANDIDATE, "dinov3_vith16plus")
        self.assertEqual(D.DINO3.name, "dinov3_vith16plus_weights_20260913")
        self.assertEqual(D.GROUPS, (D.BASELINE, D.CANDIDATE))


if __name__ == "__main__":
    unittest.main()
