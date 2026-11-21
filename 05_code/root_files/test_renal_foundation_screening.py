"""Focused checks for the locked frozen-encoder screening protocol."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

import renal_foundation_screening as F


class FoundationScreeningTests(unittest.TestCase):
    def test_every_named_model_is_registered_or_pending(self):
        self.assertEqual(set(F.GROUPS), {"radiomics_percentile", "radimagenet_percentile", "dinov2_vitl14_reg4", "radimagenet_inception_v3"})
        ids = {item["id"] for item in F.REFERENCES}
        self.assertTrue({"dinov2", "dinov3_pending_access", "registers", "radimagenet", "inception", "usf_mae_pending", "openus_pending"} <= ids)

    def test_fixed_vote_is_an_unweighted_mean(self):
        p = {name: np.array([0.1, 0.9]) for name in F.GROUPS}
        p["dinov2_vitl14_reg4"] = np.array([0.7, 0.3])
        voted = F.fixed_votes(p)
        np.testing.assert_allclose(voted["fixed_radiomics_dense"], [0.1, 0.9])
        np.testing.assert_allclose(voted["fixed_radiomics_dense_dinov2"], [0.3, 0.7])

    def test_stage_decision_requires_all_three_development_conditions(self):
        rows = []
        for seed in range(10):
            rows.append({"seed": seed, "workflow": F.BASELINE, "auroc": 0.60, "average_precision": 0.60})
            rows.append({"seed": seed, "workflow": "fixed_radiomics_dense_dinov2", "auroc": 0.66 if seed < 6 else 0.59, "average_precision": 0.65 if seed < 6 else 0.59})
            rows.append({"seed": seed, "workflow": "fixed_radiomics_dense_inception", "auroc": 0.64, "average_precision": 0.61})
        decisions = {item["candidate"]: item for item in F.stage_decisions(pd.DataFrame(rows))}
        self.assertTrue(decisions["dinov2_vitl14_reg4"]["advance"])
        self.assertFalse(decisions["radimagenet_inception_v3"]["advance"])

    def test_candidate_decisions_require_development_rows(self):
        rows = pd.DataFrame([
            {"seed": 1, "workflow": F.BASELINE, "auroc": 0.5, "average_precision": 0.5},
            {"seed": 1, "workflow": "fixed_radiomics_dense_dinov2", "auroc": 0.8, "average_precision": 0.8},
            {"seed": 1, "workflow": "fixed_radiomics_dense_inception", "auroc": 0.8, "average_precision": 0.8},
        ])
        decisions = F.stage_decisions(rows)
        self.assertEqual(len(decisions), 2)
        self.assertEqual(decisions[0]["data_role"], "development_validation")


if __name__ == "__main__":
    unittest.main()
