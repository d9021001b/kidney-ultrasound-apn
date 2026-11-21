"""Focused tests for the locked L2-SP/early-stopping protocol helpers."""
import unittest

import numpy as np
import pandas as pd
import torch
from torch import nn

import renal_l2sp_earlystop_probe as L


class L2SPEarlyStopTests(unittest.TestCase):
    def test_inner_split_is_reproducible_stratified_outer_partition(self):
        patients = pd.DataFrame(
            {
                "patient_id": [str(i) for i in range(58)],
                "patient_alias": [f"P{i:03d}" for i in range(58)],
                "label": [0] * 27 + [1] * 31,
            }
        )
        outer = np.arange(58)
        first = L.outer_inner_split(patients, outer, 11)
        second = L.outer_inner_split(patients, outer, 11)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        self.assertEqual(len(first[0]), 46)
        self.assertEqual(len(first[1]), 12)
        self.assertFalse(set(first[0]) & set(first[1]))
        self.assertEqual(set(np.concatenate(first)), set(outer))
        self.assertEqual(int(patients.label.iloc[first[1]].sum()), 6)

    def test_l2sp_zero_at_anchor_and_positive_after_tail_update(self):
        tail = nn.Sequential(nn.Linear(4, 3), nn.BatchNorm1d(3)).eval()
        anchor = {name: parameter.detach().clone() for name, parameter in tail.named_parameters()}
        self.assertEqual(float(L.l2sp_penalty(tail, anchor, False)), 0.0)
        self.assertEqual(float(L.l2sp_penalty(tail, anchor, True)), 0.0)
        with torch.no_grad():
            next(tail.parameters()).add_(.1)
        self.assertGreater(float(L.l2sp_penalty(tail, anchor, True)), 0.0)

    def test_configurations_pair_objective_and_anchor_only(self):
        self.assertEqual(set(L.CONFIGS), {"F-CE-ZERO", "F-CE-L2SP", "F-CE-T-ZERO", "F-CE-T-L2SP"})
        self.assertFalse(L.CONFIGS["F-CE-ZERO"]["l2sp"])
        self.assertTrue(L.CONFIGS["F-CE-L2SP"]["l2sp"])
        self.assertEqual(L.CONFIGS["F-CE-ZERO"]["objective"], L.CONFIGS["F-CE-L2SP"]["objective"])
        self.assertEqual(L.CONFIGS["F-CE-T-ZERO"]["objective"], L.CONFIGS["F-CE-T-L2SP"]["objective"])

    def test_patient_rows_rejects_missing_aliases(self):
        manifest = pd.DataFrame({"patient_alias": ["P001", "P001", "P002"]})
        np.testing.assert_array_equal(L.patient_rows(manifest, np.array(["P001"])), np.array([0, 1]))
        with self.assertRaises(RuntimeError):
            L.patient_rows(manifest, np.array(["missing"]))


if __name__ == "__main__":
    unittest.main()
