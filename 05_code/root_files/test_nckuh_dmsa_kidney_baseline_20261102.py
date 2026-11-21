import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from nckuh_dmsa_kidney_baseline_20261102 import AUDIT, OUT, SEEDS, load_reference
from nckuh_only_study_20261102 import weighted_metrics_batch


class CorrectedKidneyBaselineTests(unittest.TestCase):
    def test_clinical_missingness_is_not_zero(self):
        patients, kidneys, images = load_reference()
        self.assertEqual(len(patients), 210)
        self.assertEqual(kidneys.kidney_APN.sum(), 262)
        missing = images[images.kidney_APN.isna()]
        self.assertEqual(missing.patient_alias.nunique(), 54)
        self.assertEqual(missing.kidney_alias.nunique(), 107)
        self.assertEqual(len(missing), 207)
        self.assertTrue(missing.source_status.ne('source_side_code_available').all())

    def test_patient_grouped_random_assignments(self):
        patients, kidneys, images = load_reference()
        assignments = pd.read_csv(OUT / 'patient_assignments.csv')
        for seed in SEEDS:
            mapping = assignments[assignments.seed.eq(seed)].set_index('patient_alias').role
            self.assertEqual(len(mapping), 210)
            self.assertEqual(mapping.value_counts().to_dict(), {'train': 126, 'development': 42, 'exploratory_test': 42})
            assigned = kidneys.assign(role=kidneys.patient_alias.map(mapping))
            self.assertEqual(assigned.groupby('patient_alias').role.nunique().max(), 1)
            self.assertTrue(assigned.role.notna().all())
            self.assertEqual(set(images.loc[images.kidney_APN.notna(), 'patient_alias']), set(mapping.index))

    def test_cluster_multiplicity_keeps_contralateral_labels(self):
        # A unilateral patient's two kidneys have different labels but one weight.
        clusters = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        y = np.array([0, 1, 1, 0, 1, 1, 0, 0])
        score = np.array([.3, .7, .7, .4, .8, .8, .2, .2])
        counts = np.array([[2, 0, 1, 1], [0, 3, 0, 1]], dtype=float)
        weights = counts[:, clusters]
        got = weighted_metrics_batch(y, score, weights)
        for i in range(len(counts)):
            self.assertAlmostEqual(got[i, 0], roc_auc_score(y, score, sample_weight=weights[i]), places=12)
            self.assertAlmostEqual(got[i, 1], average_precision_score(y, score, sample_weight=weights[i]), places=12)
            self.assertTrue(np.array_equal(weights[i, ::2], weights[i, 1::2]))


if __name__ == '__main__':
    unittest.main()
