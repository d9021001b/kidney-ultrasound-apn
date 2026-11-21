import unittest

import numpy as np
import pandas as pd

from nckuh_dmsa_contralateral_probe_20261102 import build_kidney_features, classifier


class AnatomicalComparisonTests(unittest.TestCase):
    def setUp(self):
        self.images = pd.DataFrame(dict(patient_alias=['P1']*4 + ['P2']*2,
            view=['LL','LT','RL','RT','LL','LT'], DMSA_APN=['L']*4+['B']*2))
        self.kidneys = pd.DataFrame(dict(patient_alias=['P1','P1','P2'], side=['left','right','left'],
                                         kidney_alias=['K1','K2','K3'], kidney_APN=[1,0,1]))
        self.values = np.array([[5.],[3.],[2.],[4.],[7.],[8.]])

    def test_signed_opposite_but_absolute_kept(self):
        result, flags = build_kidney_features(self.values, self.images, self.kidneys, 'contra')
        np.testing.assert_equal(result[0,:2], [5,3])
        np.testing.assert_equal(result[0,2:4], [3,-1])
        np.testing.assert_equal(result[1,2:4], [-3,1])
        np.testing.assert_allclose(result[0,4:6], [3/7,-1/7], atol=1e-6)
        self.assertTrue(np.isnan(result[2,2:6]).all())
        np.testing.assert_equal(flags[2], [1,1,0,0])

    def test_no_diagnosis_dependency(self):
        first, _ = build_kidney_features(self.values, self.images, self.kidneys, 'contra')
        changed = self.images.assign(DMSA_APN='N', previous_patient_label=0, DMSA_RF=99)
        kidney = self.kidneys.assign(kidney_APN=0, patient_id='secret')
        second, _ = build_kidney_features(self.values, changed, kidney, 'contra')
        np.testing.assert_equal(first, second)

    def test_global_side_swap_is_equivariant(self):
        first, _ = build_kidney_features(self.values, self.images, self.kidneys, 'contra')
        flipped = self.images.copy()
        flipped['view'] = flipped['view'].map(lambda x: ('R' if x[0]=='L' else 'L')+x[1])
        kidney = self.kidneys.assign(side=self.kidneys.side.map({'left':'right','right':'left'}))
        second, _ = build_kidney_features(self.values, flipped, kidney, 'contra')
        np.testing.assert_equal(first, second)

    def test_row_permutation_preserves_matching(self):
        first, _ = build_kidney_features(self.values, self.images, self.kidneys, 'view')
        order = [5,2,0,4,3,1]
        images = self.images.iloc[order].reset_index(drop=True)
        second, _ = build_kidney_features(self.values[order], images, self.kidneys, 'view')
        np.testing.assert_equal(first, second)

    def test_svm_has_no_hidden_probability_cv(self):
        model = classifier('pca32_svm', 1)
        self.assertFalse(model.steps[-1][1].probability)


if __name__ == '__main__':
    unittest.main()
