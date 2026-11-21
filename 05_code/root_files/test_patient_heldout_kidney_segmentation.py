"""Focused tests for the all-image, patient-weighted segmentation endpoint."""
import unittest
import numpy as np
import pandas as pd
from finetune_patient_heldout_kidney_segmentation import overlap_metrics, prediction_mask, summarize


class SegmentationEvaluationTests(unittest.TestCase):
    def test_missing_prediction_is_zero_not_excluded(self):
        gt = np.ones((5, 5), np.uint8)
        self.assertEqual(overlap_metrics(prediction_mask([], gt.shape), gt), (0, 0))

    def test_perfect_prediction(self):
        gt = np.ones((5, 5), np.uint8)
        self.assertEqual(overlap_metrics(gt, gt), (1, 1))

    def test_largest_polygon_is_selected_without_ground_truth(self):
        small = np.array([[1, 1], [2, 1], [2, 2], [1, 2]], float)
        large = np.array([[4, 4], [8, 4], [8, 8], [4, 8]], float)
        mask = prediction_mask([small, large], (10, 10))
        self.assertEqual(mask[1, 1], 0)
        self.assertEqual(mask[5, 5], 1)

    def test_patients_get_equal_weight_despite_different_image_counts(self):
        frame = pd.DataFrame(dict(site=['CYCH']*4, patient_alias=['A','A','A','B'],
                                  iou=[1,1,1,0], dice=[1,1,1,0], instances=[1,1,1,0]))
        result = summarize(frame)
        self.assertEqual(result['patient_mean_iou'], .5)
        self.assertEqual(result['image_mean_iou'], .75)
        self.assertEqual(result['missing_predictions'], 1)


if __name__ == '__main__':
    unittest.main()
