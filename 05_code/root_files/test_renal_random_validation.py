"""Focused integrity tests for the new patient-random-sampling experiment."""
import unittest
import json
import cv2
import numpy as np
import pandas as pd
import renal_random_validation as R


class RandomValidationTests(unittest.TestCase):
    def test_mask_pixels_and_hashes(self):
        d = R.read_manifest()
        for row in d.itertuples():
            mpath = R.OUT / "masks" / f"{row.image_id}.png"
            m = cv2.imread(str(mpath), 0) > 0
            gray = cv2.imread(str(R.OUT / "native" / f"{row.image_id}.png"), 0)
            kidney = cv2.imread(str(R.OUT / "masked" / f"{row.image_id}.png"), 0)
            self.assertGreater(m.sum(), 64)
            np.testing.assert_array_equal(kidney, gray*m)
            self.assertEqual(R.sha256(mpath), row.mask_sha256)
            if row.mask_source == "existing_yolo":
                self.assertGreaterEqual(row.predicted_iou, .75)

    def test_random_patient_partitions(self):
        d = R.read_manifest()
        self.assertEqual(d.patient_id.nunique(), 98)
        self.assertEqual(len(d), 349)
        self.assertEqual(d.groupby("patient_id").label.nunique().max(), 1)
        splits = pd.read_csv(R.OUT / "random_patient_assignments.csv", dtype={"patient_id": str})
        self.assertEqual(sorted(splits.seed.unique()), R.SEEDS)
        for _, s in splits.groupby("seed"):
            self.assertEqual(s.patient_id.nunique(), 98)
            self.assertFalse(s.patient_id.duplicated().any())
            self.assertEqual(s.role.value_counts().to_dict(), dict(train=58, validation=20, test=20))
            self.assertTrue(s.groupby("role").label.nunique().eq(2).all())

    def test_true_two_dimensional_radiomics(self):
        a = R.radiomics_one({"image_id": "I0001"})
        prefixes = {k.split("_")[0] for k in a if k != "image_id"}
        self.assertEqual(prefixes, {"original", "wavelet-HH", "wavelet-HL", "wavelet-LH", "wavelet-LL"})
        self.assertEqual(len(a)-1, 467)
        self.assertTrue(np.isfinite([v for k, v in a.items() if k != "image_id"]).all())

    def test_augmentation_and_aggregation(self):
        gray = np.full((30, 50), 200, np.uint8)
        mask = np.zeros_like(gray)
        mask[5:25, 8:42] = 1
        for angle in R.ANGLES:
            a = R.canvas_image(gray, mask, angle)
            self.assertEqual(a.shape, (224, 224, 3))
            self.assertEqual(a.dtype, np.uint8)
            self.assertTrue(np.all(a[:10] == 0))
        d = pd.DataFrame({"patient_id": ["a", "a", "b"]})
        p = pd.DataFrame({"patient_id": ["a", "b"]})
        np.testing.assert_allclose(R.aggregate(np.array([[1.], [3.], [9.]]), d, p), [[2., 1.], [9., 0.]])

    def test_completed_results_if_present(self):
        path = R.OUT / "metrics_per_draw.csv"
        if not path.exists():
            self.skipTest("Evaluation not yet complete")
        m = pd.read_csv(path)
        pred = pd.read_csv(R.OUT / "patient_test_predictions.csv", dtype={"patient_id": str})
        split = pd.read_csv(R.OUT / "random_patient_assignments.csv", dtype={"patient_id": str})
        self.assertEqual(len(m), 80)
        for row in m.itertuples():
            p = pred[pred.seed.eq(row.seed) & pred.model.eq(row.model)]
            expected = set(split.loc[split.seed.eq(row.seed) & split.role.eq("test"), "patient_id"])
            self.assertEqual(set(p.patient_id), expected)
            self.assertEqual(len(p), 20)
            for name, score in R.metrics(p.label, p.score).items():
                self.assertAlmostEqual(score, getattr(row, name), places=10)
        config = json.loads((R.OUT / "protocol.json").read_text())
        self.assertEqual(config["model_grid"]["features"], [8, 16, 32, "all"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
