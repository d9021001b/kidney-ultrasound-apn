import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

import nckuh_dmsa_fullcohort_fullframe_20261103 as study


class FullCohortFullFrameTest(unittest.TestCase):
    def test_source_plan_covers_every_labeled_image(self):
        plan = study.source_plan()
        self.assertEqual(len(plan), 820)
        self.assertEqual(plan.image_id.nunique(), 820)
        self.assertEqual(plan.patient_alias.nunique(), 210)
        self.assertLessEqual(int((plan.source_type == "legacy_contour_crop").sum()), 3)

    def test_panel_split_selects_largest_mask_component(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "1_20200101_LL.png"
            panel = np.zeros((40, 100, 3), np.uint8)
            panel[:, :50] = 90
            panel[2:5, 52:55] = 255
            panel[5:35, 65:95] = 255
            cv2.imwrite(str(path), panel)
            image, mask = study.split_panel(path)
            self.assertEqual(image.shape, (40, 50, 3))
            self.assertEqual(int(mask.sum()), 900)

    def test_percentile_image_is_zero_outside_mask(self):
        image = np.arange(100, dtype=np.uint8).reshape(10, 10)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        mask = np.zeros((10, 10), np.uint8)
        mask[2:8, 3:9] = 1
        output = study.percentile_gray(image, mask)
        self.assertFalse(np.any(output[mask == 0]))
        self.assertGreater(int(output[mask > 0].max()), 0)

    def test_kidney_orientation_places_target_side_first(self):
        patients = __import__("pandas").DataFrame({"patient_alias": ["P1"]})
        kidneys = __import__("pandas").DataFrame({
            "patient_alias": ["P1", "P1"],
            "side": ["left", "right"],
        })
        pair = np.array([[1.0, 2.0, 10.0, 20.0]])
        oriented = study.orient_kidneys(pair, patients, kidneys)
        np.testing.assert_array_equal(oriented[0], [1.0, 2.0, 10.0, 20.0])
        np.testing.assert_array_equal(oriented[1], [10.0, 20.0, 1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
