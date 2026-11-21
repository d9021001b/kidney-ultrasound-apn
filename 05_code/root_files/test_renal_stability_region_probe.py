"""Focused deterministic tests for stability and central-region helpers."""
import unittest

import cv2
import numpy as np

import renal_stability_region_probe as S


class StabilityRegionProbeTests(unittest.TestCase):
    def test_central_half_is_inside_mask_and_uses_long_axis_middle(self):
        mask = np.zeros((80, 160), np.uint8)
        cv2.ellipse(mask, (80, 40), (65, 20), 0, 0, 360, 1, -1)
        central = S.central_half_mask(mask)
        self.assertTrue(np.all(central <= mask))
        self.assertGreater(central.sum(), 64)
        xs = np.nonzero(central)[1]
        self.assertGreater(xs.min(), 45)
        self.assertLess(xs.max(), 115)

    def test_central_half_rotation_invariance(self):
        mask = np.zeros((140, 140), np.uint8)
        cv2.ellipse(mask, (70, 70), (47, 18), 32, 0, 360, 1, -1)
        central = S.central_half_mask(mask)
        full_fraction = central.sum() / mask.sum()
        # A central half of the *length* contains more than half an ellipse's area.
        self.assertGreater(full_fraction, .50)
        self.assertLess(full_fraction, .70)

    def test_mask_variants_are_documented_one_pixel_operations(self):
        mask = np.zeros((50, 50), np.uint8)
        mask[10:40, 10:40] = 1
        eroded = S.perturb_mask(mask, "erode1")
        dilated = S.perturb_mask(mask, "dilate1")
        self.assertLess(eroded.sum(), mask.sum())
        self.assertGreater(dilated.sum(), mask.sum())
        np.testing.assert_array_equal(S.perturb_mask(mask, "baseline"), mask)

    def test_intensity_variants_preserve_mid_gray_and_bound_range(self):
        image = np.array([[0, 64, 128, 192, 255]], np.uint8)
        low = S.perturb_intensity(image, "contrast_low")
        high = S.perturb_intensity(image, "contrast_high")
        self.assertEqual(int(low[0, 2]), 128)
        self.assertEqual(int(high[0, 2]), 128)
        self.assertTrue(np.all(low <= 255))
        self.assertTrue(np.all(high <= 255))
        self.assertLess(int(high[0, 1]), int(low[0, 1]))
        self.assertGreater(int(high[0, 3]), int(low[0, 3]))

    def test_icc_is_one_for_identical_raters_and_small_for_disagreement(self):
        values = np.column_stack((np.arange(1, 10), np.arange(1, 10), np.arange(1, 10)))
        self.assertAlmostEqual(S.icc_2_1(values), 1.0, places=12)
        discordant = np.column_stack((np.arange(1, 10), np.arange(9, 0, -1), np.ones(9) * 5))
        self.assertLess(S.icc_2_1(discordant), .2)

    def test_stability_screen_uses_training_rows_only(self):
        baseline = np.arange(10, dtype=float)[:, None]
        stable_variants = {name: baseline.copy() for name in S.VARIANTS}
        selected_a, _ = S.stability_screen(stable_variants, ["x"], np.arange(6), "whole_kidney", 1)
        altered = {name: value.copy() for name, value in stable_variants.items()}
        altered["contrast_low"][6:, 0] *= -100
        altered["contrast_high"][6:, 0] *= 100
        selected_b, _ = S.stability_screen(altered, ["x"], np.arange(6), "whole_kidney", 1)
        np.testing.assert_array_equal(selected_a, selected_b)
        self.assertTrue(selected_a[0])

    def test_support_mask_remains_inside_outer_roi(self):
        mask = np.zeros((70, 100), np.uint8)
        cv2.ellipse(mask, (50, 35), (38, 20), 0, 0, 360, 1, -1)
        central = S.central_half_mask(mask)
        selected, kernel = S.support_mask(mask, central)
        self.assertIn(kernel, (S.EROSION_KERNEL, S.SMALL_EROSION_KERNEL))
        self.assertTrue(np.all(selected <= mask))
        self.assertGreaterEqual(selected.sum(), 64)


if __name__ == "__main__":
    unittest.main()
