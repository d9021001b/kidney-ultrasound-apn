"""Patient and side aggregation checks, without generating clinical side labels."""
import unittest
import numpy as np
import pandas as pd

from nckuh_representation_screen_20261102 import aggregate, tight_canvas


class RepresentationTests(unittest.TestCase):
    def setUp(self):
        self.manifest = pd.DataFrame(dict(patient_id=["a", "a", "a", "a", "b"], view=["LL", "LT", "RL", "RT", "LL"]))
        self.patients = pd.DataFrame(dict(patient_id=["a", "b"]))
        self.values = np.array([[1., 2.], [3., 4.], [5., 6.], [7., 8.], [9., 10.]])

    def test_same_plane_pairing_and_missing_side(self):
        actual = aggregate(self.values, self.manifest, self.patients, "bilateral")
        np.testing.assert_array_equal(actual[0, :4], [4., 4., 4., 4.])
        self.assertTrue(np.isnan(actual[1]).all())

    def test_patient_order_not_identifier_sort(self):
        patients = self.patients.iloc[::-1]
        actual = aggregate(self.values, self.manifest, patients, "summary")
        np.testing.assert_array_equal(actual[0], [9., 10., 0., 0.])

    def test_padding_does_not_create_non_kidney_pixels(self):
        gray = np.full((19, 31), 120, dtype=np.uint8)
        mask = np.zeros_like(gray, dtype=bool)
        mask[3:15, 6:28] = True
        image, canvas_mask = tight_canvas(gray, mask)
        self.assertEqual(image.shape, (224, 224, 3))
        self.assertTrue((image[canvas_mask == 0] == 0).all())


if __name__ == "__main__":
    unittest.main()
