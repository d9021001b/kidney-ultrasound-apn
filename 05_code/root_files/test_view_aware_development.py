import unittest

import numpy as np
import pandas as pd

from run_view_aware_development import build_view_slots


class FixedViewSlotTests(unittest.TestCase):
    def test_view_order_and_missing_indicator(self):
        patients = pd.DataFrame({"patient_alias": ["P001", "P002"]})
        manifest = pd.DataFrame(
            {
                "patient_alias": ["P001", "P001", "P002"],
                "view": ["RT", "LL", "LT"],
            }
        )
        values = np.array([[4.0, 40.0], [1.0, 10.0], [2.0, 20.0]], dtype=np.float32)
        result = build_view_slots(values, manifest, patients)
        np.testing.assert_array_equal(
            result[0],
            np.array([1, 10, 0, 0, 0, 0, 4, 40, 1, 0, 0, 1], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            result[1],
            np.array([0, 0, 2, 20, 0, 0, 0, 0, 0, 1, 0, 0], dtype=np.float32),
        )

    def test_duplicate_patient_view_is_rejected(self):
        patients = pd.DataFrame({"patient_alias": ["P001"]})
        manifest = pd.DataFrame({"patient_alias": ["P001", "P001"], "view": ["LL", "LL"]})
        with self.assertRaises(ValueError):
            build_view_slots(np.ones((2, 1), dtype=np.float32), manifest, patients)


if __name__ == "__main__":
    unittest.main()
