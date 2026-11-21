import unittest
import numpy as np

from nckuh_internal_regions_20261102 import full_frame_canvas, invert_full_frame, select_images


class RegionProbeTests(unittest.TestCase):
    def test_label_free_distinct_patient_selection(self):
        frame = select_images()
        self.assertEqual(len(frame), 32)
        self.assertEqual(frame.patient_alias.nunique(), 32)
        self.assertNotIn("label", frame.columns)
        self.assertEqual(sorted(frame["view"].value_counts().tolist()), [8, 8, 8, 8])

    def test_letterbox_inverse_roundtrip_shape(self):
        gray = np.zeros((300, 470), dtype=np.uint8)
        outer = gray.copy(); outer[50:250, 100:370] = 1
        gray[outer > 0] = 120
        _, transform = full_frame_canvas(gray, outer)
        label = np.zeros((768, 1024), dtype=np.uint8)
        t = transform
        label[t["y"]:t["y"]+t["height"], t["x"]:t["x"]+t["width"]] = 3
        restored = invert_full_frame(label, transform)
        self.assertEqual(restored.shape, gray.shape)
        self.assertTrue((restored == 3).all())


if __name__ == "__main__":
    unittest.main()
