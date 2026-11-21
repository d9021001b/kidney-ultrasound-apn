import unittest

import numpy as np
import pandas as pd

from run_view_graph_attention_mil_development import append_view_one_hot


class ViewEncodingTests(unittest.TestCase):
    def test_locked_view_order(self):
        values = np.zeros((4, 32), dtype=np.float32)
        manifest = pd.DataFrame({"view": ["LL", "LT", "RL", "RT"]})
        encoded = append_view_one_hot(values, manifest)
        np.testing.assert_array_equal(encoded[:, 32:], np.eye(4, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
