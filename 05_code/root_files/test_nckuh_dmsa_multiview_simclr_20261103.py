import numpy as np
import pandas as pd

from nckuh_dmsa_multiview_simclr_20261103 import positive_indices


def test_positive_pairs_stay_in_same_kidney_group():
    frame = pd.DataFrame({"kidney_view_group": ["a", "a", "b", "c", "c", "c"]})
    pairs = positive_indices(frame)
    assert np.array_equal(pairs, np.array([1, 0, 2, 4, 5, 3]))
