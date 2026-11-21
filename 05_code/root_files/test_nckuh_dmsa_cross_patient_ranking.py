import numpy as np
import pandas as pd

from nckuh_dmsa_cross_patient_ranking import fit_head, pair_indices


def test_matched_ranking_never_uses_same_patient_or_mismatched_acquisition():
    rows = pd.DataFrame({
        "kidney_APN": [1, 0, 0, 0, 0],
        "patient_alias": ["a", "a", "b", "c", "d"],
        "match_key": ["left:x", "right:x", "left:x", "left:x", "left:y"],
        "DMSA_APN": ["L", "L", "R", "N", "N"],
    })
    p, n, w, hard = pair_indices(rows, "matched_weighted")
    np.testing.assert_array_equal([p, n, w, hard], [[0, 0], [2, 3], [3, 1], [1, 0]])
    beta, count, hard_count = fit_head(np.array([[2.], [0.], [0.], [0.], [0.]]), rows, "matched_weighted")
    assert beta[0] > 0
    assert (count, hard_count) == (2, 1)
