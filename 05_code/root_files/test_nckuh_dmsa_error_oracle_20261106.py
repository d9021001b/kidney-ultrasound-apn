import numpy as np
import pandas as pd

import nckuh_dmsa_error_oracle_20261106 as study


def test_laterality_oracle_preserves_scores_and_orders_affected_first():
    frame = pd.DataFrame(
        {
            "patient_alias": ["P1", "P1", "P2", "P2"],
            "DMSA_APN": ["L", "L", "N", "N"],
            "side": ["left", "right", "left", "right"],
            "score": [0.2, 0.8, 0.3, 0.4],
        }
    )
    result = study.laterality_oracle(frame)
    np.testing.assert_allclose(result, [0.8, 0.2, 0.3, 0.4])
    np.testing.assert_allclose(np.sort(result), np.sort(frame.score))


def test_subgroup_assigns_four_error_components():
    frame = pd.DataFrame(
        {
            "DMSA_APN": ["B", "N", "L", "L"],
            "side": ["left", "right", "left", "right"],
        }
    )
    assert study.subgroup(frame).tolist() == [
        "B_positive",
        "N_negative",
        "U_affected",
        "U_contralateral",
    ]
