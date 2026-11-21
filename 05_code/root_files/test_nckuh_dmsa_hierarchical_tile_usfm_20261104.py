import numpy as np
import pandas as pd

from nckuh_dmsa_hierarchical_paired_20261103 import kidney_scores


def test_four_state_probabilities_become_kidney_marginals() -> None:
    patients = pd.DataFrame({"patient_alias": ["P1", "P2"]})
    kidneys = pd.DataFrame({
        "patient_alias": ["P1", "P1", "P2", "P2"],
        "side": ["left", "right", "left", "right"],
    })
    probability = np.array([
        [0.1, 0.2, 0.3, 0.4],
        [0.7, 0.1, 0.1, 0.1],
    ])
    np.testing.assert_allclose(
        kidney_scores(probability, patients, kidneys),
        [0.6, 0.7, 0.2, 0.2],
    )

