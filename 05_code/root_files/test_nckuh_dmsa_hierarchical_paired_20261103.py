from pathlib import Path

import numpy as np
import pandas as pd

from nckuh_dmsa_hierarchical_paired_20261103 import (
    ROOT,
    SOURCE,
    kidney_scores,
    structured_views,
)


def test_structured_views_are_symmetric_and_signed() -> None:
    pairs = np.asarray([[1.0, 4.0, 3.0, 2.0]], dtype=np.float32)
    symmetric, antisymmetric = structured_views(pairs)
    np.testing.assert_allclose(symmetric, [[2.0, 3.0, 2.0, 2.0]])
    np.testing.assert_allclose(antisymmetric, [[-2.0, 2.0]])


def test_kidney_marginals_follow_state_probabilities() -> None:
    patients = pd.DataFrame({"patient_alias": ["P001"]})
    kidneys = pd.DataFrame({
        "patient_alias": ["P001", "P001"],
        "side": ["left", "right"],
    })
    probability = np.asarray([[0.1, 0.2, 0.3, 0.4]])
    np.testing.assert_allclose(kidney_scores(probability, patients, kidneys), [0.6, 0.7])


def test_source_artifacts_are_available() -> None:
    assert ROOT.exists()
    assert (SOURCE / "radimagenet_color.npy").exists()
    assert (SOURCE / "dinov3_class.npy").exists()
    assert (SOURCE / "patient_assignments.csv").exists()
    assert (SOURCE / "instance_development_predictions.csv").exists()
