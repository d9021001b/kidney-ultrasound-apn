import numpy as np
import pandas as pd

import nckuh_dmsa_hierarchical_paired_20261103 as hierarchy
import nckuh_dmsa_nested_uri_stacking_20261105 as module


def test_uri_score_preserves_requested_kidney_order(monkeypatch):
    patients = pd.DataFrame(
        {"patient_alias": ["P1", "P2"], "DMSA_APN": ["L", "R"]}
    )
    kidneys = pd.DataFrame(
        {
            "patient_alias": ["P1", "P1", "P2", "P2"],
            "side": ["left", "right", "left", "right"],
        }
    )
    probability = np.array([[0.1, 0.7, 0.1, 0.1], [0.1, 0.2, 0.6, 0.1]])
    monkeypatch.setattr(
        hierarchy,
        "state_probabilities",
        lambda *args, **kwargs: (probability, {"mock": object()}),
    )
    score, models = module.uri_score(
        ["P1"],
        np.array([3, 0]),
        1,
        patients,
        kidneys,
        np.array([1, 2]),
        np.zeros((2, 2), dtype=np.float32),
        np.zeros((2, 1), dtype=np.float32),
    )
    np.testing.assert_allclose(score, [0.7, 0.8])
    assert "mock" in models
