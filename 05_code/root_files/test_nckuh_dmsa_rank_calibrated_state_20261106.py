import numpy as np
import pandas as pd

import nckuh_dmsa_rank_calibrated_state_20261106 as study


def fixtures():
    patients = pd.DataFrame({"patient_alias": ["N", "L", "R", "B"]})
    kidneys = pd.DataFrame(
        {
            "patient_alias": np.repeat(patients.patient_alias, 2),
            "side": ["left", "right"] * 4,
            "kidney_APN": [0, 0, 1, 0, 0, 1, 1, 1],
        }
    )
    probability = np.asarray(
        [
            [0.90, 0.04, 0.04, 0.02],
            [0.05, 0.85, 0.05, 0.05],
            [0.05, 0.05, 0.85, 0.05],
            [0.05, 0.03, 0.02, 0.90],
        ]
    )
    return patients, kidneys, probability


def test_state_components_recover_hierarchical_evidence():
    _, _, probability = fixtures()
    components = study.state_components(probability)
    assert components.shape == (4, 3)
    assert components[0, 0] < 0
    assert components[3, 0] > 0
    assert components[3, 1] > components[1, 1]
    assert components[1, 2] > 0 > components[2, 2]


def test_rank_scores_respect_bilateral_and_signed_laterality():
    patients, kidneys, probability = fixtures()
    score = study.rank_scores(
        study.state_components(probability), patients, kidneys, (1.0, 1.0)
    ).reshape(4, 2)
    assert score[0].max() < score[3].min()
    assert score[1, 0] > score[1, 1]
    assert score[2, 1] > score[2, 0]


def test_weight_selection_uses_training_patients_only():
    patients, kidneys, probability = fixtures()
    weights, grid = study.select_weights(
        study.state_components(probability), patients, kidneys, np.arange(4)
    )
    assert weights in set((a, b) for a in study.WEIGHT_VALUES for b in study.WEIGHT_VALUES)
    assert len(grid) == 25
    assert grid.iloc[0].oof_auroc == 1.0
