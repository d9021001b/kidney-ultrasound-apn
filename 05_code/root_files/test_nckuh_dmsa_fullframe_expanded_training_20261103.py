import numpy as np
import pandas as pd

import nckuh_dmsa_fullframe_expanded_training_20261103 as study


def test_expanded_train_excludes_every_clean_held_out_patient():
    full = pd.DataFrame({"patient_alias": [f"P{i:03d}" for i in range(210)]})
    clean = full.patient_alias.iloc[:111]
    rows = []
    for index, alias in enumerate(clean):
        role = "train" if index < 70 else "development" if index < 90 else "exploratory_test"
        rows.append({"seed": 1, "patient_alias": alias, "role": role})
    assignments = pd.DataFrame(rows)
    selected = study.expanded_train_indices(full, assignments, 1)
    assert len(selected) == 169
    held_out = set(clean.iloc[70:])
    assert not set(full.patient_alias.iloc[selected]).intersection(held_out)
    assert set(clean.iloc[:70]).issubset(set(full.patient_alias.iloc[selected]))


def test_probability_helpers_keep_valid_marginals():
    states = np.asarray([[.1, .2, .3, .4], [.5, .2, .2, .1]])
    revised = study.E.replace_extreme_probability(states, np.asarray([.75, .25]))
    left, right = study.E.kidney_marginals(revised)
    assert np.all((left >= 0) & (left <= 1))
    assert np.all((right >= 0) & (right <= 1))
    np.testing.assert_allclose(revised[:, 1:3], states[:, 1:3])
