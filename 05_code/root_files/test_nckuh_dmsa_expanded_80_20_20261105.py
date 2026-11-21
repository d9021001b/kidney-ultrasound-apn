import numpy as np
import pandas as pd

import nckuh_dmsa_expanded_80_20_20261105 as study


def test_registered_representations_have_exact_image_order_and_finite_values():
    _, _, images = study.local_study.load_reference()
    representations = study.image_representations(images)
    assert set(representations) == {"tile_usfm", "uri_detector", "all"}
    assert all(values.shape[0] == 820 for values in representations.values())
    assert all(np.isfinite(values).all() for values in representations.values())
    assert representations["all"].shape[1] == (
        representations["tile_usfm"].shape[1]
        + representations["uri_detector"].shape[1]
    )


def test_each_expanded_split_is_patient_disjoint_168_42():
    patients, _, _ = study.local_study.load_reference()
    assignments = pd.read_csv(study.base.OUT / "patient_assignments.csv")
    for seed in study.SEEDS:
        roles = patients.patient_alias.map(
            assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
        )
        train = set(patients.patient_alias[~roles.eq("development")])
        development = set(patients.patient_alias[roles.eq("development")])
        assert len(train) == 168 and len(development) == 42
        assert train.isdisjoint(development)
