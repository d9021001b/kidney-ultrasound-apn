import numpy as np
import pandas as pd

from nckuh_dmsa_patient_bagging import repeated_indices, stratified_patient_bootstrap


def test_patient_bootstrap_preserves_state_slots_and_whole_patients():
    patients = pd.DataFrame({
        "patient_alias": [f"P{i}" for i in range(8)],
        "DMSA_APN": ["N", "N", "L", "L", "R", "R", "B", "B"],
    })
    images = pd.DataFrame({
        "patient_alias": np.repeat(patients.patient_alias.to_numpy(), 2),
        "image": np.arange(16),
    })
    sampled = stratified_patient_bootstrap(patients, np.random.default_rng(7))
    states = pd.Series(sampled).map(patients.set_index("patient_alias").DMSA_APN)
    indices = repeated_indices(images, sampled)

    assert states.value_counts().to_dict() == {"N": 2, "L": 2, "R": 2, "B": 2}
    assert len(indices) == 2 * len(sampled)
    assert set(images.iloc[indices].patient_alias).issubset(set(sampled))
