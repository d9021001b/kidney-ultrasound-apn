import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def test_nested_patient_folds_keep_both_kidneys_together():
    patients = pd.DataFrame({
        "patient_alias": [f"P{i:02d}" for i in range(20)],
        "DMSA_APN": np.repeat(["N", "L", "R", "B"], 5),
    })
    kidneys = pd.DataFrame({"patient_alias": np.repeat(patients.patient_alias, 2)})
    splitter = StratifiedKFold(5, shuffle=True, random_state=11)
    covered = set()
    for fit, validation in splitter.split(patients.patient_alias, patients.DMSA_APN):
        fit_alias = set(patients.patient_alias.iloc[fit])
        validation_alias = set(patients.patient_alias.iloc[validation])
        assert fit_alias.isdisjoint(validation_alias)
        assert kidneys.patient_alias.isin(validation_alias).sum() == 2 * len(validation_alias)
        covered.update(validation_alias)
    assert covered == set(patients.patient_alias)
