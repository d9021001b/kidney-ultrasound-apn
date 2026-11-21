import numpy as np
import pandas as pd

import nckuh_dmsa_fullframe_age_normalized_geometry_20261103 as study


def test_image_covariates_encode_age_sex_side_and_view():
    images = pd.DataFrame({"patient_alias": ["P1", "P2"], "side": ["left", "right"],
                           "view": ["LL", "RT"]})
    demographics = pd.DataFrame({"patient_alias": ["P1", "P2"], "age_years": [1.0, 3.0],
                                 "female": [0.0, 1.0]})
    values = study.image_covariates(images, demographics)
    assert values.shape == (2, 6)
    np.testing.assert_allclose(values[:, 3:], [[0, 0, 0], [1, 1, 1]])


def test_residualizer_removes_training_age_trend_without_labels():
    rng = np.random.default_rng(3)
    age = np.linspace(.2, 8, 80)
    covariates = np.c_[np.log1p(age), age, age * age, np.zeros(80), np.zeros(80), np.zeros(80)]
    values = np.c_[2 + 3 * age, -1 + .5 * age * age] + rng.normal(0, .01, (80, 2))
    residual, model = study.residualize_geometry(values, covariates, np.arange(60), 7)
    assert residual.shape == values.shape
    assert set(model) == {"geometry_imputer", "normalizer", "residual_center", "residual_scale"}
    assert abs(np.corrcoef(residual[:60, 0], age[:60])[0, 1]) < .05
