import numpy as np
import pandas as pd

import nckuh_dmsa_uri_view_ensemble_20261106 as study


def test_combine_views_uses_logit_weighting_and_missing_fallback():
    longitudinal = np.array([0.8, 0.7, np.nan])
    transverse = np.array([0.2, np.nan, 0.3])
    result = study.combine_views(longitudinal, transverse, 0.5)
    np.testing.assert_allclose(result, [0.5, 0.7, 0.3])


def test_plane_kidney_scores_preserve_kidney_order():
    images = pd.DataFrame(
        {
            "kidney_alias": ["A", "A", "B"],
            "plane": ["longitudinal", "transverse", "longitudinal"],
        }
    )
    kidneys = pd.DataFrame({"kidney_alias": ["B", "A"]})
    result = study.plane_kidney_scores(
        np.array([0.4, np.nan, 0.8]), images, kidneys, "longitudinal"
    )
    np.testing.assert_allclose(result, [0.8, 0.4])


def test_extreme_view_weights_select_the_requested_complete_view():
    longitudinal = np.array([0.8, 0.3])
    transverse = np.array([0.2, 0.9])
    np.testing.assert_allclose(study.combine_views(longitudinal, transverse, 1.0), longitudinal)
    np.testing.assert_allclose(study.combine_views(longitudinal, transverse, 0.0), transverse)
