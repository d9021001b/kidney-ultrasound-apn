import numpy as np
import pandas as pd
import pytest

import nckuh_dmsa_uri_bmode_sensitivity_20261106 as study


def test_bmode_rows_excludes_only_registered_doppler_images():
    images = pd.DataFrame(
        {
            "image_id": ["A", *study.DOPPLER_IMAGE_IDS, "B"],
            "patient_alias": ["P0", "P1", "P2", "P3", "P4"],
        }
    )
    values = np.arange(10).reshape(5, 2)
    filtered, manifest = study.bmode_rows(values, images)
    np.testing.assert_array_equal(filtered, values[[0, 4]])
    assert manifest.image_id.tolist() == ["A", "B"]


def test_bmode_rows_requires_all_registered_images():
    images = pd.DataFrame({"image_id": ["I0619", "I0776"]})
    with pytest.raises(ValueError, match="Missing registered Doppler images"):
        study.bmode_rows(np.zeros((2, 1)), images)


def test_fixed_candidate_group_is_quarter_fusion():
    assert "quarter_plus_current" in study.CANDIDATE_GROUP
    assert len(study.DOPPLER_IMAGE_IDS) == 3
