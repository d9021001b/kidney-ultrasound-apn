import numpy as np
import pandas as pd

import nckuh_dmsa_urfm_huge_full_finetune_pilot_20261108 as pilot


def test_encoder_configuration_enables_norm_transfer_and_drop_path():
    with pilot.encoder_configuration():
        assert pilot.huge.study.TRANSFER_NORM_TO_FC_NORM is True
        assert pilot.huge.study.DROP_PATH_RATE == 0.10


def test_sampling_selects_exactly_one_image_per_kidney():
    patients = pd.DataFrame({"patient_alias": ["P1", "P2"]})
    kidneys = pd.DataFrame(
        {
            "patient_alias": ["P1", "P1", "P2"],
            "kidney_alias": ["P1_L", "P1_R", "P2_L"],
            "kidney_APN": [1, 0, 1],
        }
    )
    rows, labels, aliases = pilot.sample_one_image_per_kidney(
        np.array([0]),
        patients,
        kidneys,
        {"P1_L": np.array([1, 2]), "P1_R": np.array([3]), "P2_L": np.array([4])},
        np.random.default_rng(7),
    )
    assert len(rows) == 2
    assert labels.tolist() == [1, 0]
    assert aliases.tolist() == ["P1_L", "P1_R"]
    assert rows[0] in (1, 2)


def test_aggregate_kidneys_averages_only_matching_images():
    images = pd.DataFrame({"kidney_alias": ["K1", "K1", "K2"]})
    kidneys = pd.DataFrame({"kidney_alias": ["K2", "K1"]})
    result = pilot.aggregate_kidneys(np.array([0.2, 0.6, 0.9]), images, kidneys)
    assert np.allclose(result, [0.9, 0.4])
