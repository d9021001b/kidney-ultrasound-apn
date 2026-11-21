import numpy as np
import pandas as pd

import nckuh_dmsa_biomedclip_zeroshot_20261107 as study


def test_zero_shot_probability_prefers_positive_direction():
    images = np.asarray([[1.0, 0.0], [0.0, 1.0]], np.float32)
    positive = np.asarray([[1.0, 0.0]], np.float32)
    negative = np.asarray([[0.0, 1.0]], np.float32)
    score = study.zero_shot_probability(images, positive, negative, 10.0)
    assert score[0] > 0.99
    assert score[1] < 0.01


def test_kidney_pooling_respects_membership_and_method():
    images = pd.DataFrame(
        {"kidney_alias": ["A", "A", "B"], "patient_alias": ["P1", "P1", "P2"]}
    )
    kidneys = pd.DataFrame({"kidney_alias": ["B", "A"]})
    score = np.asarray([0.2, 0.8, 0.6])
    np.testing.assert_allclose(study.pool_kidneys(score, images, kidneys, "mean"), [0.6, 0.5])
    np.testing.assert_allclose(study.pool_kidneys(score, images, kidneys, "max"), [0.6, 0.8])


def test_prompt_sets_are_fixed_and_distinct():
    assert len(study.POSITIVE_PROMPTS) == 6
    assert len(study.NEGATIVE_PROMPTS) == 4
    assert set(study.POSITIVE_PROMPTS).isdisjoint(study.NEGATIVE_PROMPTS)
