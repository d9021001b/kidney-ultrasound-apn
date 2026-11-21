import numpy as np

import nckuh_dmsa_fullcohort_negative_memory_20261103 as M


def test_distance_summary_is_fixed_and_finite():
    values = M.summarize_distances(np.asarray([-1, 0, 1, 2, 3], dtype=float))
    assert values.shape == (8,)
    assert np.isfinite(values).all()
    assert values[-1] == .6


def test_reference_images_match_patch_cache_metadata():
    _, _, images = M.C.load_reference()
    metadata = M.pd.read_csv(M.META, dtype={"patient_id": str})
    assert len(images) == 820
    assert set(images.image_id).issubset(set(metadata.image_id))
