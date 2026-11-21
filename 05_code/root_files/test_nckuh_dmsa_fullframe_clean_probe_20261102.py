import zipfile

import numpy as np

import nckuh_dmsa_fullframe_clean_probe_20261102 as study


def test_archive_key_ignores_suffix():
    assert study.archive_key("x/12345678_20190101_LL-L.png") == ("12345678", "20190101", "LL")
    assert study.archive_key("x/12345678_20190101_RT-e.png") == ("12345678", "20190101", "RT")


def test_actual_archive_panel_and_descriptor_contract():
    index = study.archive_index()
    key = next(key for key, members in sorted(index.items()) if len(members) == 1)
    with zipfile.ZipFile(study.ARCHIVE) as handle:
        left, mask = study.split_archive_panel(handle.read(index[key][0]))
    assert left.shape[:2] == mask.shape and mask.dtype == np.uint8 and mask.sum() >= 1024
    roi, roi_mask = study.crop_masked(left, mask)
    gray = study.percentile_gray(roi, roi_mask)
    descriptor = study.color_descriptor(roi, roi_mask)
    assert gray.shape == roi_mask.shape and not gray[roi_mask == 0].any()
    assert descriptor.shape == (45,) and np.isfinite(descriptor).all()


def test_matched_reference_denominators():
    patients, kidneys, images = study.matched_reference()
    assert (len(patients), len(kidneys), len(images)) == (111, 212, 418)
