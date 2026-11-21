import numpy as np
import pandas as pd

import nckuh_dmsa_tiff_measurement_probe_20261102 as study


def test_parse_measurement_mm():
    value, confidence = study.parse_measurement_mm([("Dist A", .99), ("55.0 mm", .93)])
    assert value == 55.0 and confidence == .93
    assert np.isnan(study.parse_measurement_mm([("12.0 mm 55.0 mm", .9)])[0])
    assert np.isnan(study.parse_measurement_mm([("5.0 mm", .9)])[0])


def test_measurement_features_expose_missingness():
    frame = pd.DataFrame({"measurement_mm": [55.0, np.nan], "ocr_confidence": [.9, 0.]})
    values = study.measurement_features(frame)
    assert values.shape == (2, 4)
    assert values[0, -1] == 1 and values[1, -1] == 0


def test_tiff_index_has_study_coverage():
    index = study.tiff_index()
    patients, kidneys, images = study.matched_tiff_reference()
    counts = [len(index.get((row.patient_id, row.date_key, row.view), [])) for row in images.itertuples(index=False)]
    assert (len(patients), len(kidneys), len(images)) == (81, 159, 310)
    assert all(count == 1 for count in counts)
