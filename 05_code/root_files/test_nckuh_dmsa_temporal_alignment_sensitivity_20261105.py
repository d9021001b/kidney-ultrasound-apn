import pandas as pd

import nckuh_dmsa_temporal_alignment_sensitivity_20261105 as module


def test_temporal_table_has_one_row_per_current_patient():
    frame = module.temporal_table()
    assert len(frame) == 210
    assert frame.patient_alias.nunique() == 210
    assert int(frame.temporally_aligned.sum()) == 208
    assert int(frame.gap_days.isna().sum()) == 1
    assert int(frame.gap_days.abs().gt(14).sum()) == 1


def test_metric_calculation_uses_one_row_per_kidney():
    frame = pd.DataFrame(
        {
            "patient_alias": ["P1", "P1", "P2", "P2"],
            "label": [0, 1, 0, 1],
            "score": [0.1, 0.9, 0.2, 0.8],
        }
    )
    result = module.metrics(frame)
    assert result["kidneys"] == 4
    assert result["patients"] == 2
    assert result["auroc"] == 1.0
    assert result["ap"] == 1.0
