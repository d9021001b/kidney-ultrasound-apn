from __future__ import annotations

import math

from analyze_locked_unique_patient_sensitivity import (
    FINAL,
    PAIR,
    collapse_to_equal_patients,
    load_and_validate,
    paired_estimands,
    paired_table,
    summarize_groups,
)
import numpy as np


def test_equal_patient_locked_sensitivity() -> None:
    collapsed = collapse_to_equal_patients(load_and_validate())
    summary = summarize_groups(collapsed).set_index("group")
    paired = paired_table(collapsed)
    observed = paired_estimands(paired, np.ones(len(paired), dtype=int))

    assert len(collapsed) == 6 * 87
    assert len(paired) == 87
    assert paired.label.value_counts().to_dict() == {1: 46, 0: 41}
    assert paired.test_draw_appearances.min() == 1
    assert paired.test_draw_appearances.median() == 2
    assert paired.test_draw_appearances.max() == 5
    assert math.isclose(summary.loc[FINAL, "auroc"], 0.7751855779427359)
    assert math.isclose(summary.loc[FINAL, "average_precision"], 0.7883745652109124)
    assert math.isclose(summary.loc[PAIR, "auroc"], 0.7327677624602332)
    assert math.isclose(summary.loc[PAIR, "average_precision"], 0.7686821748802997)
    assert math.isclose(observed["delta_auroc"], 0.0424178154825027)
    assert math.isclose(observed["delta_average_precision"], 0.0196923903306127)
