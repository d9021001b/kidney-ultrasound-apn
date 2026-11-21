import pandas as pd
import pytest

from audit_nckuh_fullcohort_ranking_20261103 import rank_breakdown, state_breakdown


def test_rank_breakdown_preserves_ties_and_excludes_same_patient_from_state_table():
    draw = pd.DataFrame({
        "patient_alias": ["a", "a", "b", "c"],
        "label": [1, 0, 1, 0],
        "score": [.4, .4, .2, .3],
        "DMSA_APN": ["L", "L", "B", "N"],
    })
    within, across = rank_breakdown(draw)
    assert within["pairs"] == 1 and within["concordant"] == .5
    assert across["pairs"] == 3 and across["concordant"] == 1
    assert within["contribution"] + across["contribution"] == pytest.approx(.375)
    assert sum(row["pairs"] for row in state_breakdown(draw)) == 3
