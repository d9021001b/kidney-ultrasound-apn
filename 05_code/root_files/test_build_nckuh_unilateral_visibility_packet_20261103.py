import pandas as pd

import build_nckuh_unilateral_visibility_packet_20261103 as B


def test_unilateral_case_count_and_distribution():
    patients, _, _ = B.C.load_reference()
    unilateral = patients[patients.DMSA_APN.isin(["L", "R"])]
    assert len(unilateral) == 83
    assert unilateral.DMSA_APN.value_counts().to_dict() == {"L": 51, "R": 32}


def test_selected_views_never_cross_patient():
    patients, _, images = B.C.load_reference()
    alias = patients.patient_alias.iloc[0]
    selected = B.selected_views(images, alias)
    chosen = set(selected.values())
    assert chosen.issubset(set(images.loc[images.patient_alias.eq(alias), "image_id"]))
