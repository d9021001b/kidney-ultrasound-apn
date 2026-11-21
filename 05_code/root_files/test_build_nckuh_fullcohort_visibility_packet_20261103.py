import build_nckuh_fullcohort_visibility_packet_20261103 as B


def test_full_cohort_count_and_view_selection():
    patients, _, images = B.C.load_reference()
    assert len(patients) == 210
    selected = B.U.selected_views(images, patients.patient_alias.iloc[0])
    assert set(selected).issubset({("left", "L"), ("left", "T"), ("right", "L"), ("right", "T")})
