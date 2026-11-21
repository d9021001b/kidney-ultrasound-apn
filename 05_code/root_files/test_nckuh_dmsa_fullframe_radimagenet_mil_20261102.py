import nckuh_dmsa_fullframe_radimagenet_mil_20261102 as study


def test_only_last_dense_block_norm_and_head_are_trainable():
    model=study.build_model()
    names=study.trainable_parameter_names(model)
    assert names
    assert any(name.startswith("features.denseblock4") for name in names)
    assert any(name.startswith("features.norm5") for name in names)
    assert any(name.startswith("classifier") for name in names)
    assert all(name.startswith(("features.denseblock4","features.norm5","classifier")) for name in names)
