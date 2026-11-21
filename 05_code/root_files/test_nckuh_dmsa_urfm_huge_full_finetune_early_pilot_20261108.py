import nckuh_dmsa_urfm_huge_full_finetune_early_pilot_20261108 as early


def test_early_control_configuration_is_isolated():
    original = (
        early.study.OUT,
        early.study.MIN_EPOCHS,
        early.study.PATIENCE,
        early.study.MAX_EPOCHS,
    )
    with early.configured():
        assert early.study.OUT == early.OUT
        assert early.study.MIN_EPOCHS == 1
        assert early.study.PATIENCE == 3
        assert early.study.MAX_EPOCHS == 8
    assert (
        early.study.OUT,
        early.study.MIN_EPOCHS,
        early.study.PATIENCE,
        early.study.MAX_EPOCHS,
    ) == original
