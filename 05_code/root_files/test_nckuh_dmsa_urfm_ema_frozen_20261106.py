import nckuh_dmsa_urfm_ema_frozen_20261106 as study


def test_ema_ablation_has_a_separate_output_directory():
    assert study.OUT.name == "NCKUH_DMSA_urfm_ema_frozen_20261106"
    assert study.OUT != study.online.OUT


def test_ema_loader_is_distinct_from_online_loader():
    assert study.load_ema_encoder is not study.online.load_encoder
