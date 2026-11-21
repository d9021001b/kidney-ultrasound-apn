import torch

import nckuh_dmsa_urfm_frozen_20261106 as study
import nckuh_dmsa_urfm_huge_norm_transfer_20261108 as mapped


def test_norm_transfer_is_off_by_default():
    state = {"norm.weight": torch.tensor([2.0])}
    current = {"fc_norm.weight": torch.tensor([1.0])}
    assert study.transfer_global_pool_norm(state, current) == {}
    assert "fc_norm.weight" not in state


def test_huge_norm_transfer_maps_weight_and_bias_without_changing_source():
    state = {
        "norm.weight": torch.tensor([2.0, 3.0]),
        "norm.bias": torch.tensor([4.0, 5.0]),
    }
    current = {
        "fc_norm.weight": torch.ones(2),
        "fc_norm.bias": torch.zeros(2),
    }
    with mapped.configured():
        transferred = study.transfer_global_pool_norm(state, current)
    assert transferred == {
        "fc_norm.weight": "norm.weight",
        "fc_norm.bias": "norm.bias",
    }
    assert torch.equal(state["fc_norm.weight"], state["norm.weight"])
    assert torch.equal(state["fc_norm.bias"], state["norm.bias"])
    assert state["fc_norm.weight"].data_ptr() != state["norm.weight"].data_ptr()


def test_norm_transfer_uses_separate_output_directory():
    with mapped.configured():
        assert study.OUT == mapped.OUT
        assert study.TRANSFER_NORM_TO_FC_NORM is True
