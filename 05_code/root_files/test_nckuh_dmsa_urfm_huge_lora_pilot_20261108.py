import torch

import nckuh_dmsa_urfm_huge_lora_pilot_20261108 as pilot


def test_huge_lora_pilot_configuration_is_isolated():
    with pilot.configured():
        assert pilot.study.OUT == pilot.OUT
        assert pilot.study.CACHE == pilot.CACHE
        assert pilot.study.SEEDS == (20261102,)
        assert pilot.study.ARMS == ("lora_rank",)
        assert pilot.study.EMBED_DIM == 1280
        assert pilot.study.CACHE_BLOCKS == 28
        assert pilot.study.MODEL_BLOCKS == 32
        assert pilot.study.urfm.TRANSFER_NORM_TO_FC_NORM is True


def test_huge_patient_bag_head_accepts_1280_dimensional_images():
    with pilot.configured():
        head = pilot.study.PatientBagHead()
        logits, present = head(
            torch.randn(2, 5, 1280),
            torch.tensor([[0, 0, 1, 1, -1], [0, 1, -1, -1, -1]]),
        )
    assert logits.shape == (2, 2)
    assert present.tolist() == [[True, True], [True, True]]
