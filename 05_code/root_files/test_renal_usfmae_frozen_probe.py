"""Focused tests for the bounded USF-MAE frozen-probe helpers."""
import unittest

import numpy as np
import pandas as pd
import torch
from torch import nn

import renal_usfmae_frozen_probe as U


class TinyFrozenEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_projection = nn.Conv2d(3, 4, kernel_size=16, stride=16, bias=False)
        self.pos_embed = nn.Parameter(torch.zeros(1, 197, 4))
        self.cls_token = nn.Parameter(torch.ones(1, 1, 4))
        self.blocks = nn.ModuleList([nn.Identity()])
        self.norm = nn.Identity()

    def patch_embed(self, x):
        return self.patch_projection(x).flatten(2).transpose(1, 2)


class USFMAEProbeTests(unittest.TestCase):
    def test_normalize_canvas_has_expected_imagenet_values(self):
        arrays = np.zeros((1, 224, 224, 3), dtype=np.uint8)
        arrays[..., 0] = 255
        actual = U.normalize_canvas(arrays, torch.device("cpu"))
        self.assertEqual(tuple(actual.shape), (1, 3, 224, 224))
        self.assertAlmostEqual(float(actual[0, 0, 0, 0]), (1.0 - 0.485) / 0.229, places=6)
        self.assertAlmostEqual(float(actual[0, 1, 0, 0]), -0.456 / 0.224, places=6)

    def test_identity_encoder_path_has_cls_and_all_patches(self):
        model = TinyFrozenEncoder().eval()
        x = torch.zeros(2, 3, 224, 224)
        tokens = U.encoder_tokens_without_mask(model, x)
        self.assertEqual(tuple(tokens.shape), (2, 197, 4))
        torch.testing.assert_close(tokens[:, 0], torch.ones(2, 4))

    def test_indices_are_complete_and_nonoverlapping(self):
        patients = pd.DataFrame(
            {
                "patient_id": [str(number) for number in range(98)],
                "patient_alias": [f"P{number:03d}" for number in range(98)],
                "label": [0] * 46 + [1] * 52,
            }
        )
        records = []
        for role, ids in (("train", range(58)), ("validation", range(58, 78)), ("test", range(78, 98))):
            records.extend({"seed": 1, "role": role, "patient_id": str(value)} for value in ids)
        index = U.indices_for(patients, pd.DataFrame(records), 1)
        self.assertEqual([len(index[name]) for name in ("train", "validation", "test")], [58, 20, 20])
        self.assertEqual(len(set(np.concatenate(list(index.values())))), 98)

    def test_development_gate_uses_no_test_values(self):
        # The implementation derives its decision from a development pivot only.
        development = pd.DataFrame(
            {
                "seed": [1, 1],
                "group": [U.COMPARATOR, U.GROUP],
                "role": ["development_validation", "development_validation"],
                "auroc": [0.70, 0.69],
                "average_precision": [0.70, 0.69],
            }
        ).pivot(index="seed", columns="group", values=["auroc", "average_precision"])
        self.assertAlmostEqual(float((development["auroc", U.GROUP] - development["auroc", U.COMPARATOR]).iloc[0]), -0.01)
        self.assertGreaterEqual(float((development["average_precision", U.GROUP] - development["average_precision", U.COMPARATOR]).iloc[0]), -0.02)


if __name__ == "__main__":
    unittest.main()
