from __future__ import annotations

import numpy as np
import pandas as pd
import torch

import original_nine_model_comparison as M


def test_exact_nine_paper_models() -> None:
    assert [spec["paper_name"] for spec in M.MODEL_SPECS.values()] == [
        "VGG16",
        "VGG19",
        "ResNet50",
        "ResNet101",
        "DenseNet121",
        "MobileNet",
        "Xception",
        "InceptionV3",
        "InceptionResNetV2",
    ]
    assert len(M.PAPER_TABLE) == 9


def test_fixed_patient_roles_are_disjoint() -> None:
    patients = pd.read_csv(M.SOURCE / "patients.csv", dtype={"patient_id": str})
    assignments = pd.read_csv(M.SOURCE / "random_patient_assignments.csv", dtype={"patient_id": str})
    for seed in M.R.SEEDS:
        indices = M.cohort_indices(patients, assignments, seed)
        assert len(set(np.concatenate(list(indices.values())))) == 98


def test_dense_legacy_key_repair() -> None:
    source = {
        "features.denseblock1.denselayer1.norm.1.weight": torch.ones(2),
        "features.conv0.weight": torch.ones(1),
    }
    fixed = M.repair_densenet_keys(source)
    assert "features.denseblock1.denselayer1.norm1.weight" in fixed
    assert "features.denseblock1.denselayer1.norm.1.weight" not in fixed
    assert "features.conv0.weight" in fixed


def test_triplet_mining_uses_different_class_negative() -> None:
    embeddings = torch.nn.functional.normalize(torch.randn(8, 4), dim=1)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    positive, negative, fraction = M.mine_triplets(embeddings, labels, torch.Generator().manual_seed(7))
    assert torch.all(labels[positive] == labels)
    assert torch.all(labels[negative] != labels)
    assert 0.0 <= fraction <= 1.0


def test_score_row_reports_ranking_metrics() -> None:
    y = np.array([0, 0, 1, 1])
    score = np.array([-2.0, -1.0, 1.0, 2.0])
    predicted = np.array([0, 0, 1, 1])
    row = M.score_row("vgg16", "frozen_pca2_svm", 1, "validation", y, score, predicted)
    assert row["auroc"] == 1.0
    assert row["average_precision"] == 1.0
    assert row["role"] == "development_validation"
