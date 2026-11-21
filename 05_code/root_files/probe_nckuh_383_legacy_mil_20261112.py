"""Re-fit the old four-query local MIL branch on revised patient roles."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

import nckuh_dmsa_multiquery_mil_20261103 as old_mil
from prepare_nckuh_383_contour_cohort_20261111 import OUT as REGISTRY, digest


ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "02_experiments"
FEATURES = EXP / "NCKUH_revised_383_legacy_features_20261112"
OUT = EXP / "NCKUH_revised_383_root_cause_20261112"
ARM = "cross_attention_q4"


def reproduce_old():
    patients, kidneys, images = old_mil.load_reference()
    values, patch_patients, patch_views, groups = old_mil.patch_data(images, kidneys)
    seed = 20261102
    assignments = pd.read_csv(old_mil.BASE / "patient_assignments.csv")
    mapping = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
    train_aliases = set(mapping[mapping.eq("train")].index)
    train = np.flatnonzero(kidneys.patient_alias.map(mapping).eq("train"))
    dev = np.flatnonzero(kidneys.patient_alias.map(mapping).eq("development"))
    bags, mask, views, _ = old_mil.prepare_bags(
        values, patch_patients, patch_views, groups, train_aliases, seed + 100
    )
    labels = torch.from_numpy(kidneys.kidney_APN.to_numpy(np.float32)).cuda()
    model, _ = old_mil.train_model(
        ARM, seed + 10000, bags, mask, views, labels, train, (80,), None
    )
    model.eval()
    with torch.inference_mode():
        score = torch.sigmoid(model(bags[dev], mask[dev], views[dev])[0]).cpu().numpy()
    target = kidneys.iloc[dev]
    saved = pd.read_csv(
        EXP / "NCKUH_DMSA_multiquery_mil_20261103/development_predictions.csv"
    )
    saved = saved[saved.seed.eq(seed) & saved.group.eq(ARM)].set_index("kidney_alias")
    expected = saved.loc[target.kidney_alias]
    assert np.array_equal(expected.label.to_numpy(int), target.kidney_APN.to_numpy(int))
    error = float(np.max(np.abs(score - expected.score.to_numpy())))
    if error > 1e-3:
        raise AssertionError(f"Old MIL reproduction failed: {error}")
    return error


def revised_patch_data(images, kidneys, selected):
    offsets = np.load(FEATURES / "patch_offsets.npy")
    raw = np.load(FEATURES / "radimagenet_local_float16.npy", mmap_mode="r")
    assert len(offsets) == 1363 and offsets[-1] == len(raw)
    indices = np.concatenate([np.arange(offsets[i], offsets[i + 1]) for i in selected])
    values = np.asarray(raw[indices], dtype=np.float32)
    values /= np.linalg.norm(values, axis=1, keepdims=True).clip(1e-8)
    image_index = np.repeat(np.arange(len(images)), np.diff(offsets)[selected])
    patch_patients = images.patient_alias.to_numpy()[image_index]
    patch_kidneys = images.kidney_alias.to_numpy()[image_index]
    patch_views = np.asarray([
        0 if str(view).endswith("L") else 1 for view in images.view.to_numpy()[image_index]
    ], dtype=np.int64)
    groups = [np.flatnonzero(patch_kidneys == alias) for alias in kidneys.kidney_alias]
    assert all(len(group) for group in groups)
    return values, patch_patients, patch_views, groups


def main():
    reproduction = reproduce_old()
    lock = json.loads((REGISTRY / "cohort_lock.json").read_text(encoding="utf-8"))
    assert digest(REGISTRY / "contour_image_manifest.csv") == lock["files_sha256"]["contour_image_manifest.csv"]
    manifest = pd.read_csv(REGISTRY / "contour_image_manifest.csv")
    kidney_manifest = pd.read_csv(REGISTRY / "contour_kidney_manifest.csv")
    assignments = pd.read_csv(REGISTRY / "patient_assignments.csv")
    patients = assignments[
        assignments.role.isin(("train", "development"))
        & assignments.patient_alias.isin(manifest.patient_alias)
    ].reset_index(drop=True)
    assert patients.role.value_counts().to_dict() == {"train": 204, "development": 70}
    selected = np.flatnonzero(manifest.patient_alias.isin(patients.patient_alias))
    images = manifest.iloc[selected].reset_index(drop=True)
    kidneys = kidney_manifest[kidney_manifest.patient_alias.isin(patients.patient_alias)].reset_index(drop=True)
    roles = patients.set_index("patient_alias").role
    values, patch_patients, patch_views, groups = revised_patch_data(images, kidneys, selected)
    labels_np = kidneys.kidney_APN.to_numpy(int)
    labels = torch.from_numpy(labels_np.astype(np.float32)).cuda()

    train_patients = patients[patients.role.eq("train")].reset_index(drop=True)
    fit_local, select_local = train_test_split(
        np.arange(len(train_patients)), train_size=162,
        random_state=20261111 + 6000, stratify=train_patients.DMSA_APN,
    )
    fit_aliases = set(train_patients.patient_alias.iloc[fit_local])
    selection_aliases = set(train_patients.patient_alias.iloc[select_local])
    fit = np.flatnonzero(kidneys.patient_alias.isin(fit_aliases))
    selection = np.flatnonzero(kidneys.patient_alias.isin(selection_aliases))
    bags, mask, views, _ = old_mil.prepare_bags(
        values, patch_patients, patch_views, groups, fit_aliases, 20261111
    )
    _, history = old_mil.train_model(
        ARM, 20261111, bags, mask, views, labels, fit, old_mil.EPOCHS, selection
    )
    options = [
        (-(old_mil.metric(labels_np[selection], score)["auroc"]
           + old_mil.metric(labels_np[selection], score)["ap"]) / 2, epoch)
        for epoch, score in history.items()
    ]
    _, chosen_epoch = min(options, key=lambda item: (item[0], item[1]))

    train_aliases = set(train_patients.patient_alias)
    train = np.flatnonzero(kidneys.patient_alias.isin(train_aliases))
    dev = np.flatnonzero(kidneys.patient_alias.map(roles).eq("development"))
    bags, mask, views, _ = old_mil.prepare_bags(
        values, patch_patients, patch_views, groups, train_aliases, 20261111 + 100
    )
    model, _ = old_mil.train_model(
        ARM, 20261111 + 10000, bags, mask, views, labels, train, (chosen_epoch,), None
    )
    model.eval()
    with torch.inference_mode():
        score = torch.sigmoid(model(bags[dev], mask[dev], views[dev])[0]).cpu().numpy()
    target = kidneys.iloc[dev].reset_index(drop=True)
    result = {
        "old_draw_max_reproduction_error": reproduction,
        "train_patients": len(train_patients),
        "inner_fit_patients": len(fit_aliases),
        "inner_selection_patients": len(selection_aliases),
        "chosen_epoch": chosen_epoch,
        "inner_epoch_scores": {str(epoch): old_mil.metric(labels_np[selection], value)
                               for epoch, value in history.items()},
        "development_patients": target.patient_alias.nunique(),
        "development_kidneys": len(target),
        **old_mil.metric(target.kidney_APN.to_numpy(int), score),
        "test_scored": False,
        "adaptation": "Old 100/26 inner selection scaled to 162/42 for 204 training patients; model and epoch set unchanged.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    target[["patient_alias", "kidney_alias", "side", "kidney_APN"]].assign(
        score=score
    ).to_csv(OUT / "legacy_mil_development_predictions.csv", index=False)
    (OUT / "legacy_mil.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        main()
