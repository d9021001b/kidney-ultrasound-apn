"""Verify the current NCKUH incumbent uncertainty and bilateral pooling invariance."""
import json
import numpy as np
import pandas as pd

from nckuh_only_study_20261102 import ROOT, OUT as BASE, SEEDS, digest, write_json, weighted_metrics_batch
from nckuh_ultrasound_probe_20261102 import OUT as CANDIDATE
from nckuh_lora_bilateral_pilot_20261102 import OUT as PILOT, WEIGHT
from nckuh_representation_screen_20261102 import load_metadata


def uncertainty():
    selected = pd.read_csv(CANDIDATE / "development_predictions.csv")
    selected = selected[selected.group.isin(["bilateral_full", "bilateral_pca32"])].copy()
    reference = pd.read_csv(BASE / "predictions.csv")
    reference = reference[reference.role.eq("development") & reference.group.eq("Fixed three source fusion")]
    frame = pd.concat([selected, reference], ignore_index=True)
    groups = ["Fixed three source fusion", "bilateral_full", "bilateral_pca32"]
    aliases = sorted(frame.patient_alias.unique())
    outcomes = frame.groupby("patient_alias").label.first().reindex(aliases).to_numpy(int)
    rng = np.random.default_rng(20261102)
    positive, negative = np.flatnonzero(outcomes == 1), np.flatnonzero(outcomes == 0)
    sampled = np.c_[rng.choice(positive, (10000, len(positive)), replace=True), rng.choice(negative, (10000, len(negative)), replace=True)]
    counts = np.zeros((10000, len(aliases)))
    np.add.at(counts, (np.arange(10000)[:, None], sampled), 1)
    samples = np.zeros((10000, 3, 4))
    points = np.zeros((3, 4))
    for seed in SEEDS:
        draw = frame[frame.seed.eq(seed)]
        pivot = draw.pivot(index="patient_alias", columns="group", values="score").reindex(columns=groups)
        y = draw.groupby("patient_alias").label.first().reindex(pivot.index).to_numpy(int)
        indices = np.array([aliases.index(a) for a in pivot.index])
        weights = counts[:, indices]
        assert (weights[:, y == 0].sum(1) > 0).all() and (weights[:, y == 1].sum(1) > 0).all()
        for j in range(3):
            scores = pivot.iloc[:, j].to_numpy(float)
            samples[:, j] += weighted_metrics_batch(y, scores, weights) / 10
            points[j] += weighted_metrics_batch(y, scores, np.ones((1, len(y))))[0] / 10
    rows, differences = [], []
    for j, group in enumerate(groups):
        for k, name in enumerate(["auroc", "ap", "auprc", "brier"]):
            low, high = np.quantile(samples[:, j, k], [.025, .975])
            rows.append(dict(group=group, role="development", metric=name, mean=points[j, k], low=low, high=high, distinct_patients=len(aliases)))
            if j > 0:
                low, high = np.quantile(samples[:, j, k] - samples[:, 0, k], [.025, .975])
                differences.append(dict(group=group, baseline=groups[0], metric=name, difference=points[j, k] - points[0, k], low=low, high=high))
    pd.DataFrame(rows).to_csv(CANDIDATE / "selected_development_ci.csv", index=False)
    pd.DataFrame(differences).to_csv(CANDIDATE / "selected_development_paired_ci.csv", index=False)
    return [r for r in rows if r["group"] == "bilateral_full" and r["metric"] in ["auroc", "ap"]]


def reconstruct_frozen_head():
    import torch
    from torch.nn import functional as F
    manifest, patients = load_metadata()
    selected = json.loads((PILOT / "result_frozen_original_20261102.json").read_text(encoding="utf-8"))
    saved = torch.load(PILOT / "models/frozen_original_20261102.pth", map_location="cpu", weights_only=True)
    state = saved["head"]
    source = np.load(PILOT / "frozen_embedding.npy", mmap_mode="r")
    indices = [patients.index[patients.patient_alias.eq(r["patient_alias"])][0] for r in selected["predictions"]]
    values = np.zeros((len(indices), 4, 1536), dtype=np.float32)
    sides = np.full((len(indices), 4), -1, dtype=int)
    for i, patient_index in enumerate(indices):
        rows = np.flatnonzero(manifest.patient_id.eq(patients.patient_id.iloc[patient_index]))
        values[i, :len(rows)] = source[0, rows]
        sides[i, :len(rows)] = manifest.side.iloc[rows].map({"left": 0, "right": 1})
    with torch.inference_mode():
        node = torch.relu(F.linear(F.layer_norm(torch.tensor(values), (1536,), state["project.0.weight"], state["project.0.bias"]),
                                   state["project.1.weight"], state["project.1.bias"]))

        def pool(side):
            left, right = torch.tensor(side == 0), torch.tensor(side == 1)
            l = (node * left[..., None]).sum(1) / left.sum(1).clamp_min(1)[:, None]
            r = (node * right[..., None]).sum(1) / right.sum(1).clamp_min(1)[:, None]
            both = left.any(1) & right.any(1)
            data = torch.cat([torch.maximum(l, r), (l-r).abs()*both[:, None]], 1)
            hidden = torch.relu(F.linear(data, state["classifier.0.weight"], state["classifier.0.bias"]))
            return torch.sigmoid(F.linear(hidden, state["classifier.3.weight"], state["classifier.3.bias"])).flatten().numpy()

        actual = pool(sides)
        flipped = np.where(sides < 0, -1, 1-sides)
        swapped = pool(flipped)
    expected = np.array([r["score"] for r in selected["predictions"]])
    error = float(np.max(np.abs(actual-expected)))
    swap_error = float(np.max(np.abs(actual-swapped)))
    assert error < 1e-4 and swap_error < 1e-6
    protocol = json.loads((PILOT / "protocol.json").read_text(encoding="utf-8"))
    assert digest(WEIGHT) == protocol["checkpoint_sha256"]
    return dict(saved_prediction_reconstruction_maxabs=error, left_right_swap_maxabs=swap_error, source_checkpoint_unchanged=True)


if __name__ == "__main__":
    result = dict(best_development_ci=uncertainty(), pooling_checks=reconstruct_frozen_head(),
                  uncertainty_limit="10000 conditional stratified patient-cluster resamples; selected developmental estimates are exploratory and selection optimism is not included",
                  target_achieved=False)
    write_json(BASE / "progress_verification.json", result)
    print(json.dumps(result, indent=2))
