"""Compare one-sided negative-patch supervision on the full NCKUH cohort."""
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from torch import nn

import nckuh_dmsa_fullcohort_directional as directional
import nckuh_dmsa_fullcohort_parenchyma_mil_20261103 as mil
from nckuh_dmsa_cluster_ci import cluster_intervals
from nckuh_dmsa_visibility_auxiliary import balanced_bce

OUT = mil.ROOT / "02_experiments/NCKUH_DMSA_fullcohort_negative_patches_20261103"
ARMS = ("bag_only", "all_negative_patches", "bilateral_negative_patches")


class PatchHead(mil.ParenchymaMIL):
    def __init__(self):
        super().__init__("gated")

    def forward(self, values, mask):
        hidden = self.dropout(torch.relu(self.projection(values)))
        raw = self.w(torch.tanh(self.v(hidden)) * torch.sigmoid(self.u(hidden))).squeeze(-1)
        attention = torch.softmax(raw.masked_fill(~mask, -torch.inf), dim=1)
        local = self.classifier(hidden).squeeze(-1)
        return (local * attention).sum(1), local


def negative_loss(local, valid, eligible):
    # Give each eligible kidney the same auxiliary weight regardless of bag size.
    losses = nn.functional.softplus(local).masked_fill(~valid, 0)
    return (losses[eligible].sum(1) / valid[eligible].sum(1)).mean()


def fit(arm, bags, masks, labels, states, seed):
    torch.manual_seed(seed)
    model = PatchHead().to(bags.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    eligible = (labels == 0) if arm == ARMS[1] else (labels == 0) & states
    if arm != ARMS[0] and not eligible.any():
        raise ValueError("No eligible training negatives")
    for _ in range(40):
        optimizer.zero_grad(set_to_none=True)
        logits, local = model(bags, masks)
        loss = balanced_bce(logits, labels)
        if arm != ARMS[0]:
            loss = loss + .3 * negative_loss(local, masks, eligible)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
    return model.eval()


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    torch.backends.cuda.matmul.allow_tf32 = False
    _, kidneys, images = mil.load_reference()
    qc = pd.read_csv(mil.OUT / "deidentified_patch_qc.csv")
    np.testing.assert_array_equal(qc.image_id, images.image_id)
    values, patch_patients, groups = mil.patch_data(images, kidneys)
    assignments = pd.read_csv(mil.BASE / "patient_assignments.csv")
    reference = pd.read_csv(directional.OUT / "development_predictions.csv")
    reference = reference[reference.group.eq("aligned_fusion")]
    records, coverage = [reference], []
    y = kidneys.kidney_APN.to_numpy(int)
    for seed in mil.SEEDS:
        mapping = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
        roles = kidneys.patient_alias.map(mapping)
        train, dev = np.flatnonzero(roles.eq("train")), np.flatnonzero(roles.eq("development"))
        target = kidneys.iloc[dev]
        assert target.patient_alias.nunique() == 42 and kidneys.iloc[train].patient_alias.nunique() == 126
        bags, masks, _ = mil.prepare_bags(values, patch_patients, groups, set(kidneys.iloc[train].patient_alias), seed)
        labels = torch.as_tensor(y[train], device=bags.device, dtype=torch.float32)
        states = torch.as_tensor(kidneys.iloc[train].DMSA_APN.eq("N").to_numpy(), device=bags.device)
        ref = reference[reference.seed.eq(seed)].set_index("kidney_alias").loc[target.kidney_alias]
        np.testing.assert_array_equal(ref.label, y[dev])
        np.testing.assert_array_equal(ref.patient_alias, target.patient_alias)
        coverage.append(dict(seed=seed, all_negative_kidneys=int((labels == 0).sum()),
                             bilateral_negative_kidneys=int(((labels == 0) & states).sum())))
        for arm in ARMS:
            model = fit(arm, bags[train], masks[train], labels, states, seed)
            with torch.inference_mode():
                score = model(bags[dev], masks[dev])[0].sigmoid().cpu().numpy()
            for group, s in ((arm, score), (arm + "_fusion", .75 * ref.score.to_numpy() + .25 * score)):
                records.append(target[["patient_alias", "kidney_alias", "side", "DMSA_APN"]].assign(
                    seed=seed, group=group, label=y[dev], score=s))
    pred = pd.concat(records, ignore_index=True)
    ci, delta, _ = cluster_intervals(pred, list(pred.group.unique()), mil.SEEDS, "aligned_fusion")
    _, ablation, _ = cluster_intervals(pred, list(ARMS), mil.SEEDS, "bag_only")
    OUT.mkdir()
    pred.to_csv(OUT / "development_predictions.csv", index=False)
    ci.to_csv(OUT / "development_ci.csv", index=False)
    pd.concat([delta, ablation]).to_csv(OUT / "paired_delta_ci.csv", index=False)
    pd.DataFrame(coverage).to_csv(OUT / "training_negative_coverage.csv", index=False)
    print(ci.pivot(index="group", columns="metric", values="mean").to_string())


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        run()
