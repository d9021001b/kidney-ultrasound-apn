"""Matched single-task versus sparse report-auxiliary multitask heads."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn

from evaluate_nckuh_383_echo_report_20261112 import patient_key, report_mapping, side_features
from evaluate_nckuh_383_oof_stacking_20261112 import make_geometry
from evaluate_nckuh_383_view_paired_20261112 import paired_features


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "02_experiments/NCKUH_revised_383_audit_20261111"
DINO = ROOT / "02_experiments/NCKUH_revised_383_dinov3_20261111"
GEOMETRY = ROOT / "02_experiments/NCKUH_revised_383_relative_geometry_20261112"
OUT = ROOT / "02_experiments/NCKUH_revised_383_multitask_report_20261112"
SEEDS = (20261112, 20261113, 20261114)
CHECKPOINTS = (10, 25, 50, 100)


class Head(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, 64), nn.GELU(), nn.Dropout(.2), nn.Linear(64, 4))

    def forward(self, values):
        return self.net(values)


def transform(train, evaluation, fit):
    outputs_train, outputs_eval = [], []
    for values, components in ((train["dino"], 32), (train["geometry"], 16)):
        imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(values[fit])
        filled = imputer.transform(values)
        scaler = StandardScaler().fit(filled[fit]); standardized = scaler.transform(filled)
        pca = PCA(components, svd_solver="randomized", random_state=20261112).fit(standardized[fit])
        outputs_train.append(pca.transform(standardized).astype(np.float32))
    demo_scaler = StandardScaler().fit(train["demo"][fit])
    outputs_train.append(demo_scaler.transform(train["demo"]).astype(np.float32))
    return np.concatenate(outputs_train, axis=1)


def train_predict(x, y, auxiliary, available, fit, evaluation, epochs, seed, multitask, checkpoints=()):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = Head(x.shape[1]).cuda(); optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    rng = np.random.default_rng(seed)
    class_weight = np.ones(len(y), np.float32)
    for label in (0, 1):
        selected = fit[y[fit] == label]
        class_weight[selected] = len(fit) / (2 * max(len(selected), 1))
    records = {}
    for epoch in range(1, epochs + 1):
        model.train(); order = rng.permutation(fit)
        for start in range(0, len(order), 64):
            index = order[start:start + 64]
            values = torch.from_numpy(x[index]).cuda()
            values = values + torch.randn_like(values) * .02
            logits = model(values)
            target = torch.from_numpy(y[index].astype(np.float32)).cuda()
            weight = torch.from_numpy(class_weight[index]).cuda()
            primary = (nn.functional.binary_cross_entropy_with_logits(logits[:, 0], target,
                                                                       reduction="none") * weight).mean()
            loss = primary
            if multitask:
                mask = available[index]
                if mask.any():
                    aux_target = torch.from_numpy(auxiliary[index][mask].astype(np.float32)).cuda()
                    aux_loss = nn.functional.binary_cross_entropy_with_logits(logits[mask, 1:], aux_target)
                    loss = loss + .2 * aux_loss
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        if epoch in checkpoints:
            model.eval()
            with torch.inference_mode():
                records[epoch] = torch.sigmoid(model(torch.from_numpy(x[evaluation]).cuda())[:, 0]).cpu().numpy()
    model.eval()
    with torch.inference_mode():
        final = torch.sigmoid(model(torch.from_numpy(x[evaluation]).cuda())[:, 0]).cpu().numpy()
    return records, final


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "development_metrics.csv").exists():
        raise FileExistsError("Preserve multitask experiment")
    images = pd.read_csv(REGISTRY / "contour_image_manifest.csv")
    kidneys = pd.read_csv(REGISTRY / "contour_kidney_manifest.csv")
    roles = pd.read_csv(REGISTRY / "patient_assignments.csv")
    kidneys = kidneys.merge(roles[["patient_alias", "role", "DMSA_APN"]], on="patient_alias", validate="many_to_one")
    dino = paired_features(np.load(DINO / "dinov3_class.npy"), images, kidneys)
    geometry = make_geometry(np.load(GEOMETRY / "relative_geometry_image.npy"), images, kidneys)
    linkage = pd.read_csv(REGISTRY / "patient_linkage_private.csv", dtype=str)
    aliases = {patient_key(row.patient_id): row.patient_alias for row in linkage.itertuples(index=False)}
    reports = {aliases[key]: value[1] for key, value in report_mapping().items() if key in aliases}
    auxiliary = np.zeros((len(kidneys), 3), np.float32); available = np.zeros(len(kidneys), bool)
    for i, row in enumerate(kidneys.itertuples(index=False)):
        if row.patient_alias in reports:
            auxiliary[i] = side_features(reports[row.patient_alias], row.side)[:3]
            available[i] = True
    active = roles[roles.patient_alias.isin(kidneys.patient_alias)].reset_index(drop=True)
    from evaluate_nckuh_383_clinical_hierarchy_20261111 import clinical_table
    demo = clinical_table(active.patient_alias); lookup = {alias: demo[i] for i, alias in enumerate(active.patient_alias)}
    demographics = np.asarray([lookup[alias] for alias in kidneys.patient_alias], np.float32)
    raw = {"dino": dino, "geometry": geometry, "demo": demographics}
    y = kidneys.kidney_APN.to_numpy(int)
    train_kidneys = np.flatnonzero(kidneys.role.eq("train")); dev = np.flatnonzero(kidneys.role.eq("development"))
    train_patients = active[active.role.eq("train")].reset_index(drop=True)
    predictions, selections = [], []
    arm_scores = {"single_task": [], "report_aux_multitask": []}
    for seed in SEEDS:
        fit_p, select_p = train_test_split(np.arange(len(train_patients)), train_size=160,
                                           stratify=train_patients.DMSA_APN, random_state=seed)
        fit_aliases = set(train_patients.patient_alias.iloc[fit_p]); select_aliases = set(train_patients.patient_alias.iloc[select_p])
        fit = np.flatnonzero(kidneys.patient_alias.isin(fit_aliases)); select = np.flatnonzero(kidneys.patient_alias.isin(select_aliases))
        inner_x = transform(raw, None, fit)
        final_x = transform(raw, None, train_kidneys)
        for arm, multitask in (("single_task", False), ("report_aux_multitask", True)):
            records, _ = train_predict(inner_x, y, auxiliary, available, fit, select, max(CHECKPOINTS),
                                       seed, multitask, set(CHECKPOINTS))
            rows = []
            for epoch, score in records.items():
                auc = roc_auc_score(y[select], score); ap = average_precision_score(y[select], score)
                rows.append((min(auc, ap), epoch, auc, ap))
            best = max(row[0] for row in rows); selected_epoch = min(row[1] for row in rows if row[0] == best)
            selections.extend({"seed": seed, "arm": arm, "epoch": epoch, "auroc": auc, "ap": ap,
                               "selected": epoch == selected_epoch} for _, epoch, auc, ap in rows)
            _, score = train_predict(final_x, y, auxiliary, available, train_kidneys, dev, selected_epoch,
                                     seed, multitask)
            arm_scores[arm].append(score)
            for row, value in zip(kidneys.iloc[dev].itertuples(index=False), score):
                predictions.append({"seed": seed, "group": arm, "patient_alias": row.patient_alias,
                                    "kidney_alias": row.kidney_alias, "side": row.side,
                                    "label": int(row.kidney_APN), "score": float(value)})
    results = []
    target = kidneys.iloc[dev].reset_index(drop=True); labels = y[dev]
    clusters = [np.flatnonzero(target.patient_alias.eq(pid)) for pid in sorted(target.patient_alias.unique())]
    rng = np.random.default_rng(20261112)
    draws = [np.concatenate([clusters[j] for j in rng.integers(0, len(clusters), len(clusters))]) for _ in range(2000)]
    draws = [idx for idx in draws if np.unique(labels[idx]).size == 2]
    for arm, values in arm_scores.items():
        score = np.mean(values, axis=0)
        auc = np.asarray([roc_auc_score(labels[idx], score[idx]) for idx in draws])
        ap = np.asarray([average_precision_score(labels[idx], score[idx]) for idx in draws])
        results.append({"group": f"{arm}_three_seed_mean", "patients": target.patient_alias.nunique(),
                        "kidneys": len(dev), "auroc": roc_auc_score(labels, score),
                        "ap": average_precision_score(labels, score), "auroc_ci_low": np.percentile(auc, 2.5),
                        "auroc_ci_high": np.percentile(auc, 97.5), "ap_ci_low": np.percentile(ap, 2.5),
                        "ap_ci_high": np.percentile(ap, 97.5)})
    pd.DataFrame(results).to_csv(OUT / "development_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "development_predictions.csv", index=False)
    pd.DataFrame(selections).to_csv(OUT / "inner_epoch_selections.csv", index=False)
    (OUT / "protocol.json").write_text(json.dumps({
        "input": "train-fitted DINOv3 paired PCA32, relative geometry PCA16, standardized age/sex",
        "arms": "matched MLP primary DMSA BCE versus primary plus 0.2-weight report auxiliary BCE",
        "auxiliary": "explicit APN wording, abnormal parenchymal density, swelling; missing reports masked",
        "augmentation": "Gaussian feature noise SD .02 in both arms",
        "selection": "160/44 training-patient split per seed; earliest epoch maximizing min(AUROC,AP)",
        "ensemble": list(SEEDS), "test_scored": False}, indent=2), encoding="utf-8")
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
