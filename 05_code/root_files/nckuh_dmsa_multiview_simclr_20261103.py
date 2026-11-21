"""Same-kidney longitudinal/transverse contrastive adaptation for NCKUH panels."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import nckuh_dmsa_domain_simclr_20261103 as D
import nckuh_dmsa_fullcohort_fullframe_20261103 as F
from nckuh_only_study_20261102 import metric


ROOT, BASE = F.ROOT, F.OUT
OUT = ROOT / "02_experiments/NCKUH_DMSA_multiview_simclr_20261103"
SEEDS, EPOCHS, BATCH = F.SEEDS, 20, 64
REFERENCE_GROUP = "fullcohort_rad_color_image_plus_patient_direct"
GROUPS = ("multiview_simclr_pca_lr", "multiview_simclr_quarter_plus_instance_reference")


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def panel_metadata() -> pd.DataFrame:
    """Reconstruct cache order and same-kidney view group without APN labels."""
    _, _, labelled = D.load_reference()
    aliases = labelled[["patient_id", "patient_alias"]].drop_duplicates().copy()
    aliases["normalized_patient_id"] = aliases.patient_id.astype(str).map(F.normalize_patient_id)
    alias_by_source = aliases.set_index("normalized_patient_id").patient_alias.to_dict()
    exact, _ = F.panel_index()
    rows = []
    for (patient, date, view), paths in sorted(exact.items()):
        side = view[0]
        for _ in paths:
            rows.append({"known_patient_alias": alias_by_source.get(patient, ""),
                         "kidney_view_group": f"{patient}|{date}|{side}"})
    values, _ = D.build_cache()
    frame = pd.DataFrame(rows)
    assert len(frame) == len(values)
    return frame


def positive_indices(metadata: pd.DataFrame) -> np.ndarray:
    """Select another view of the same kidney where available; else self-view."""
    result = np.arange(len(metadata), dtype=np.int64)
    for _, rows in metadata.groupby("kidney_view_group", sort=False):
        indices = rows.index.to_numpy(np.int64)
        if len(indices) > 1:
            result[indices] = np.roll(indices, -1)
    return result


def pretrain(values: np.ndarray, available: np.ndarray, partner: np.ndarray, seed: int) -> D.SimCLR:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = D.SimCLR(D.radimagenet_encoder()).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    model.train()
    for _ in range(EPOCHS):
        order = rng.permutation(available)
        for start in range(0, len(order), BATCH):
            index = order[start:start + BATCH]
            first = torch.from_numpy(np.asarray(values[index]).copy()).cuda().float() / 127.5 - 1
            second = torch.from_numpy(np.asarray(values[partner[index]]).copy()).cuda().float() / 127.5 - 1
            optimizer.zero_grad(set_to_none=True)
            loss = D.nt_xent(model(D.color_augment(first)), model(D.color_augment(second)))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
    return model.eval()


def prepare() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        raise FileExistsError("Preserve registered multiview-SimCLR protocol")
    inputs = [Path(__file__), D.OUT / "panel_cache_audit.json", D.OUT / "run_lock.json", BASE / "patient_assignments.csv", BASE / "instance_development_predictions.csv"]
    write_json(OUT / "protocol.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hypothesis": "A longitudinal and transverse view of the same kidney preserve shared anatomy/pathology while differing in probe plane; using them as contrastive positives should be more useful than two augmentations of one frame.",
        "corpus": "Reuse the 1,692 same-center panel cache. For each draw, admit only outer-training patients and outcome-unavailable patients, excluding all labelled development and exploratory-test patients before contrastive pretraining.",
        "positive_pair": "Same normalized source patient, date and anatomical side, with a different LL/LT or RL/RT view when available; otherwise the panel's own augmented view.",
        "encoder": "RadImageNet DenseNet-121 plus fixed 20-epoch multiview NT-Xent adaptation. The downstream training-only PCA/LR head and 25% fixed fusion are unchanged from the prior domain-SimCLR branch.",
        "evaluation": "Three fixed patient development roles only; no exploratory-test scoring, no development-selected epoch count or fusion weight.",
        "target": "Development AUROC>0.8 and AP>0.8 followed by locked confirmation.",
        "inputs": {str(path): D.digest(path) for path in inputs},
    })
    write_json(OUT / "run_lock.json", {"protocol_sha256": D.digest(OUT / "protocol.json"), "source_sha256": D.digest(Path(__file__))})


def require_lock() -> None:
    lock = json.loads((OUT / "run_lock.json").read_text(encoding="utf-8"))
    assert D.digest(OUT / "protocol.json") == lock["protocol_sha256"]
    assert D.digest(Path(__file__)) == lock["source_sha256"]


def run() -> None:
    from sklearn.model_selection import train_test_split
    require_lock()
    if (OUT / "development_predictions.csv").exists():
        raise FileExistsError("Preserve completed multiview-SimCLR predictions")
    torch.set_num_threads(4)
    patients, kidneys, images = D.load_reference()
    panels, panel_meta = D.build_cache()
    metadata = panel_metadata()
    partner = positive_indices(metadata)
    assignments = pd.read_csv(BASE / "patient_assignments.csv")
    reference = pd.read_csv(BASE / "instance_development_predictions.csv")
    reference = reference[reference.group.eq(REFERENCE_GROUP)].set_index(["seed", "kidney_alias"]).score
    labels = kidneys.kidney_APN.to_numpy(int)
    metrics, predictions, selections = [], [], []
    (OUT / "models").mkdir(exist_ok=True)
    for seed in SEEDS:
        mapping = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
        train_alias, excluded_alias = set(mapping[mapping.eq("train")].index), set(mapping[~mapping.eq("train")].index)
        allowed = metadata.known_patient_alias.eq("") | metadata.known_patient_alias.isin(train_alias)
        assert not metadata.loc[allowed, "known_patient_alias"].isin(excluded_alias).any()
        model = pretrain(panels, np.flatnonzero(allowed), partner, seed)
        kidney_features = D.kidney_means(images, kidneys, D.encode_images(model, images))
        roles = kidneys.patient_alias.map(mapping)
        train, development = np.flatnonzero(roles.eq("train")), np.flatnonzero(roles.eq("development"))
        train_patients = patients[patients.patient_alias.isin(train_alias)].reset_index(drop=True)
        fit_patient, inner_patient = train_test_split(np.arange(len(train_patients)), train_size=100, random_state=seed + 81103, stratify=train_patients.DMSA_APN)
        fit_alias, inner_alias = set(train_patients.patient_alias.iloc[fit_patient]), set(train_patients.patient_alias.iloc[inner_patient])
        fit, inner = np.flatnonzero(kidneys.patient_alias.isin(fit_alias)), np.flatnonzero(kidneys.patient_alias.isin(inner_alias))
        choices = []
        for c in D.INNER_C:
            for k in D.INNER_K:
                head = D.make_head(k, c, kidney_features.shape[1]).fit(kidney_features[fit], labels[fit])
                result = metric(labels[inner], head.predict_proba(kidney_features[inner])[:, 1])
                choices.append((-(result["auroc"] + result["ap"]) / 2, k, c, result))
        _, k, c, inner_result = min(choices, key=lambda value: (value[0], value[1], value[2]))
        head = D.make_head(k, c, kidney_features.shape[1]).fit(kidney_features[train], labels[train])
        score = head.predict_proba(kidney_features[development])[:, 1]
        ref = np.asarray([reference.loc[(seed, alias)] for alias in kidneys.kidney_alias.iloc[development]])
        scores = {GROUPS[0]: score, GROUPS[1]: .25 * score + .75 * ref}
        torch.save({"encoder": model.encoder.state_dict(), "projector": model.projector.state_dict(), "seed": seed}, OUT / "models" / f"{seed}.pth")
        selections.append({"seed": seed, "pretrain_panels": int(allowed.sum()), "excluded_panels": int((~allowed).sum()), "cross_view_positive_panels": int((partner != np.arange(len(partner))).sum()), "chosen_c": c, "chosen_pca": k, "inner_auroc": inner_result["auroc"], "inner_ap": inner_result["ap"]})
        for group, candidate in scores.items():
            metrics.append({"seed": seed, "role": "development", "group": group, "patients": 42, "kidneys": len(development), **metric(labels[development], candidate)})
            predictions.extend({"seed": seed, "group": group, "patient_alias": kidneys.patient_alias.iloc[index], "kidney_alias": kidneys.kidney_alias.iloc[index], "DMSA_APN": kidneys.DMSA_APN.iloc[index], "side": kidneys.side.iloc[index], "label": int(labels[index]), "score": float(candidate[local])} for local, index in enumerate(development))
        print(f"Completed multiview SimCLR draw {seed}: {int(allowed.sum())} panels", flush=True)
        del model; torch.cuda.empty_cache()
    pd.DataFrame(metrics).to_csv(OUT / "development_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "development_predictions.csv", index=False)
    pd.DataFrame(selections).to_csv(OUT / "inner_selection.csv", index=False)
    summary = pd.DataFrame(metrics).groupby("group")[["auroc", "ap", "brier"]].agg(["mean", "std"])
    summary.columns = ["_".join(x) for x in summary.columns]
    summary.to_csv(OUT / "development_summary.csv")
    write_json(OUT / "development_decision.json", {"groups_above_both_development_targets": summary[(summary.auroc_mean > .8) & (summary.ap_mean > .8)].index.tolist(), "exploratory_test_scored": False})
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run"))
    globals()[parser.parse_args().stage]()
