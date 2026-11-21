"""Paper 2, step 7: the two published method chains, refitted inside each probe group.

NCKUH paper-1 chain (original code and fixed weights, refitted on the probe group's training patients):
  fixed chain = .75*(.75*(.75*tile_two_heads + .25*four-query local MIL) + .25*kNN) + .25*aligned texture
  final       = .75*(.75*fixed chain + .25*tile+USFM four-state hierarchy) + .25*URI-CADS embedding head
CYCH paper-1 chain (kidney bag instead of patient bag, because the NCKUH reference is side-specific):
  reliable-negative GCN-MIL on frozen RadImageNet DenseNet-121 (train-only scaler + PCA32, 2 GCN layers
  32-64-32, gated attention 16, dropout .25, 300 AdamW epochs), PyRadiomics probe and directional I3 probe
  (ANOVA<=32 + balanced LR C=0.1); final = equal mean of the three probabilities.
Combined = equal mean of the two finals. Nothing is tuned on held-out folds; the shared five patient
folds are those of the other paper-2 steps. Exploratory-test role is never scored.
"""

import json
import sys
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from torch import nn

import nckuh_dmsa_multiquery_mil_20261103 as old_mil
from p2_05_nested_frozen import shared_folds
from p2_common import CACHE, EXP, FOLD_IDS, P2, P2_OUT, SEED, cluster_interval, metric, pool_mask
from p2_features import LEGACY, routed_images
from probe_nckuh_383_legacy_fixed_components_20261112 import fit_components
from probe_nckuh_383_legacy_hierarchy_20261112 import score_head

OUT = P2_OUT / "paper_methods"
I3_KEYS = ("firstorder_Entropy", "firstorder_Variance", "glrlm_RunLengthNonUniformityNormalized",
           "glrlm_GrayLevelNonUniformityNormalized", "glrlm_RunEntropy")


# ---------------------------------------------------------------- NCKUH paper-1 chain
def local_mil(images, kidneys, patients, train_patients, dev, seed):
    offsets = np.load(LEGACY / "patch_offsets.npy")
    raw = np.load(LEGACY / "radimagenet_local_float16.npy", mmap_mode="r")
    rows = images.cache_row.to_numpy()
    values = np.asarray(raw[np.concatenate([np.arange(offsets[i], offsets[i + 1]) for i in rows])], dtype=np.float32)
    values /= np.linalg.norm(values, axis=1, keepdims=True).clip(1e-8)
    image_index = np.repeat(np.arange(len(images)), np.diff(offsets)[rows])
    patch_patients = images.patient_alias.to_numpy()[image_index]
    patch_kidneys = images.kidney_alias.to_numpy()[image_index]
    patch_views = np.asarray([0 if str(v).endswith("L") else 1 for v in images.view.to_numpy()[image_index]], np.int64)
    groups = [np.flatnonzero(patch_kidneys == alias) for alias in kidneys.kidney_alias]
    labels_np = kidneys.kidney_APN.to_numpy(int)
    labels = torch.from_numpy(labels_np.astype(np.float32)).cuda()
    people = patients.iloc[train_patients].reset_index(drop=True)
    fit_local, select_local = train_test_split(np.arange(len(people)), train_size=162 / 204,
                                               random_state=seed + 6000, stratify=people.DMSA_APN)
    fit_aliases, select_aliases = set(people.patient_alias.iloc[fit_local]), set(people.patient_alias.iloc[select_local])
    fit = np.flatnonzero(kidneys.patient_alias.isin(fit_aliases))
    selection = np.flatnonzero(kidneys.patient_alias.isin(select_aliases))
    bags, mask, views, _ = old_mil.prepare_bags(values, patch_patients, patch_views, groups, fit_aliases, seed)
    _, history = old_mil.train_model("cross_attention_q4", seed, bags, mask, views, labels, fit, old_mil.EPOCHS, selection)
    options = [(-(old_mil.metric(labels_np[selection], s)["auroc"] + old_mil.metric(labels_np[selection], s)["ap"]) / 2, e)
               for e, s in history.items()]
    _, epoch = min(options)
    train_aliases = set(people.patient_alias)
    train = np.flatnonzero(kidneys.patient_alias.isin(train_aliases))
    bags, mask, views, _ = old_mil.prepare_bags(values, patch_patients, patch_views, groups, train_aliases, seed + 100)
    model, _ = old_mil.train_model("cross_attention_q4", seed + 10000, bags, mask, views, labels, train, (epoch,), None)
    model.eval()
    with torch.inference_mode():
        return torch.sigmoid(model(bags[dev], mask[dev], views[dev])[0]).cpu().numpy()


def nckuh_chain(patients, kidneys, images, arrays, texture, train_patients, dev, seed):
    parts = fit_components(patients, kidneys, images, arrays["tile"], arrays["rad_color"], arrays["rad_mirror"],
                           texture, train_patients, dev, seed)
    mil = local_mil(images, kidneys, patients, train_patients, dev, seed)
    fixed = .75 * (.75 * (.75 * parts["tile_two_heads"] + .25 * mil) + .25 * parts["knn"]) + .25 * parts["aligned"]
    hierarchy, _ = score_head(patients, kidneys, images, np.c_[arrays["tile"], arrays["usfm"]], train_patients, dev, seed)
    uri, _ = score_head(patients, kidneys, images, arrays["uri_folds"], train_patients, dev, seed)
    return {"nckuh1_fixed_chain": fixed, "nckuh1_hierarchy": hierarchy, "nckuh1_uricads": uri,
            "nckuh1_final": .75 * (.75 * fixed + .25 * hierarchy) + .25 * uri}


# ---------------------------------------------------------------- CYCH paper-1 chain
def normalized_rbf_adjacency(values, valid):  # copied from run_graph_attention_mil_development.py
    difference = values.unsqueeze(2) - values.unsqueeze(1)
    distance2 = torch.sum(difference * difference, dim=-1)
    pair_valid = valid.unsqueeze(2) & valid.unsqueeze(1)
    eye = torch.eye(values.shape[1], dtype=torch.bool).unsqueeze(0)
    positive = torch.where(pair_valid & ~eye & (distance2 > 0), distance2, torch.nan)
    bandwidth = torch.nanmedian(positive.flatten(1), dim=1).values
    bandwidth = torch.where(torch.isfinite(bandwidth), bandwidth, torch.ones_like(bandwidth)).clamp_min(1e-6)
    adjacency = torch.exp(-distance2 / bandwidth[:, None, None]) * pair_valid
    adjacency = torch.where(eye & pair_valid, torch.ones_like(adjacency), adjacency)
    degree = adjacency.sum(dim=-1).clamp_min(1e-8)
    return adjacency / torch.sqrt(degree.unsqueeze(-1) * degree.unsqueeze(-2)) * pair_valid


class ReliableNegativeGraphMIL(nn.Module):
    def __init__(self):
        super().__init__()
        self.gcn1, self.gcn2 = nn.Linear(32, 64, bias=False), nn.Linear(64, 32, bias=False)
        self.dropout = nn.Dropout(0.25)
        self.attention_v, self.attention_u = nn.Linear(32, 16), nn.Linear(32, 16)
        self.attention_w = nn.Linear(16, 1, bias=False)
        self.bag_classifier, self.instance_classifier = nn.Linear(32, 1), nn.Linear(32, 1)

    def forward(self, values, valid):
        adjacency = normalized_rbf_adjacency(values, valid)
        hidden = self.dropout(torch.relu(self.gcn1(torch.bmm(adjacency, values))))
        hidden = torch.relu(self.gcn2(torch.bmm(adjacency, hidden)))
        raw = self.attention_w(torch.tanh(self.attention_v(hidden)) * torch.sigmoid(self.attention_u(hidden))).squeeze(-1)
        attention = torch.softmax(raw.masked_fill(~valid, -1e9), dim=1)
        pooled = torch.sum(attention.unsqueeze(-1) * hidden, dim=1)
        return self.bag_classifier(pooled).squeeze(-1), self.instance_classifier(hidden).squeeze(-1)


def gcn_mil(dense, images, kidneys, train, dev, seed):
    train_aliases = set(kidneys.patient_alias.iloc[train])
    image_train = np.flatnonzero(images.patient_alias.isin(train_aliases))
    reducer = make_pipeline(StandardScaler(), PCA(32, random_state=seed)).fit(dense[image_train])
    nodes = reducer.transform(dense).astype(np.float32)
    members = [np.flatnonzero(images.kidney_alias.to_numpy() == alias) for alias in kidneys.kidney_alias]
    width = max(map(len, members))
    bags, masks = np.zeros((len(members), width, 32), np.float32), np.zeros((len(members), width), bool)
    for i, m in enumerate(members):
        bags[i, :len(m)], masks[i, :len(m)] = nodes[m], True
    bags, masks = torch.from_numpy(bags), torch.from_numpy(masks)
    torch.manual_seed(seed)
    model = ReliableNegativeGraphMIL()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    index = torch.as_tensor(train, dtype=torch.long)
    target = torch.as_tensor(kidneys.kidney_APN.to_numpy(np.float32)[train])
    weights = torch.where(target > 0, len(target) / (2 * target.sum()), len(target) / (2 * (len(target) - target.sum())))
    for _ in range(300):
        model.train()
        optimizer.zero_grad()
        logits, instance_logits = model(bags[index], masks[index])
        loss = (nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none") * weights).mean()
        reliable = masks[index] & (target[:, None] == 0)
        if bool(reliable.any()):
            loss = loss + nn.functional.binary_cross_entropy_with_logits(
                instance_logits[reliable], torch.zeros_like(instance_logits[reliable]))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    model.eval()
    with torch.inference_mode():
        return torch.sigmoid(model(bags[dev], masks[dev])[0]).numpy()


def probe(features, labels, train, dev):
    model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), VarianceThreshold(),
                          StandardScaler(), SelectKBest(f_classif, k=min(32, features.shape[1])),
                          LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000))
    return model.fit(features[train], labels[train]).predict_proba(features[dev])[:, 1]


def cych_chain(kidneys, images, arrays, radiomics, train, dev, seed):
    labels = kidneys.kidney_APN.to_numpy(int)
    members = [np.flatnonzero(images.kidney_alias.to_numpy() == alias) for alias in kidneys.kidney_alias]
    values = radiomics.to_numpy(float)
    rad_mean = np.stack([np.nanmean(values[m], 0) for m in members])
    i3_cols = [i for i, c in enumerate(radiomics.columns) if c.endswith(I3_KEYS)]
    assert len(i3_cols) == 25
    i3 = np.stack([np.r_[np.nanmean(values[m][:, i3_cols], 0), np.nanstd(values[m][:, i3_cols], 0)] for m in members])
    parts = {"cych_gcn_mil": gcn_mil(arrays["rad_dense"], images, kidneys, train, dev, seed),
             "cych_pyradiomics": probe(rad_mean, labels, train, dev),
             "cych_i3": probe(i3, labels, train, dev)}
    parts["cych_final"] = (parts["cych_gcn_mil"] + parts["cych_pyradiomics"] + parts["cych_i3"]) / 3
    return parts


def main():
    only = sys.argv[1].split(",") if len(sys.argv) > 1 else ("convex", "linear")
    OUT.mkdir(parents=True, exist_ok=True)
    all_patients, _, all_images = routed_images()
    pool = all_images[pool_mask(all_images)]
    fold_of = shared_folds(all_patients, pool)
    uri = np.load(CACHE / "uricads_fold_embeddings.npy", mmap_mode="r")
    texture_all = pd.read_csv(LEGACY / "directional_features.csv").set_index("image_id")
    raw = pd.read_csv(CACHE / "radiomics.csv").set_index("image_id")
    raw = raw[[c for c in raw if not c.endswith(("_Energy", "_TotalEnergy"))]].apply(pd.to_numeric, errors="coerce")
    raw = raw.replace([np.inf, -np.inf], np.nan)
    for probe_name in only:
        images = pool[pool.route.eq(probe_name)].reset_index(drop=True)
        patients = all_patients[all_patients.patient_alias.isin(images.patient_alias)].reset_index(drop=True)
        kidneys = images[["kidney_alias", "patient_alias", "side", "kidney_APN"]].drop_duplicates().reset_index(drop=True)
        rows = images.cache_row.to_numpy()
        arrays = {"tile": np.load(LEGACY / "tile_mean.npy")[rows], "usfm": np.load(LEGACY / "usfm_gap.npy")[rows],
                  "rad_color": np.load(LEGACY / "rad_color.npy")[rows],
                  "rad_mirror": np.load(LEGACY / "rad_color_horizontal_reflection.npy")[rows],
                  "rad_dense": np.load(CACHE / "radimagenet_dense_tight.npy")[rows].astype(np.float64),
                  "uri_folds": np.asarray(uri[rows]).reshape(len(rows), -1)}
        texture = texture_all.loc[images.image_id].reset_index(drop=True)
        radiomics = raw.loc[images.image_id].reset_index(drop=True)
        patient_fold = patients.patient_alias.map(fold_of).to_numpy()
        kidney_fold = kidneys.patient_alias.map(fold_of).to_numpy()
        scores = {}
        for k in FOLD_IDS:
            train_patients = np.flatnonzero((patient_fold != k) & (patient_fold >= 0))
            train, dev = np.flatnonzero((kidney_fold != k) & (kidney_fold >= 0)), np.flatnonzero(kidney_fold == k)
            parts = nckuh_chain(patients, kidneys, images, arrays, texture, train_patients, dev, SEED + k)
            parts |= cych_chain(kidneys, images, arrays, radiomics, train, dev, SEED + k)
            parts["combined_equal_mean"] = (parts["nckuh1_final"] + parts["cych_final"]) / 2
            for name, value in parts.items():
                scores.setdefault(name, np.full(len(kidneys), np.nan))[dev] = value
            y = kidneys.kidney_APN.to_numpy(int)[dev]
            print(probe_name, "fold", k, {n: f"{metric(y, v)['auroc']:.3f}/{metric(y, v)['ap']:.3f}"
                                          for n, v in parts.items() if n.endswith(("final", "mean"))}, flush=True)
        keep = np.isin(kidney_fold, FOLD_IDS)
        kidneys, kidney_fold = kidneys[keep].reset_index(drop=True), kidney_fold[keep]
        scores = {name: value[keep] for name, value in scores.items()}
        y = kidneys.kidney_APN.to_numpy(int)
        report = {"patients": int(kidneys.patient_alias.nunique()), "kidneys": len(kidneys), "prevalence": float(y.mean()),
                  "models": {n: metric(y, v) | cluster_interval(kidneys.patient_alias, y, v) for n, v in scores.items()},
                  "test_APN_scored": False}
        kidneys.assign(route=probe_name, fold=kidney_fold, **scores).rename(columns={"kidney_APN": "label"}).to_csv(
            OUT / f"{probe_name}_kidney_oof_scores.csv", index=False)
        (OUT / f"{probe_name}_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        for name, value in report["models"].items():
            print(f"{probe_name:6s} {name:22s} AUROC {value['auroc']:.3f} ({value['auroc_95pct'][0]:.3f}-{value['auroc_95pct'][1]:.3f})"
                  f"  AP {value['ap']:.3f} ({value['ap_95pct'][0]:.3f}-{value['ap_95pct'][1]:.3f})", flush=True)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    with threadpool_limits(limits=6):
        main()
