"""Bounded view-aware and signed contralateral probes with actual kidney labels."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd

from nckuh_dmsa_kidney_baseline_20261102 import (
    AUDIT, FEATURES, OUT as BASE, ROOT, SOURCE, SEEDS, load_reference, verify_lock,
)
from nckuh_only_study_20261102 import digest, weighted_metrics_batch, write_json

OUT = ROOT / "02_experiments/NCKUH_DMSA_contralateral_probe_20261102"
REP = ROOT / "02_experiments/NCKUH_representation_screen_20261102"
USF = ROOT / "02_experiments/NCKUH_ultrasound_probe_20261102"
RAW = ROOT / "02_experiments/NCKUH_raw_intensity_probe_20261102"
CONFIGS = [("handcrafted", mode, head) for mode in ["view", "contra"]
           for head in ["top32_lr", "full_lr", "pca32_svm"]]
CONFIGS += [(family, mode, "full_lr") for family in ["raw_handcrafted", "dense", "dino_class", "dino_patch", "usf_class"]
            for mode in ["view", "contra"]]
CONTROL = "acquisition_only"
REFERENCE_GROUP = "Summary three source fusion"
PATHS = [SOURCE, Path(__file__), Path(__file__).with_name("nckuh_dmsa_kidney_baseline_20261102.py"),
         BASE / "patient_assignments.csv", BASE / "development_predictions.csv", BASE / "development_metrics.csv",
         AUDIT / "image_labels_dmsa_private.csv", AUDIT / "kidney_label_linkage_private.csv", AUDIT / "audit_summary.json",
         FEATURES / "image_manifest.csv", FEATURES / "dense_tight_image.npy", FEATURES / "radiomics_percentile_image.csv",
         FEATURES / "i3_image.csv", RAW / "raw_radiomics_image.csv", REP / "dino_class_image.npy",
         REP / "dino_patch_image.npy", REP / "encoder_provenance.json", USF / "usf_class_image.npy", USF / "encoder_provenance.json"]


def build_kidney_features(values, images, kidneys, mode):
    """Inputs are image descriptors and anatomical keys; never read diagnosis fields."""
    if mode not in ["view", "contra"]:
        raise ValueError(mode)
    values = np.asarray(values)
    if values.ndim != 2 or len(values) != len(images) or not np.isfinite(values).all():
        raise ValueError("Invalid image feature array")
    slots = {}
    for (alias, view), group in images.groupby(["patient_alias", "view"], sort=False):
        if view not in ["LL", "LT", "RL", "RT"]:
            raise ValueError("Invalid anatomical view")
        slots[(alias, view)] = values[group.index.to_numpy()].mean(0)
    width = values.shape[1]
    result, flags = [], []
    for kidney in kidneys.itertuples(index=False):
        side = {"left": "L", "right": "R"}[kidney.side]
        other = "R" if side == "L" else "L"
        own_keys = [(kidney.patient_alias, side + plane) for plane in ["L", "T"]]
        other_keys = [(kidney.patient_alias, other + plane) for plane in ["L", "T"]]
        own = np.stack([slots.get(key, np.full(width, np.nan)) for key in own_keys])
        opposite = np.stack([slots.get(key, np.full(width, np.nan)) for key in other_keys])
        valid_own = np.array([key in slots for key in own_keys], dtype=np.float32)
        valid_other = np.array([key in slots for key in other_keys], dtype=np.float32)
        if not valid_own.any():
            raise ValueError("Kidney has no image")
        if mode == "view":
            row = np.r_[own.ravel(), valid_own]
        else:
            difference = own - opposite
            relative = difference / (np.abs(own) + np.abs(opposite) + 1e-6)
            row = np.r_[own.ravel(), difference.ravel(), relative.ravel(), valid_own, valid_other]
        result.append(row)
        flags.append(np.r_[valid_own, valid_other])
    return np.asarray(result, dtype=np.float32), np.asarray(flags, dtype=np.float32)


def rank_metrics(y, score):
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
    precision, recall, _ = precision_recall_curve(y, score)
    return dict(auroc=float(roc_auc_score(y, score)), ap=float(average_precision_score(y, score)),
                auprc=float(abs(np.trapezoid(precision, recall))))


def prepare():
    verify_lock()
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        raise FileExistsError("Preserve registered protocol")
    patients, kidneys, _ = load_reference()
    for folder in [REP, USF]:
        proof = json.loads((folder / "encoder_provenance.json").read_text(encoding="utf-8"))
        assert proof["frozen"] and proof["images"] == 1027
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=digest(SOURCE), cohort=dict(patients=len(patients), kidneys=len(kidneys), images=820),
        objective="Find kidney-level AUROC and AP both >.8 using the corrected clinical side labels",
        hypotheses=["Canonical longitudinal/transverse slots preserve view information lost by summary pooling",
                    "Signed within-patient same-plane differences can identify the more abnormal side, unlike unsigned patient asymmetry",
                    "Retaining absolute own-kidney values avoids forcing bilateral APN kidneys to appear normal",
                    "Frozen ultrasound/general dense features must be retested under corrected side references before adaptation or fusion"],
        literature=[dict(url="https://pubmed.ncbi.nlm.nih.gov/3513490/", title="Dinkel et al. 1986", supports="Clinical bilateral changes/asymmetry; NOT a validation of these computational features or pixel-based physical size"),
                    dict(url="https://pubmed.ncbi.nlm.nih.gov/17987762/", title="Rathaus and Werner 2007", supports="Focal renal echogenicity differences; NOT permission to infer lesion truth from brightness"),
                    dict(url="https://arxiv.org/abs/2508.10104", title="DINOv3", supports="Frozen dense representation"),
                    dict(url="https://arxiv.org/abs/2510.22990", title="USF-MAE", supports="Ultrasound-domain pretrained representation"),
                    dict(url="https://scikit-learn.org/stable/modules/svm.html", title="scikit-learn official SVM documentation", supports="Use decision_function, probability=False, avoiding hidden internal probability-calibration folds")],
        feature_definition="For each kidney, concatenate mean image descriptors of own longitudinal/transverse views; contra appends signed own-minus-opposite and signed difference/(abs(own)+abs(opposite)+1e-6), plus explicit view-availability flags. No opposite kidney diagnosis input",
        configurations=CONFIGS, disease_candidates=len(CONFIGS), control="Four view-availability flags only, LR C=.1; never eligible for promotion",
        representations="Existing label-independent frozen image caches aligned to the exact canonical1027 image order; supervised fitting uses only the820 images in414 kidney bags",
        preprocessing="Median imputation, standardization and variance filtering fit on training kidneys only. LR top32 uses ANOVA32 and C=.1; full LR C=.01. SVM uses whitened PCA32 then RBF C=1 gamma=1/32, class_weight=balanced, probability=False",
        score_boundary="LR outputs probabilities; SVM outputs raw decision margins. Compare AUROC/AP/trapezoidal AUPRC only; no Brier claim or probability voting with SVM margins",
        sampling="Same corrected-reference ten random patient draws,126/42/42, R/L/B/N stratification; no k-fold. Both sides/all images stay together",
        selection="Mean development (AUROC+AP)/2, then fixed group-name ordering; report every candidate, never score test during this screen",
        gate=dict(comparator=REFERENCE_GROUP, min_delta_auroc=.03, min_delta_ap=.02, min_joint_wins=6),
        confirmation="Even a promotion-gate pass is not target completion. Do not inspect exploratory test before a separate single-candidate lock; require both development metrics>.8 before target confirmation",
        exclusions="No CYCH classifier weights or images; no mixed-site segmentation finetune; no clinical/source identifiers, opposite diagnosis, DMSA_RF/LF/YN or previous labels as predictors",
        caveat="All patients historically explored. One malformed image date/precise DMSA timing remain unresolved. Conditional within-cohort CIs do not establish external generalization")
    write_json(OUT / "protocol.json", protocol)
    write_json(OUT / "run_lock.json", dict(protocol_sha256=digest(OUT / "protocol.json"),
        inputs={p.relative_to(ROOT).as_posix(): digest(p) for p in PATHS}))
    print(f"Registered {len(CONFIGS)} candidates plus one acquisition control before corrected-label scoring", flush=True)


def require_lock():
    lock = json.loads((OUT / "run_lock.json").read_text(encoding="utf-8"))
    assert digest(OUT / "protocol.json") == lock["protocol_sha256"]
    for name, value in lock["inputs"].items():
        assert digest(ROOT / name) == value, name


def build():
    require_lock()
    _, kidneys, images = load_reference()
    legacy = pd.read_csv(FEATURES / "image_manifest.csv", dtype={"patient_id": str})
    assert legacy.image_id.tolist() == images.image_id.tolist() and legacy.filename.tolist() == images.filename.tolist()
    rad = pd.read_csv(FEATURES / "radiomics_percentile_image.csv").set_index("image_id").reindex(images.image_id)
    rad = rad[[c for c in rad if not c.endswith(("_Energy", "_TotalEnergy"))]]
    direction = pd.read_csv(FEATURES / "i3_image.csv").set_index("image_id").reindex(images.image_id)
    raw = pd.read_csv(RAW / "raw_radiomics_image.csv").set_index("image_id").reindex(images.image_id)
    assert raw.columns.tolist() == rad.columns.tolist()
    arrays = dict(handcrafted=np.c_[rad.to_numpy(), direction.to_numpy()],
                  raw_handcrafted=np.c_[raw.to_numpy(), direction.to_numpy()],
                  dense=np.load(FEATURES / "dense_tight_image.npy"),
                  dino_class=np.load(REP / "dino_class_image.npy"),
                  dino_patch=np.load(REP / "dino_patch_image.npy"),
                  usf_class=np.load(USF / "usf_class_image.npy"))
    audit = []
    for family, values in arrays.items():
        for mode in ["view", "contra"]:
            result, flags = build_kidney_features(values, images, kidneys, mode)
            np.save(OUT / f"{family}_{mode}.npy", result)
            audit.append(dict(family=family, mode=mode, shape=list(result.shape),
                              missing_values=int(np.isnan(result).sum()), source_image_shape=list(values.shape),
                              sha256=digest(OUT / f"{family}_{mode}.npy")))
    np.save(OUT / f"{CONTROL}.npy", flags)
    write_json(OUT / "feature_audit.json", dict(arrays=audit, disease_labels_used_for_features=False,
        physical_size_inferred=False, omitted_opposite_views_preserved_as_missing=True,
        source_columns_handcrafted=rad.columns.tolist()+direction.columns.tolist(),
        kidney_order=kidneys.kidney_alias.tolist(), image_order_sha256=digest(AUDIT / "image_labels_dmsa_private.csv")))
    print("Built anatomical signed comparisons without using either kidney's diagnosis", flush=True)


def classifier(head, seed):
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    stages = [SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), VarianceThreshold()]
    if head == "top32_lr":
        stages.append(SelectKBest(f_classif, k=32))
    elif head == "pca32_svm":
        stages.append(PCA(32, whiten=True, random_state=seed))
    if head == "pca32_svm":
        stages.append(SVC(C=1, gamma=1/32, kernel="rbf", class_weight="balanced", probability=False, random_state=seed))
    else:
        stages.append(LogisticRegression(C=.01 if head == "full_lr" else .1, solver="liblinear",
                                        class_weight="balanced", max_iter=3000, random_state=seed))
    return make_pipeline(*stages)


def screen():
    from threadpoolctl import threadpool_limits
    require_lock()
    if (OUT / "development_predictions.csv").exists():
        raise FileExistsError("Preserve scored candidates")
    _, kidneys, _ = load_reference()
    y = kidneys.kidney_APN.to_numpy(int)
    assignments = pd.read_csv(BASE / "patient_assignments.csv")
    metrics, predictions, fitted = [], [], []
    (OUT / "models").mkdir(exist_ok=True)
    configs = CONFIGS + [(CONTROL, "", "control_lr")]
    with threadpool_limits(limits=4), warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        for family, mode, head in configs:
            feature = family + ("_"+mode if mode else "")
            group = feature + "_" + head
            values = np.load(OUT / (feature + ".npy"))
            for seed in SEEDS:
                mapping = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
                roles = kidneys.patient_alias.map(mapping)
                train, dev = np.flatnonzero(roles.eq("train")), np.flatnonzero(roles.eq("development"))
                assert not set(kidneys.patient_alias.iloc[train]) & set(kidneys.patient_alias.iloc[dev])
                model = classifier(head, seed)
                model.fit(values[train], y[train])
                score = model.decision_function(values[dev]) if head == "pca32_svm" else model.predict_proba(values[dev])[:, 1]
                assert np.isfinite(score).all()
                metrics.append(dict(group=group, seed=seed, role="development", patients=42, kidneys=len(dev),
                                    score_type="SVM margin" if head == "pca32_svm" else "LR probability",
                                    **rank_metrics(y[dev], score)))
                for i, s in zip(dev, score):
                    predictions.append(dict(group=group, seed=seed, patient_alias=kidneys.patient_alias.iloc[i],
                        kidney_alias=kidneys.kidney_alias.iloc[i], DMSA_APN=kidneys.DMSA_APN.iloc[i],
                        side=kidneys.side.iloc[i], label=int(y[i]), score=float(s)))
                joblib.dump(model, OUT / "models" / f"{seed}_{group}.joblib", compress=3)
                fitted.append(dict(group=group, seed=seed, training_kidneys=len(train), training_patients=126))
            print(f"Development only completed {group}", flush=True)
    frame = pd.DataFrame(metrics)
    frame.to_csv(OUT / "development_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "development_predictions.csv", index=False)
    pd.DataFrame(fitted).to_csv(OUT / "fit_audit.csv", index=False)
    summary = frame.groupby("group")[["auroc", "ap", "auprc"]].mean()
    summary["mean_score"] = summary[["auroc", "ap"]].mean(axis=1)
    summary = summary.sort_values(["mean_score", "group"], ascending=[False, True])
    reference = pd.read_csv(BASE / "development_metrics.csv")
    reference = reference[reference.group.eq(REFERENCE_GROUP)].set_index("seed")[["auroc", "ap"]]
    decisions = []
    for group in summary.index:
        delta = frame[frame.group.eq(group)].set_index("seed")[["auroc", "ap"]] - reference
        wins = int(delta.gt(0).all(axis=1).sum())
        eligible = not group.startswith(CONTROL) and delta.auroc.mean() >= .03 and delta.ap.mean() >= .02 and wins >= 6
        decisions.append(dict(group=group, delta_auroc=float(delta.auroc.mean()), delta_ap=float(delta.ap.mean()),
                              joint_wins=wins, eligible=bool(eligible)))
    decision = pd.DataFrame(decisions).set_index("group")
    summary = summary.join(decision)
    summary.to_csv(OUT / "development_summary.csv")
    eligible = summary.index[summary.eligible].tolist()
    disease = summary[~summary.index.str.startswith(CONTROL)]
    write_json(OUT / "candidate_decision.json", dict(
        best_by_development=disease.index[0], selected_for_promotion=eligible[0] if eligible else None,
        eligible=eligible, comparator=REFERENCE_GROUP, test_scoring_performed=False,
        candidates=len(CONFIGS), controls=1, target_met_on_leading_development=bool((disease.iloc[0][["auroc", "ap"]] > .8).all())))
    print(summary.to_string(), flush=True)


def bootstrap():
    require_lock()
    frame = pd.read_csv(OUT / "development_predictions.csv")
    points = pd.read_csv(OUT / "development_metrics.csv")
    summary = pd.read_csv(OUT / "development_summary.csv")
    groups = summary.group.tolist()
    aliases = sorted(frame.patient_alias.unique())
    codes = frame.groupby("patient_alias").DMSA_APN.first().reindex(aliases).to_numpy()
    strata = [np.flatnonzero(codes == c) for c in ["R", "L", "B", "N"]]
    base_predictions = pd.read_csv(BASE / "development_predictions.csv")
    base_predictions = base_predictions[base_predictions.group.eq(REFERENCE_GROUP)].copy()
    base_predictions["group"] = REFERENCE_GROUP
    all_predictions = pd.concat([frame, base_predictions], ignore_index=True)
    groups.append(REFERENCE_GROUP)
    matrices = []
    for seed in SEEDS:
        draw = all_predictions[all_predictions.seed.eq(seed)]
        pivot = draw.pivot(index="kidney_alias", columns="group", values="score").reindex(columns=groups)
        assert pivot.notna().all().all()
        key = draw.groupby("kidney_alias").first().reindex(pivot.index)
        clusters = np.array([aliases.index(a) for a in key.patient_alias])
        matrices.append((clusters, key.label.to_numpy(int), pivot.to_numpy()))
    rng = np.random.default_rng(20261102)
    batches, total = [], 0
    while total < 10000:
        size = min(1000, 10000-total)
        sample = np.concatenate([rng.choice(s, (size, len(s)), replace=True) for s in strata], axis=1)
        counts = np.zeros((size, len(aliases)))
        np.add.at(counts, (np.arange(size)[:, None], sample), 1)
        values = np.zeros((size, len(groups), 3))
        keep = np.ones(size, dtype=bool)
        for clusters, y, scores in matrices:
            weights = counts[:, clusters]
            keep &= (weights[:, y==0].sum(1)>0) & (weights[:, y==1].sum(1)>0)
            for j in range(len(groups)):
                values[:, j] += weighted_metrics_batch(y, scores[:, j], weights)[:, :3]
        batches.append(values[keep] / len(SEEDS))
        total += int(keep.sum())
    samples = np.concatenate(batches)[:10000]
    np.save(OUT / "development_cluster_bootstrap.npy", samples)
    rows, paired = [], []
    base_points = pd.read_csv(BASE / "development_metrics.csv")
    base_points = base_points[base_points.group.eq(REFERENCE_GROUP)]
    for j, group in enumerate(groups[:-1]):
        for k, metric in enumerate(["auroc", "ap", "auprc"]):
            point = float(points[points.group.eq(group)][metric].mean())
            low, high = np.quantile(samples[:, j, k], [.025, .975])
            rows.append(dict(group=group, metric=metric, mean=point, low=float(low), high=float(high),
                             replicates=10000, distinct_patients=len(aliases)))
            delta = samples[:, j, k] - samples[:, -1, k]
            lo, hi = np.quantile(delta, [.025, .975])
            paired.append(dict(group=group, metric=metric, comparator=REFERENCE_GROUP,
                difference=point-float(base_points[metric].mean()), low=float(lo), high=float(hi)))
    pd.DataFrame(rows).to_csv(OUT / "development_confidence_intervals.csv", index=False)
    pd.DataFrame(paired).to_csv(OUT / "development_paired_intervals.csv", index=False)
    write_json(OUT / "completion_audit.json", dict(completed_utc=datetime.now(timezone.utc).isoformat(),
        test_scoring_performed=False, source_unchanged=True, label_unit="kidney", split_and_bootstrap_cluster="patient",
        fitted_candidates=len(CONFIGS), diagnostic_controls=1, unique_development_patients=len(aliases),
        prediction_sha256=digest(OUT / "development_predictions.csv"),
        caveat="Conditional fixed-prediction within-cohort intervals; not adjusted for selection/historical exploration"))
    print("Finished all candidate and paired patient-cluster intervals; no exploratory-test scoring", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "build", "screen", "bootstrap"])
    globals()[parser.parse_args().stage]()
