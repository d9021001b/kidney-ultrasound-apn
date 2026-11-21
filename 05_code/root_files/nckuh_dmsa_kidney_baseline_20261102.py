"""Fixed-method, patient-grouped kidney exploration after clinical label repair."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from nckuh_only_study_20261102 import (
    GROUPS, SEEDS, digest, metric, train_neural, weighted_metrics_batch, write_json,
)

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "02_experiments/NCKUH_DMSA_label_audit_20261102"
OUT = ROOT / "02_experiments/NCKUH_DMSA_kidney_baseline_20261102"
FEATURES = ROOT / "02_experiments/CYCH_honest_reevaluation_20261003/feat_NCKUH"
SOURCE = ROOT / "2004-2019-NCKUH-亞大-20210723.xlsx"
INPUTS = [AUDIT / n for n in ["audit_summary.json", "patient_label_linkage_private.csv",
          "kidney_label_linkage_private.csv", "image_labels_dmsa_private.csv"]]
INPUTS += [FEATURES / n for n in ["image_manifest.csv", "dense_tight_image.npy",
           "radiomics_percentile_image.csv", "i3_image.csv"]]
INPUTS += [Path(__file__), Path(__file__).with_name("nckuh_only_study_20261102.py")]


def load_reference():
    patients = pd.read_csv(AUDIT / "patient_label_linkage_private.csv", dtype={"patient_id": str})
    kidneys = pd.read_csv(AUDIT / "kidney_label_linkage_private.csv", dtype={"patient_id": str})
    images = pd.read_csv(AUDIT / "image_labels_dmsa_private.csv", dtype={"patient_id": str})
    patients = patients[patients.source_status.eq("source_side_code_available")].reset_index(drop=True)
    kidneys = kidneys[kidneys.kidney_APN.notna()].reset_index(drop=True)
    assert len(patients) == 210 and len(kidneys) == 414 and images.kidney_APN.notna().sum() == 820
    assert patients.patient_alias.is_unique and kidneys.kidney_alias.is_unique and images.image_id.is_unique
    assert set(kidneys.patient_alias) == set(patients.patient_alias)
    assert "label" not in images and "previous_patient_label" in images
    assert not kidneys.DMSA_APN.isna().any()
    expected = {"R": {"left": 0, "right": 1}, "L": {"left": 1, "right": 0},
                "B": {"left": 1, "right": 1}, "N": {"left": 0, "right": 0}}
    for row in kidneys.itertuples(index=False):
        assert int(row.kidney_APN) == expected[row.DMSA_APN][row.side]
    return patients, kidneys, images


def prepare():
    from sklearn.model_selection import train_test_split
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        raise FileExistsError("Preserve registered corrected-label protocol")
    audit = json.loads((AUDIT / "audit_summary.json").read_text(encoding="utf-8"))
    assert digest(SOURCE) == audit["source_sha256"] and audit["source_unchanged"]
    patients, kidneys, _ = load_reference()
    assignments = []
    for seed in SEEDS:
        train, rest = train_test_split(np.arange(len(patients)), train_size=126,
                                      random_state=seed, stratify=patients.DMSA_APN)
        development, test = train_test_split(rest, train_size=42, random_state=seed + 1000,
                                             stratify=patients.DMSA_APN.to_numpy()[rest])
        assert not set(train).intersection(rest)
        assert set(train) | set(development) | set(test) == set(range(210))
        for role, indices in [("train", train), ("development", development), ("exploratory_test", test)]:
            for index in indices:
                assignments.append(dict(seed=seed, patient_alias=patients.patient_alias.iloc[index], role=role,
                                        DMSA_APN=patients.DMSA_APN.iloc[index]))
    pd.DataFrame(assignments).to_csv(OUT / "patient_assignments.csv", index=False)
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        clinical_source_sha256=audit["source_sha256"],
        cohort=dict(patients=210, imaged_kidneys=414, images=820, positive_kidneys=262, negative_kidneys=152),
        reference="User-defined side-specific DMSA_APN from original clinical workbook; not inherited patient CSV outcomes",
        missing="Exclude 51 no-report and 3 blank-code patients from all supervised roles; never label missing as negative",
        temporal_caveat="Source-code-available exploratory cohort. One patient's malformed image date remains flagged; admission proximity does not establish exact DMSA timing. No definitive episode-adjudication claim.",
        endpoint="One probability/label per imaged kidney; all available images from that kidney form its bag",
        split="Ten fixed random patient draws, stratified by R/L/B/N, 126 train / 42 development / 42 exploratory test; no k-fold",
        seeds=SEEDS, groups=GROUPS, primary_fixed_method=GROUPS[10],
        fixed_reference_pair=GROUPS[3],
        features="Reuse image-only caches aligned by exact image_id/order. Aggregate mean and population SD within kidney, not across both kidneys",
        preprocessing="Train-patient images only for DenseNet StandardScaler/PCA32; training kidneys only for median imputation, StandardScaler, ANOVA32 and balanced LR C=.1",
        neural="Unchanged existing train_neural: gated/identity/graph/negative controls, 300 fixed epochs, AdamW lr .001 wd .001; no new hyperparameter selection",
        fusion="Unchanged fixed arithmetic probability averages; no test-derived weights, thresholds or best-seed choice",
        prohibited_predictors="All DMSA/clinical diagnosis fields, MRN, source sheet/row, admission dates, prior labels; side is only an anatomical grouping key",
        image_pipeline="Unchanged external non-CYCH/non-NCKUH whole-kidney model and existing contour-assisted QC/fallback; no mixed-site segmentation finetune",
        stage="Development-only baseline. Do not export or score exploratory-test predictions until a separate method lock",
        uncertainty="10000 stratified patient-cluster bootstraps of fixed development predictions; resample patients by R/L/B/N, retaining both kidneys and all repeated draw appearances; conditional within-cohort intervals",
        historical_exploration="These patients/images were already explored under old labels. New labels/splits do not create an independent, previously unseen cohort",
        interpretation="This is an endpoint/reference/cohort correction, not a paired performance gain versus the historical 264-patient experiments")
    write_json(OUT / "protocol.json", protocol)
    lock = dict(utc=datetime.now(timezone.utc).isoformat(),
                inputs={str(p.relative_to(ROOT)).replace('\\', '/'): digest(p) for p in INPUTS},
                assignments_sha256=digest(OUT / "patient_assignments.csv"),
                protocol_sha256=digest(OUT / "protocol.json"), clinical_source_sha256=digest(SOURCE))
    write_json(OUT / "pre_run_lock.json", lock)
    print("Registered corrected-label 210-patient / 414-kidney fixed-method development baseline", flush=True)


def verify_lock():
    lock = json.loads((OUT / "pre_run_lock.json").read_text(encoding="utf-8"))
    assert digest(SOURCE) == lock["clinical_source_sha256"]
    assert digest(OUT / "patient_assignments.csv") == lock["assignments_sha256"]
    assert digest(OUT / "protocol.json") == lock["protocol_sha256"]
    for name, sha in lock["inputs"].items():
        assert digest(ROOT / name) == sha, name


def run():
    import joblib
    import torch
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    torch.set_num_threads(4)
    verify_lock()
    if (OUT / "development_predictions.csv").exists():
        raise FileExistsError("Preserve completed corrected-label experiment")
    (OUT / "models").mkdir(exist_ok=True)
    patients, kidneys, images = load_reference()
    cached = pd.read_csv(FEATURES / "image_manifest.csv", dtype={"patient_id": str})
    assert images.image_id.tolist() == cached.image_id.tolist()
    assert images.filename.tolist() == cached.filename.tolist()
    assert images.patient_id.tolist() == cached.patient_id.tolist()
    y = kidneys.kidney_APN.to_numpy(int)
    row_groups = [np.flatnonzero(images.kidney_alias.eq(alias)) for alias in kidneys.kidney_alias]
    assert sum(map(len, row_groups)) == 820 and set(np.concatenate(row_groups)) == set(np.flatnonzero(images.kidney_APN.notna()))
    dense = np.load(FEATURES / "dense_tight_image.npy")
    feature_arrays = []
    for filename in ["radiomics_percentile_image.csv", "i3_image.csv"]:
        frame = pd.read_csv(FEATURES / filename).set_index("image_id").reindex(images.image_id)
        if filename.startswith("radiomics"):
            frame = frame[[c for c in frame if not c.endswith(("_Energy", "_TotalEnergy"))]]
        values = frame.to_numpy(float)
        assert np.isfinite(values).all()
        feature_arrays.append(values)
    arrays = [dense, *feature_arrays]
    expected_shapes = [(1027, 1024), (1027, 457), (1027, 25)]
    assert [a.shape for a in arrays] == expected_shapes
    summaries = [np.stack([np.r_[a[g].mean(0), a[g].std(0)] for g in row_groups]) for a in arrays]
    assignments = pd.read_csv(OUT / "patient_assignments.csv")
    outputs, metrics, training = [], [], []
    for seed in SEEDS:
        mapping = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
        roles = kidneys.patient_alias.map(mapping)
        assert roles.notna().all() and kidneys.assign(role=roles).groupby("patient_alias").role.nunique().max() == 1
        train = np.flatnonzero(roles.eq("train"))
        dev = np.flatnonzero(roles.eq("development"))
        assert kidneys.patient_alias.iloc[train].nunique() == 126 and kidneys.patient_alias.iloc[dev].nunique() == 42
        scores = {}
        for group, values, select in zip(GROUPS[:3], summaries, [True, True, False]):
            selector = SelectKBest(f_classif, k=32) if select else VarianceThreshold()
            head = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), selector,
                                 LogisticRegression(C=.1, class_weight="balanced", solver="liblinear", max_iter=2000, random_state=seed))
            head.fit(values[train], y[train])
            scores[group] = head.predict_proba(values)[:, list(head.classes_).index(1)]
            joblib.dump(head, OUT / "models" / f"{seed}_{group.split()[0]}_head.joblib", compress=3)
        scores[GROUPS[3]] = (scores[GROUPS[0]] + scores[GROUPS[1]]) / 2
        train_images = np.concatenate([row_groups[i] for i in train])
        assert set(images.patient_alias.iloc[train_images]) == set(mapping[mapping.eq("train")].index)
        scaler = StandardScaler().fit(dense[train_images])
        standardized = scaler.transform(dense)
        pca = PCA(32, random_state=seed).fit(standardized[train_images])
        reduced = pca.transform(standardized).astype(np.float32)
        width = max(map(len, row_groups))
        bags = np.zeros((len(y), width, 32), dtype=np.float32)
        valid = np.zeros((len(y), width), dtype=bool)
        for i, rows in enumerate(row_groups):
            bags[i, :len(rows)] = reduced[rows]
            valid[i, :len(rows)] = True
        bags, valid = torch.from_numpy(bags), torch.from_numpy(valid)
        joblib.dump(dict(scaler=scaler, pca=pca), OUT / "models" / f"{seed}_preprocess.joblib", compress=3)
        for group, mode in [(GROUPS[4], "gated"), (GROUPS[5], "identity"), (GROUPS[6], "graph"), (GROUPS[7], "negative")]:
            scores[group], state, attention, _, parameters = train_neural(seed, mode, train, y, bags, valid)
            torch.save(state, OUT / "models" / f"{seed}_{mode}.pth")
            training.append(dict(seed=seed, group=group, parameters=parameters, train_patients=126,
                                 train_kidneys=len(train), train_images=len(train_images)))
        scores[GROUPS[8]] = (scores[GROUPS[7]] + scores[GROUPS[1]]) / 2
        scores[GROUPS[9]] = (scores[GROUPS[7]] + scores[GROUPS[2]]) / 2
        scores[GROUPS[10]] = (scores[GROUPS[7]] + scores[GROUPS[1]] + scores[GROUPS[2]]) / 3
        scores[GROUPS[11]] = (scores[GROUPS[0]] + scores[GROUPS[1]] + scores[GROUPS[2]]) / 3
        for group in GROUPS:
            metrics.append(dict(seed=seed, role="development", group=group, patients=42,
                                kidneys=len(dev), prevalence=float(y[dev].mean()), **metric(y[dev], scores[group][dev])))
            for i in dev:
                outputs.append(dict(seed=seed, group=group, patient_alias=kidneys.patient_alias.iloc[i],
                                    kidney_alias=kidneys.kidney_alias.iloc[i], DMSA_APN=kidneys.DMSA_APN.iloc[i],
                                    side=kidneys.side.iloc[i], label=int(y[i]), score=float(scores[group][i])))
        print(f"Completed development draw {seed}: 42 patients, {len(dev)} kidneys", flush=True)
    pd.DataFrame(outputs).to_csv(OUT / "development_predictions.csv", index=False)
    pd.DataFrame(metrics).to_csv(OUT / "development_metrics.csv", index=False)
    pd.DataFrame(training).to_csv(OUT / "training_records.csv", index=False)
    summary = pd.DataFrame(metrics).groupby("group")[["auroc", "ap", "auprc", "brier", "prevalence"]].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.to_csv(OUT / "development_summary.csv")
    print(summary[["auroc_mean", "ap_mean"]].to_string(), flush=True)


def bootstrap():
    verify_lock()
    predictions = pd.read_csv(OUT / "development_predictions.csv")
    points = pd.read_csv(OUT / "development_metrics.csv")
    reference = predictions[predictions.group.eq(GROUPS[10])]
    aliases = sorted(reference.patient_alias.unique())
    codes = reference.groupby("patient_alias").DMSA_APN.first().reindex(aliases).to_numpy()
    strata = [np.flatnonzero(codes == code) for code in ["R", "L", "B", "N"]]
    assert all(len(s) for s in strata)
    matrices = []
    for seed in SEEDS:
        draw = predictions[predictions.seed.eq(seed)]
        pivot = draw.pivot(index="kidney_alias", columns="group", values="score").reindex(columns=GROUPS)
        key = draw.groupby("kidney_alias").first().reindex(pivot.index)
        clusters = np.array([aliases.index(alias) for alias in key.patient_alias])
        matrices.append((clusters, key.label.to_numpy(int), pivot.to_numpy(float)))
    rng = np.random.default_rng(20261102)
    batches, total = [], 0
    while total < 10000:
        size = min(1000, 10000 - total)
        sample = np.concatenate([rng.choice(s, (size, len(s)), replace=True) for s in strata], axis=1)
        counts = np.zeros((size, len(aliases)), dtype=float)
        np.add.at(counts, (np.arange(size)[:, None], sample), 1)
        values = np.zeros((size, len(GROUPS), 4))
        keep = np.ones(size, dtype=bool)
        for clusters, target, scores in matrices:
            weights = counts[:, clusters]
            keep &= (weights[:, target == 0].sum(1) > 0) & (weights[:, target == 1].sum(1) > 0)
            for j in range(len(GROUPS)):
                values[:, j] += weighted_metrics_batch(target, scores[:, j], weights)
        batches.append(values[keep] / len(SEEDS))
        total += int(keep.sum())
    samples = np.concatenate(batches)[:10000]
    np.save(OUT / "development_patient_cluster_bootstrap.npy", samples)
    rows = []
    for j, group in enumerate(GROUPS):
        for k, name in enumerate(["auroc", "ap", "auprc", "brier"]):
            lo, hi = np.quantile(samples[:, j, k], [.025, .975])
            rows.append(dict(group=group, metric=name, mean=float(points[points.group.eq(group)][name].mean()),
                             low=float(lo), high=float(hi), distinct_patients=len(aliases), replicates=10000,
                             interpretation="Conditional development fixed-prediction patient-cluster CI; not external validation"))
    pd.DataFrame(rows).to_csv(OUT / "development_confidence_intervals.csv", index=False)
    write_json(OUT / "completion_audit.json", dict(
        completed_utc=datetime.now(timezone.utc).isoformat(), clinical_source_unchanged=True,
        reference_cohort=210, reference_kidneys=414, reference_images=820,
        patient_grouped=True, supervised_missing_outcomes=0, scored_roles=["development"],
        exploratory_test_scored=False, bootstrap_unit="patient, stratified by R/L/B/N",
        temporal_review_pending_patients=1, source_DMSA_dates_independently_verified=False,
        outputs={name: digest(OUT / name) for name in ["development_predictions.csv", "development_metrics.csv",
                 "development_summary.csv", "development_confidence_intervals.csv"]}))
    print("Completed 10000 patient-cluster development replicates; exploratory test remains unscored", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "run", "bootstrap"])
    globals()[parser.parse_args().stage]()
