"""Test whether within-kidney percentile normalization removes bilateral information."""
import argparse
from datetime import datetime, timezone
import json
from multiprocessing import Pool
import sys
import warnings

import cv2
import joblib
import numpy as np
import pandas as pd

from nckuh_only_study_20261102 import ROOT, FEATURES, OUT as BASE, SEEDS, digest, write_json, metric
from nckuh_representation_screen_20261102 import OUT as PREVIOUS, aggregate, load_metadata

OUT = ROOT / "02_experiments/NCKUH_raw_intensity_probe_20261102"
CONFIGS = [(name, head) for name in ["raw_summary", "raw_view", "raw_bilateral", "raw_normalized_bilateral"] for head in ["full_lr", "pca32_lr", "top32_hgb"]]


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                    hypothesis="Per-image percentile remapping may suppress between-kidney echogenicity differences; this is a computational hypothesis, not a confirmed pathological mechanism",
                    motivation_sources=["https://pubmed.ncbi.nlm.nih.gov/12069464/", "https://pubmed.ncbi.nlm.nih.gov/17987762/"],
                    extraction="Same PyRadiomics descriptor implementation and mask as the percentile baseline, but native raw 0-255 grayscale rather than independently percentile-mapped images; Energy and TotalEnergy excluded",
                    clinical_boundary="No calibrated renal size claim; no guessed lesion or anatomical compartment annotation; no deletion of calipers based on disease label",
                    configurations=CONFIGS,
                    heads="Training-only imputation and standardization; full balanced LR C=.01, PCA32 balanced LR C=.1, or ANOVA32 balanced HGB(max_iter100, max_leaf_nodes7, min_samples_leaf15, l2_regularization1)",
                    cohorts="Same NCKUH-only fixed patient assignments. All development candidates retained; no test scoring",
                    promotion="At least +.03 mean AUROC, +.02 mean AP, >=6 joint improving draws against original fixed fusion; goal remains both AUROC and AP > .8",
                    user_confirmation_sha256=digest(BASE / "clinical_confirmation_20261102.json"),
                    extractor_source_sha256=digest(ROOT / "02_experiments/CYCH_honest_reevaluation_20261003/extract_site.py"))
    if (OUT / "protocol.json").exists():
        raise FileExistsError("Preserve existing protocol")
    write_json(OUT / "protocol.json", protocol)
    write_json(OUT / "run_lock.json", dict(source_sha256=digest(__file__), protocol_sha256=digest(OUT / "protocol.json")))
    print("Raw versus normalized intensity hypothesis fixed before extraction", flush=True)


def require_lock():
    lock = json.loads((OUT / "run_lock.json").read_text(encoding="utf-8"))
    assert digest(__file__) == lock["source_sha256"] and digest(OUT / "protocol.json") == lock["protocol_sha256"]


def extract_one(job):
    cv2.setNumThreads(1)
    sys.path.insert(0, str(ROOT / "02_experiments/CYCH_honest_reevaluation_20261003"))
    from extract_site import _radiomics_worker
    return _radiomics_worker(job)


def extract():
    require_lock()
    manifest, patients = load_metadata()
    jobs = [(ident, FEATURES / "native" / f"{ident}.png", FEATURES / "masks" / f"{ident}.png") for ident in manifest.image_id]
    values = []
    with Pool(8) as workers:
        for i, record in enumerate(workers.imap(extract_one, jobs, chunksize=4), 1):
            values.append(record)
            if i % 100 == 0:
                print(f"Native raw PyRadiomics {i}/{len(jobs)}", flush=True)
    frame = pd.DataFrame(values).set_index("image_id").reindex(manifest.image_id)
    frame = frame[[c for c in frame if not c.endswith(("_Energy", "_TotalEnergy"))]]
    assert frame.shape == (1027, 457) and np.isfinite(frame.to_numpy()).all()
    frame.to_csv(OUT / "raw_radiomics_image.csv")
    for mode in ["summary", "view", "bilateral"]:
        np.save(OUT / f"raw_{mode}.npy", aggregate(frame.to_numpy(), manifest, patients, mode))
    # Use matching radiomic columns only; no new supervised feature combination.
    normalized = pd.read_csv(FEATURES / "radiomics_percentile_image.csv").set_index("image_id").reindex(manifest.image_id)[frame.columns]
    matched = aggregate(normalized.to_numpy(), manifest, patients, "bilateral")
    raw = np.load(OUT / "raw_bilateral.npy")
    np.save(OUT / "raw_normalized_bilateral.npy", np.c_[raw, matched])
    print("Native raw and matched percentile descriptor arrays saved", flush=True)


def screen():
    require_lock()
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
    _, patients = load_metadata()
    assignments = pd.read_csv(BASE / "random_patient_assignments.csv")
    (OUT / "models").mkdir(exist_ok=True)
    rows, predictions = [], []
    with threadpool_limits(limits=4), warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning); warnings.simplefilter("ignore", RuntimeWarning)
        for name, head in CONFIGS:
            values = np.load(OUT / f"{name}.npy")
            group = name + "_" + head
            for seed in SEEDS:
                roles = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role.reindex(patients.patient_alias)
                train, dev = np.flatnonzero(roles.eq("train")), np.flatnonzero(roles.eq("development"))
                if head == "top32_hgb":
                    transform = SelectKBest(f_classif, k=32)
                    classifier = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=7, min_samples_leaf=15,
                                                               l2_regularization=1, class_weight="balanced", random_state=seed, early_stopping=False)
                else:
                    transform = PCA(32, random_state=seed) if head == "pca32_lr" else VarianceThreshold()
                    classifier = LogisticRegression(C=.1 if head == "pca32_lr" else .01, class_weight="balanced", solver="liblinear", max_iter=2000, random_state=seed)
                model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), transform, classifier)
                model.fit(values[train], patients.label.to_numpy()[train])
                score = model.predict_proba(values[dev])[:, 1]
                rows.append(dict(seed=seed, group=group, role="development", **metric(patients.label.to_numpy()[dev], score)))
                for i, probability in zip(dev, score):
                    predictions.append(dict(seed=seed, group=group, role="development", patient_alias=patients.patient_alias.iloc[i], label=int(patients.label.iloc[i]), score=float(probability)))
                joblib.dump(model, OUT / "models" / f"{group}_{seed}.joblib", compress=3)
            print(f"Development only completed {group}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "development_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "development_predictions.csv", index=False)
    summary = frame.groupby("group")[["auroc", "ap"]].mean()
    summary["mean_score"] = summary.mean(axis=1)
    summary = summary.sort_values(["mean_score", "group"], ascending=[False, True])
    summary.to_csv(OUT / "development_summary.csv")
    reference = pd.read_csv(BASE / "metrics_per_draw.csv")
    reference = reference[reference.role.eq("development") & reference.group.eq("Fixed three source fusion")].set_index("seed")
    eligible = []
    for group in summary.index:
        delta = frame[frame.group.eq(group)].set_index("seed")[["auroc", "ap"]] - reference[["auroc", "ap"]]
        if delta.auroc.mean() >= .03 and delta.ap.mean() >= .02 and delta.gt(0).all(axis=1).sum() >= 6:
            eligible.append(group)
    write_json(OUT / "candidate_decision.json", dict(selected=eligible[0] if eligible else None, eligible=eligible,
                                                     selection_role="development only", test_scoring_performed=False,
                                                     tested_candidates=len(CONFIGS), goal_met_in_development=bool(eligible and (summary.loc[eligible[0], ["auroc", "ap"]] > .8).all())))
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["prepare", "extract", "screen"])
    a = p.parse_args()
    globals()[a.stage]()
