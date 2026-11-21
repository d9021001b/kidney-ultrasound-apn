"""Train-only clinical disease heads with the fixed URI-CADS image laterality head."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import nckuh_dmsa_fullcohort_fullframe_20261103 as base
import nckuh_dmsa_hierarchical_paired_20261103 as hierarchy
from audit_nckuh_383_clinical_availability_20261111 import patient_key, sex_code


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "02_experiments/NCKUH_revised_383_audit_20261111"
CACHE = ROOT / "02_experiments/NCKUH_revised_383_contour_cache_20261111"
URI = ROOT / "02_experiments/NCKUH_revised_383_uricads_20261111"
OUT = ROOT / "02_experiments/NCKUH_revised_383_clinical_hierarchy_20261111"


def fit_head(x, y, seed):
    model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                          StandardScaler(), LogisticRegression(solver="liblinear",
                          class_weight="balanced", random_state=seed, max_iter=2000))
    search = GridSearchCV(model, {"logisticregression__C": [0.01, 0.1, 1.0]},
                          scoring="roc_auc", cv=StratifiedKFold(4, shuffle=True,
                          random_state=seed), n_jobs=1, refit=True)
    search.fit(x, y)
    return search


def clinical_table(aliases):
    book = load_workbook(ROOT / "2004-2019-NCKUH-亞大.xlsx", read_only=True, data_only=True)
    age = {patient_key(r[0]): float(r[4]) for r in list(book["完整資料彙整"].values)[1:]}
    sex = {}
    for sheet in ("2007-2012", "2014-2019", "工作表2"):
        for row in list(book[sheet].values)[1:]:
            if row[0] is None or row[3] is None:
                continue
            key = patient_key(row[0])
            value = sex_code(row[3])
            assert key not in sex or sex[key] == value
            sex[key] = value
    linkage = pd.read_csv(REGISTRY / "patient_linkage_private.csv", dtype=str).set_index("patient_alias")
    ids = linkage.loc[aliases, "patient_id"].map(patient_key)
    result = np.asarray([(age[key], float(sex[key] == "M")) for key in ids], dtype=np.float32)
    assert np.isfinite(result).all()
    return result


def main():
    if (OUT / "development_metrics.csv").exists():
        raise FileExistsError("Preserve development experiment")
    images = pd.read_csv(REGISTRY / "contour_image_manifest.csv")
    kidneys = pd.read_csv(REGISTRY / "contour_kidney_manifest.csv")
    assignments = pd.read_csv(REGISTRY / "patient_assignments.csv")
    active = assignments[assignments.role.isin(["train", "development"]) &
                         assignments.patient_alias.isin(images.patient_alias)].reset_index(drop=True)
    assert active.role.value_counts().to_dict() == {"train": 204, "development": 70}
    selected_images = images[images.patient_alias.isin(active.patient_alias)].reset_index(drop=True)
    selected_kidneys = kidneys[kidneys.patient_alias.isin(active.patient_alias)].reset_index(drop=True)
    selected_kidneys = selected_kidneys.merge(active[["patient_alias", "role"]],
                                               on="patient_alias", validate="many_to_one")
    states = active.DMSA_APN.map(hierarchy.STATE_CODE).to_numpy(int)
    train = np.flatnonzero(active.role.eq("train"))
    dev = np.flatnonzero(selected_kidneys.role.eq("development"))
    clinical = clinical_table(active.patient_alias)
    image_ids = images.index[images.patient_alias.isin(active.patient_alias)].to_numpy()
    semantic = np.load(CACHE / "uricads_fold_pathology_scores.npy")[image_ids].mean(axis=1)
    semantic_pairs = base.patient_pair_features(semantic, selected_images, active)
    semantic_sym, _ = hierarchy.structured_views(semantic_pairs)
    fold_emb = np.load(CACHE / "uricads_fold_embeddings.npy")[image_ids]
    embedding_pairs = base.patient_pair_features(fold_emb.reshape(len(fold_emb), -1),
                                                  selected_images, active)
    _, embedding_anti = hierarchy.structured_views(embedding_pairs)
    saved = joblib.load(URI / "models/uri_embedding_folds.joblib")
    p_left = saved["left_given_unilateral"].predict_proba(embedding_anti)[:, 1]
    prior_predictions = pd.read_csv(URI / "development_predictions.csv")
    prior = prior_predictions[prior_predictions.group.eq("uri_embedding_folds")].set_index("kidney_alias")
    target = selected_kidneys.iloc[dev].reset_index(drop=True)
    assert set(target.kidney_alias) == set(prior.index)
    results, predictions, selections = [], [], []
    for group, x in {"age_sex_state": clinical,
                     "age_sex_plus_uri_semantic_state": np.c_[clinical, semantic_sym]}.items():
        any_label = (states != hierarchy.STATE_CODE["N"]).astype(int)
        any_model = fit_head(x[train], any_label[train], 20261111)
        affected = train[any_label[train] == 1]
        bilateral_label = (states == hierarchy.STATE_CODE["B"]).astype(int)
        bilateral_model = fit_head(x[affected], bilateral_label[affected], 20261212)
        p_any = any_model.predict_proba(x)[:, 1]
        p_b = bilateral_model.predict_proba(x)[:, 1] * p_any
        p_u = p_any - p_b
        probability = np.c_[1 - p_any, p_u * p_left, p_u * (1 - p_left), p_b]
        assert np.allclose(probability.sum(axis=1), 1)
        score = hierarchy.kidney_scores(probability, active, selected_kidneys)[dev]
        label = target.kidney_APN.to_numpy(int)
        results.append({"group": group, "patients": 70, "kidneys": len(dev),
                        "auroc": roc_auc_score(label, score),
                        "ap": average_precision_score(label, score)})
        predictions.extend({"group": group, "patient_alias": row.patient_alias,
                            "kidney_alias": row.kidney_alias, "side": row.side,
                            "label": int(row.kidney_APN), "score": float(value)}
                           for row, value in zip(target.itertuples(index=False), score))
        for head, model in (("any", any_model), ("bilateral_given_any", bilateral_model)):
            selections.append({"group": group, "head": head,
                               "inner_cv_auroc": model.best_score_,
                               "C": model.best_params_["logisticregression__C"]})
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(OUT / "development_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "development_predictions.csv", index=False)
    pd.DataFrame(selections).to_csv(OUT / "training_inner_cv_selections.csv", index=False)
    (OUT / "protocol.json").write_text(json.dumps({
        "cohort": "revised 383 patient roster, 347 contour-ready",
        "train_patients": 204, "development_patients": 70,
        "heads": "train-only any-APN and bilateral-given-APN, age/sex or age/sex plus frozen URI semantic features",
        "laterality": "unchanged previously trained URI embedding-fold laterality head",
        "selection": "four-fold stratified training-only CV, C in [0.01,0.1,1]",
        "test_scored": False,
        "warning": "development exploratory; no test labels/scores used"}, indent=2), encoding="utf-8")
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
