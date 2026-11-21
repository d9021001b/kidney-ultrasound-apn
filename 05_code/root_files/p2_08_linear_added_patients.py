"""Paper 2, step 8 (provisional sensitivity): add linear-only patients that have only predicted contours.

32 linear-only patients in the train/development roles have no accepted contour; the linear-specific
segmenter produced component masks that are still `pending_manual_review`. Here those components are
used as provisional kidney regions. Each component is treated as one view of the kidney named by the
file's side code. The frozen RadImageNet DenseNet-121 tight-canvas feature (the source most often chosen
for the linear group) is extracted with the same function used for the registered cohort.

Two questions, answered on the shared patient folds (added patients receive folds by a stratified split):
  A. same 79 registered linear patients scored, training with vs without the added patients (paired);
  B. all linear patients scored (registered + added).
Head: train-only standardise + PCA16 + balanced LR C=0.1 on own/contralateral/difference features,
fixed in advance. Exploratory-test patients are excluded entirely.
"""

import json
import sys
import warnings

import cv2
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from sklearn.model_selection import StratifiedKFold

from nckuh_dmsa_fullcohort_fullframe_20261103 import crop_masked, percentile_gray
from p2_03_landscape import head, paired
from p2_05_nested_frozen import shared_folds
from p2_common import CACHE, OLD_ROUTER, P2, REGISTRY, ROOT, SEED, STATE_SIDES, WORKBOOK_V2, cluster_interval, metric
from p2_features import routed_images

sys.path.insert(0, str(ROOT / "02_experiments/CYCH_honest_reevaluation_20261003"))
from extract_site import dense_tight  # noqa: E402

CAND = OLD_ROUTER / "linear_segmentation_finetune/missing_linear_candidates"
OUT = P2 / "linear_added_patients"


def workbook_states():
    sheet = load_workbook(WORKBOOK_V2, read_only=True, data_only=True)["完整資料彙整"]
    rows = sheet.iter_rows(values_only=True)
    header = list(next(rows))
    i_id, i_dmsa, i_probe = header.index("id"), header.index("DMSA_APN"), header.index("Probe")
    book = {str(r[i_id]).strip().lstrip("0"): (str(r[i_dmsa]).strip().upper(), str(r[i_probe]).strip().lower())
            for r in rows if r[i_id] is not None}
    link = pd.read_csv(REGISTRY / "patient_linkage_private.csv", dtype=str)
    return {alias: book.get(pid.lstrip("0")) for alias, pid in zip(link.patient_alias, link.patient_id)}


def prepare():
    components = pd.read_csv(CAND / "component_review/component_manifest_deidentified.csv")
    components = components[components.role.isin(("train", "development"))].reset_index(drop=True)
    states = workbook_states()
    components["state"] = [states[a][0] if states[a] else None for a in components.patient_alias]
    components["workbook_probe"] = [states[a][1] if states[a] else None for a in components.patient_alias]
    dropped = components[components.state.isna() | components.workbook_probe.ne("linear")].patient_alias.nunique()
    components = components[components.state.notna() & components.workbook_probe.eq("linear")].reset_index(drop=True)
    paths = pd.read_csv(CAND / "source_linkage_private.csv").set_index("candidate_id").source_path
    (OUT / "percentile").mkdir(parents=True, exist_ok=True)
    (OUT / "masks").mkdir(exist_ok=True)
    for item in components.itertuples(index=False):
        image = cv2.imread(str(paths[item.candidate_id]), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(CAND / "component_review/masks" / f"{item.instance_id}.png"), cv2.IMREAD_GRAYSCALE)
        roi, roi_mask = crop_masked(image, (mask > 0).astype(np.uint8))
        cv2.imwrite(str(OUT / "percentile" / f"{item.instance_id}.png"), percentile_gray(roi, roi_mask))
        cv2.imwrite(str(OUT / "masks" / f"{item.instance_id}.png"), roi_mask * 255)
    components["full_side"] = components.side.map({"L": "left", "R": "right"})
    components["kidney_alias"] = components.patient_alias + "_" + components.side
    components["label"] = [STATE_SIDES[s][0 if side == "left" else 1] for s, side in zip(components.state, components.full_side)]
    features = dense_tight(OUT / "percentile", OUT / "masks", components.instance_id.tolist())
    np.save(OUT / "radimagenet_dense_tight.npy", features)
    components.drop(columns=["bbox_xywh"]).to_csv(OUT / "components.csv", index=False)
    print("added components", len(components), "patients", components.patient_alias.nunique(),
          "dropped patients (not linear / absent in v2)", dropped, flush=True)


def kidney_frame(image_table, values, side_col):
    table = image_table.groupby("kidney_alias", sort=False).agg(
        patient_alias=("patient_alias", "first"), side=(side_col, "first"), label=("label", "first")).reset_index()
    members = [np.flatnonzero(image_table.kidney_alias.to_numpy() == k) for k in table.kidney_alias]
    return table, np.stack([values[m].mean(0) for m in members])


def evaluate():
    patients, _, images = routed_images()
    base_images = images[images.route.eq("linear") & images.role.isin(("train", "development"))].reset_index(drop=True)
    base_images = base_images.rename(columns={"kidney_APN": "label"})
    fold_of = shared_folds(patients, images[images.role.isin(("train", "development")) & images.route.isin(("convex", "linear"))])
    dense = np.load(CACHE / "radimagenet_dense_tight.npy", mmap_mode="r")
    base_table, base_x = kidney_frame(base_images, np.asarray(dense[base_images.cache_row.to_numpy()], float), "side")
    base_table["added"] = False

    components = pd.read_csv(OUT / "components.csv")
    add_table, add_x = kidney_frame(components, np.load(OUT / "radimagenet_dense_tight.npy").astype(float), "full_side")
    add_table["added"] = True
    people = components.drop_duplicates("patient_alias")[["patient_alias", "state"]].reset_index(drop=True)
    for k, (_, hold) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(people, people.state)):
        fold_of.update({alias: k for alias in people.patient_alias.iloc[hold]})

    table = pd.concat([base_table, add_table], ignore_index=True)
    x = paired(np.r_[base_x, add_x], table)
    y, fold, added = table.label.to_numpy(int), table.patient_alias.map(fold_of).to_numpy(), table.added.to_numpy()
    without, with_added = np.full(len(y), np.nan), np.full(len(y), np.nan)
    for k in range(5):
        hold = np.flatnonzero(fold == k)
        fit_all = np.flatnonzero(fold != k)
        fit_base = np.flatnonzero((fold != k) & ~added)
        without[hold] = head(16, 0.1).fit(x[fit_base], y[fit_base]).predict_proba(x[hold])[:, 1]
        with_added[hold] = head(16, 0.1).fit(x[fit_all], y[fit_all]).predict_proba(x[hold])[:, 1]

    def block(mask, scores):
        return metric(y[mask], scores[mask]) | cluster_interval(table.patient_alias[mask], y[mask], scores[mask]) | {
            "patients": int(table.patient_alias[mask].nunique()), "kidneys": int(mask.sum()), "prevalence": float(y[mask].mean())}

    rng = np.random.default_rng(SEED)
    registered = np.flatnonzero(~added)
    groups = [registered[table.patient_alias.to_numpy()[registered] == a] for a in np.unique(table.patient_alias.to_numpy()[registered])]
    deltas = []
    for _ in range(2000):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        if np.unique(y[idx]).size == 2:
            a, b = metric(y[idx], with_added[idx]), metric(y[idx], without[idx])
            deltas.append((a["auroc"] - b["auroc"], a["ap"] - b["ap"]))
    low, high = np.quantile(deltas, [.025, .975], axis=0)
    report = {"protocol": __doc__, "warning": "Added masks are unreviewed predicted components; provisional only.",
              "added_patients": int(add_table.patient_alias.nunique()), "added_kidneys": len(add_table),
              "A_registered_patients_scored": {"trained_without_added": block(~added, without),
                                               "trained_with_added": block(~added, with_added),
                                               "paired_difference_auroc_95pct": [float(low[0]), float(high[0])],
                                               "paired_difference_ap_95pct": [float(low[1]), float(high[1])]},
              "B_all_linear_patients_scored": block(np.ones(len(y), bool), with_added),
              "added_patients_only": block(added, with_added), "test_APN_scored": False}
    table.assign(fold=fold, score_without_added=without, score_with_added=with_added).to_csv(OUT / "kidney_oof_scores.csv", index=False)
    (OUT / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "protocol"}, indent=1))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    if not (OUT / "components.csv").exists():
        prepare()
    evaluate()
