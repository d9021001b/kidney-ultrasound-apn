"""Why is paper 2 lower than paper 1? Refit paper 1's own chain on paper 1's own cohort with NEW draws.

Same 210 patients, same archived features, same code, same fixed weights. Only the patient draw changes:
the three original 126/42/42 assignments (reproduction check against the archived final scores) and ten
new stratified 126/42 draws that were never used while the chain was being developed. Read-only with
respect to every paper-1 folder; outputs go to the paper-2 folder. No exploratory-test scoring: new draws
only use a development part, and the patients are those of the published development experiments.
"""

import json
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from threadpoolctl import threadpool_limits

import nckuh_dmsa_fullcohort_fullframe_20261103 as base
import nckuh_dmsa_multiquery_mil_20261103 as old_mil
from p2_common import EXP, P2, cluster_interval, metric
from probe_nckuh_383_legacy_fixed_components_20261112 import fit_components
from probe_nckuh_383_legacy_hierarchy_20261112 import score_head

OUT = P2 / "paper1_resplit_check"
ORIGINAL = (20261102, 20261103, 20261104)
NEW = tuple(range(20270001, 20270011))
FINAL = EXP / "NCKUH_DMSA_uricads_full_detector_20261105_r2/development_predictions.csv"


def chain(data, train_patients, dev, seed, inner_seed):
    patients, kidneys, images = data["tables"]
    parts = fit_components(patients, kidneys, images, data["tile"], data["rad"], data["mirror"], data["texture"],
                           train_patients, dev, seed)
    labels_np = kidneys.kidney_APN.to_numpy(int)
    labels = torch.from_numpy(labels_np.astype(np.float32)).cuda()
    people = patients.iloc[train_patients].reset_index(drop=True)
    fit_local, select_local = train_test_split(np.arange(len(people)), train_size=100, random_state=inner_seed + 6000,
                                               stratify=people.DMSA_APN)
    fit_alias, select_alias = set(people.patient_alias.iloc[fit_local]), set(people.patient_alias.iloc[select_local])
    fit = np.flatnonzero(kidneys.patient_alias.isin(fit_alias))
    selection = np.flatnonzero(kidneys.patient_alias.isin(select_alias))
    values, patch_patients, patch_views, groups = data["patches"]
    bags, mask, views, _ = old_mil.prepare_bags(values, patch_patients, patch_views, groups, fit_alias, seed)
    _, history = old_mil.train_model("cross_attention_q4", seed, bags, mask, views, labels, fit, old_mil.EPOCHS, selection)
    choices = [(-(old_mil.metric(labels_np[selection], s)["auroc"] + old_mil.metric(labels_np[selection], s)["ap"]) / 2, e)
               for e, s in history.items()]
    _, epoch = min(choices)
    train_alias = set(people.patient_alias)
    train = np.flatnonzero(kidneys.patient_alias.isin(train_alias))
    bags, mask, views, _ = old_mil.prepare_bags(values, patch_patients, patch_views, groups, train_alias, seed + 100)
    model, _ = old_mil.train_model("cross_attention_q4", seed + 10000, bags, mask, views, labels, train, (epoch,), None)
    model.eval()
    with torch.inference_mode():
        mil = torch.sigmoid(model(bags[dev], mask[dev], views[dev])[0]).cpu().numpy()
    fixed = .75 * (.75 * (.75 * parts["tile_two_heads"] + .25 * mil) + .25 * parts["knn"]) + .25 * parts["aligned"]
    hierarchy, _ = score_head(patients, kidneys, images, np.c_[data["tile"], data["usfm"]], train_patients, dev, seed)
    uri, _ = score_head(patients, kidneys, images, data["uri"], train_patients, dev, seed)
    return {"fixed_chain": fixed, "tile_usfm_reference": .75 * fixed + .25 * hierarchy,
            "final": .75 * (.75 * fixed + .25 * hierarchy) + .25 * uri}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    patients, kidneys, images = old_mil.load_reference()
    with np.load(EXP / "NCKUH_DMSA_native_tile_probe_20261103/tile_features.npz") as bundle:
        tile = bundle["mean"]
    uri = np.load(EXP / "NCKUH_DMSA_uricads_official_frozen_20261105/official_fold_embeddings.npy")
    data = {"tables": (patients, kidneys, images), "tile": tile,
            "rad": np.load(base.OUT / "radimagenet_color.npy"),
            "mirror": np.load(EXP / "NCKUH_DMSA_reflection_probe_20261103/radimagenet_color_horizontal_reflection.npy"),
            "usfm": np.load(EXP / "NCKUH_DMSA_usfm_frozen_20261104/usfm_gap.npy"),
            "uri": uri.reshape(len(uri), -1),
            "texture": pd.read_csv(EXP / "NCKUH_DMSA_fullcohort_directional_20261103/directional_features.csv"
                                   ).set_index("image_id").loc[images.image_id].reset_index(drop=True),
            "patches": old_mil.patch_data(images, kidneys)}
    assignments = pd.read_csv(base.OUT / "patient_assignments.csv")
    saved = pd.read_csv(FINAL)
    saved = saved[saved.group.eq("official_uri_fixed_quarter_reference")].set_index(["seed", "kidney_alias"]).score
    labels = kidneys.kidney_APN.to_numpy(int)
    rows, frames = [], []
    for kind, seeds in (("original", ORIGINAL), ("new", NEW)):
        for seed in seeds:
            if kind == "original":
                role = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
                train_patients = np.flatnonzero(patients.patient_alias.map(role).eq("train"))
                dev = np.flatnonzero(kidneys.patient_alias.map(role).eq("development"))
            else:
                rest, _ = next(StratifiedShuffleSplit(1, test_size=42, random_state=seed).split(patients, patients.DMSA_APN))
                a, b = next(StratifiedShuffleSplit(1, test_size=42, random_state=seed + 1).split(
                    rest, patients.DMSA_APN.iloc[rest]))
                train_patients = np.sort(rest[a])
                dev = np.flatnonzero(kidneys.patient_alias.isin(patients.patient_alias.iloc[rest[b]]))
            assert len(train_patients) == 126 and kidneys.patient_alias.iloc[dev].nunique() == 42
            scores = chain(data, train_patients, dev, seed, seed)
            row = {"kind": kind, "seed": seed}
            for name, value in scores.items():
                m = metric(labels[dev], value)
                row |= {f"{name}_auroc": m["auroc"], f"{name}_ap": m["ap"]}
            if kind == "original":
                expected = saved.loc[[(seed, k) for k in kidneys.kidney_alias.iloc[dev]]].to_numpy()
                row["max_abs_difference_from_archived_final"] = float(np.max(np.abs(scores["final"] - expected)))
                row |= {"archived_final_auroc": metric(labels[dev], expected)["auroc"],
                        "archived_final_ap": metric(labels[dev], expected)["ap"]}
            rows.append(row)
            frames.append(kidneys.iloc[dev][["patient_alias", "kidney_alias", "kidney_APN"]].assign(
                kind=kind, seed=seed, final=scores["final"]))
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "per_draw_metrics.csv", index=False)
    pd.concat(frames, ignore_index=True).to_csv(OUT / "development_predictions.csv", index=False)
    summary = {"protocol": __doc__}
    for kind in ("original", "new"):
        part = table[table.kind.eq(kind)]
        summary[kind] = {"draws": len(part),
                         **{c: {"mean": float(part[c].mean()), "sd": float(part[c].std()),
                                "min": float(part[c].min()), "max": float(part[c].max())}
                            for c in ("fixed_chain_auroc", "fixed_chain_ap", "tile_usfm_reference_auroc",
                                      "tile_usfm_reference_ap", "final_auroc", "final_ap")}}
    summary["original"]["max_abs_difference_from_archived_final"] = float(
        table.max_abs_difference_from_archived_final.max())
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "protocol"}, indent=1))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    with threadpool_limits(limits=6):
        main()
