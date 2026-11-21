"""Route the historical 210-patient development draws before score reporting."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from sklearn.metrics import average_precision_score, roc_auc_score

from prepare_nckuh_383_contour_cohort_20261111 import digest


ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "02_experiments"
OLD = EXP / "NCKUH_DMSA_fullcohort_fullframe_20261103"
LABELS = EXP / "NCKUH_DMSA_label_audit_20261102"
OLD_SCORES = EXP / "NCKUH_DMSA_uricads_full_detector_20261105_r2"
NEW = EXP / "NCKUH_revised_383_audit_20261111"
ROUTER = EXP / "NCKUH_revised_383_probe_router_20261112"
OUT = ROUTER / "historical_210_route_audit"
WORKBOOK = ROOT / "2004-2019-NCKUH-亞大(1).xlsx"
GROUP = "official_uri_fixed_quarter_reference"
SEEDS = (20261102, 20261103, 20261104)


def score(frame):
    y, p = frame.label.to_numpy(int), frame.score.to_numpy(float)
    return {"patients": int(frame.patient_alias.nunique()), "kidneys": len(frame),
            "positive_fraction": float(y.mean()),
            "auroc": float(roc_auc_score(y, p)),
            "ap": float(average_precision_score(y, p))}


def interval(frame, seed, draws=2000):
    members = [np.flatnonzero(frame.patient_alias.eq(alias))
               for alias in sorted(frame.patient_alias.unique())]
    labels, scores = frame.label.to_numpy(int), frame.score.to_numpy(float)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        idx = np.concatenate([members[k] for k in rng.integers(0, len(members), len(members))])
        if np.unique(labels[idx]).size == 2:
            values.append((roc_auc_score(labels[idx], scores[idx]),
                           average_precision_score(labels[idx], scores[idx])))
    if not values:
        return {"valid_bootstraps": 0, "auroc_95pct": None, "ap_95pct": None}
    limits = np.quantile(values, [.025, .975], axis=0)
    return {"valid_bootstraps": len(values), "auroc_95pct": limits[:, 0].tolist(),
            "ap_95pct": limits[:, 1].tolist()}


def repeated_draw_cluster_interval(frame, draws=10000):
    aliases = sorted(frame.patient_alias.unique())
    bags = {(seed, alias): group[["label", "score"]].to_numpy(float)
            for (seed, alias), group in frame.groupby(["seed", "patient_alias"])}
    rng = np.random.default_rng(20261112)
    sampled_metrics = []
    for _ in range(draws):
        sample = [aliases[k] for k in rng.integers(0, len(aliases), len(aliases))]
        per_draw = []
        for seed in SEEDS:
            rows = [bags[(seed, alias)] for alias in sample if (seed, alias) in bags]
            if not rows:
                break
            table = np.concatenate(rows)
            if np.unique(table[:, 0]).size != 2:
                break
            per_draw.append((roc_auc_score(table[:, 0], table[:, 1]),
                             average_precision_score(table[:, 0], table[:, 1])))
        if len(per_draw) == len(SEEDS):
            sampled_metrics.append(np.mean(per_draw, axis=0))
    limits = np.quantile(sampled_metrics, [.025, .975], axis=0)
    return {"unique_patients_in_development_draw_union": len(aliases),
            "valid_bootstraps": len(sampled_metrics),
            "auroc_95pct": limits[:, 0].tolist(), "ap_95pct": limits[:, 1].tolist()}


def main():
    sheet = load_workbook(WORKBOOK, read_only=True, data_only=True)["完整資料彙整"]
    rows = sheet.iter_rows(values_only=True)
    header = list(next(rows))
    assert header[0] == "id" and header[7] == "Probe"
    workbook = {str(row[0]).strip().lstrip("0"): str(row[7]).strip().lower()
                for row in rows if row[0] is not None}
    assert len(workbook) == 383
    old_link = pd.read_csv(LABELS / "patient_label_linkage_private.csv", dtype={"patient_id": str})
    old_link["workbook_probe"] = old_link.patient_id.str.lstrip("0").map(workbook).fillna("not_in_new_workbook")
    old_manifest = pd.read_csv(OLD / "deidentified_source_manifest.csv")
    old_private = pd.read_csv(LABELS / "image_labels_dmsa_private.csv", dtype={"patient_id": str, "exam_date_raw": str})
    new_private = pd.read_csv(NEW / "source_linkage_private.csv", dtype={"patient_id": str, "exam_date_raw": str})
    new_manifest = pd.read_csv(NEW / "contour_image_manifest.csv")
    routes = pd.read_csv(ROUTER / "image_routes.csv")
    old_images = old_manifest[["image_id", "patient_alias", "view"]].merge(
        old_private[["image_id", "patient_id", "exam_date_raw"]], on="image_id", validate="one_to_one")
    new_images = new_private[["image_id", "patient_id", "exam_date_raw"]].merge(
        new_manifest[["image_id", "view"]], on="image_id", validate="one_to_one"
    ).merge(routes[["image_id", "route"]], on="image_id", validate="one_to_one")
    old_images["patient_id"] = old_images.patient_id.str.lstrip("0")
    new_images["patient_id"] = new_images.patient_id.str.lstrip("0")
    key = ["patient_id", "exam_date_raw", "view"]
    assert not old_images.duplicated(key).any() and not new_images.duplicated(key).any()
    matched = old_images.merge(new_images[key + ["route"]], on=key, how="left", validate="one_to_one")
    assert len(matched) == 820 and matched.route.notna().sum() == 787
    matched = matched.merge(old_link[["patient_alias", "workbook_probe"]], on="patient_alias", validate="many_to_one")
    assert not (matched.workbook_probe.eq("convex") & matched.route.eq("linear")).any()
    patient_probe = old_link.set_index("patient_alias").workbook_probe
    predictions = pd.read_csv(OLD_SCORES / "development_predictions.csv")
    predictions = predictions[predictions.group.eq(GROUP)].copy()
    assert set(predictions.seed) == set(SEEDS)
    assert predictions.groupby("seed").patient_alias.nunique().eq(42).all()
    predictions["probe_route"] = predictions.patient_alias.map(patient_probe)
    assert predictions.probe_route.notna().all()
    old_roles = pd.read_csv(OLD / "patient_assignments.csv")
    assert set(predictions.patient_alias) <= set(old_roles.patient_alias)
    for seed in SEEDS:
        expected = set(old_roles.loc[old_roles.seed.eq(seed) & old_roles.role.eq("development"), "patient_alias"])
        actual = set(predictions.loc[predictions.seed.eq(seed), "patient_alias"])
        assert expected == actual
    published = pd.read_csv(OLD_SCORES / "development_metrics.csv")
    published = published[published.group.eq(GROUP) & published.role.eq("development")]
    report = {
        "source_sha256": {"workbook": digest(WORKBOOK),
                          "old_patient_linkage": digest(LABELS / "patient_label_linkage_private.csv"),
                          "old_image_manifest": digest(OLD / "deidentified_source_manifest.csv"),
                          "old_development_predictions": digest(OLD_SCORES / "development_predictions.csv"),
                          "new_image_routes": digest(ROUTER / "image_routes.csv")},
        "old_reference_group": GROUP,
        "interpretation": "Fixed historical classifier and historical patient splits; route-conditional score evaluation, NOT refitting a new probe-specific classifier.",
        "old_patients_by_workbook_probe": old_link[old_link.patient_alias.isin(old_manifest.patient_alias)].workbook_probe.value_counts().to_dict(),
        "old_images_by_workbook_and_matched_route": pd.crosstab(
            matched.workbook_probe, matched.route.fillna("unmatched")
        ).to_dict(),
        "matched_old_images": int(matched.route.notna().sum()),
        "old_images_total": len(matched),
        "test_APN_scored": False, "draws": [],
    }
    if report["old_images_by_workbook_and_matched_route"].get("linear"):
        raise AssertionError("Historical reference has linear-routed images; audit before interpretation")
    summary_rows = []
    for seed in SEEDS:
        draw = predictions[predictions.seed.eq(seed)].reset_index(drop=True)
        full = score(draw)
        source = published[published.seed.eq(seed)].iloc[0]
        assert abs(full["auroc"] - source.auroc) < 1e-10
        assert abs(full["ap"] - source.ap) < 1e-10
        groups = {}
        for group in ("convex", "both", "not_in_new_workbook"):
            subset = draw[draw.probe_route.eq(group)].reset_index(drop=True)
            if len(subset) and subset.label.nunique() == 2:
                groups[group] = score(subset) | interval(subset, seed)
            else:
                groups[group] = {"patients": int(subset.patient_alias.nunique()),
                                 "kidneys": len(subset), "auroc": None, "ap": None}
            summary_rows.append({"seed": seed, "probe_route": group, **groups[group]})
        report["draws"].append({"seed": seed, "full_historical": full,
                                "routed_patient_groups": groups})
        summary_rows.append({"seed": seed, "probe_route": "full_historical", **full})
    report["three_draw_mean"] = {
        group: {name: float(np.mean([draw["routed_patient_groups"][group][name]
                                     for draw in report["draws"]]))
                for name in ("auroc", "ap")}
        for group in ("convex",) if all(draw["routed_patient_groups"][group]["auroc"] is not None
                                      for draw in report["draws"])
    }
    report["three_draw_mean"]["full_historical"] = {
        name: float(np.mean([draw["full_historical"][name] for draw in report["draws"]]))
        for name in ("auroc", "ap")}
    report["convex_three_draw_cluster_interval"] = repeated_draw_cluster_interval(
        predictions[predictions.probe_route.eq("convex")].reset_index(drop=True)
    )
    assert abs(report["three_draw_mean"]["full_historical"]["ap"] - .8250005524744628) < 1e-10
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(OUT / "development_by_probe_and_seed.csv", index=False)
    (OUT / "route_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"old_patients": report["old_patients_by_workbook_probe"],
                      "old_image_routes": report["old_images_by_workbook_and_matched_route"],
                      "three_draw_mean": report["three_draw_mean"],
                      "draws": [{"seed": d["seed"], "convex": d["routed_patient_groups"]["convex"]}
                                for d in report["draws"]]}, indent=2))


if __name__ == "__main__":
    main()
