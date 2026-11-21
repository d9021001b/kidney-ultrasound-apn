"""Paper 2, step 6: assemble every probe-specific model on the same kidneys and check the target.

All scores are patient-grouped out-of-fold scores from the same five shared folds. Two fusions are
pre-specified (no fitted weights): `fusion_frozen_e2e` = mean within-group rank of the nested
frozen-feature score and the probe-specific end-to-end (context) score; `fusion_all` additionally
averages in the NCKUH paper-1 final and CYCH paper-1 final. Target: AUROC > 0.8 and AP > 0.8 in both groups.
"""

import json

import pandas as pd
from scipy.stats import rankdata

from p2_common import P2, P2_OUT, cluster_interval, metric

E2E = P2_OUT / "end_to_end/image_oof_logits_specific_context_specific_masked_pooled_init.csv"
MODELS = {
    "NCKUH paper-1 chain (final)": "nckuh1_final",
    "  fixed chain": "nckuh1_fixed_chain",
    "  tile+USFM four-state hierarchy": "nckuh1_hierarchy",
    "  URI-CADS embedding head": "nckuh1_uricads",
    "CYCH paper-1 chain (final)": "cych_final",
    "  reliable-negative GCN-MIL": "cych_gcn_mil",
    "  PyRadiomics probe": "cych_pyradiomics",
    "  directional I3 probe": "cych_i3",
    "Both paper chains, equal mean": "combined_equal_mean",
    "Probe-specific nested frozen features": "frozen_nested",
    "Probe-specific end-to-end, context crop": "specific_context",
    "Probe-specific end-to-end, masked kidney": "specific_masked",
    "Pooled pre-training then probe fine-tuning": "pooled_init",
    "Fusion: frozen + end-to-end": "fusion_frozen_e2e",
    "Fusion: all four families": "fusion_all",
}


def main():
    frozen = pd.read_csv(P2_OUT / "nested_frozen/kidney_oof_scores.csv")
    e2e = pd.read_csv(E2E)
    detector = json.loads((P2 / "router/detector_validation.json").read_text(encoding="utf-8"))
    summary = {"target": "AUROC > 0.8 and AP > 0.8 in each probe group",
               "router": {"cv_pool": detector["cv_pool_patient_grouped_oof"],
                          "held_out": detector["held_out_exploratory_test_probe_only"]},
               "groups": {}, "test_APN_scored": False}
    rows = []
    for probe in ("convex", "linear"):
        table = pd.read_csv(P2_OUT / f"paper_methods/{probe}_kidney_oof_scores.csv").set_index("kidney_alias")
        a = frozen[frozen.route.eq(probe)].set_index("kidney_alias")
        assert (a.loc[table.index, "label"] == table.label).all() and len(a) == len(table)
        table["frozen_nested"] = a.loc[table.index, "score"]
        part = e2e[e2e.route.eq(probe)].groupby("kidney_alias")[["specific_context", "specific_masked", "pooled_init"]].mean()
        table = table.join(part, how="inner", validate="one_to_one")
        assert len(table) == len(a)
        rank = lambda column: rankdata(table[column]) / len(table)
        table["fusion_frozen_e2e"] = (rank("frozen_nested") + rank("specific_context")) / 2
        table["fusion_all"] = (rank("frozen_nested") + rank("specific_context") + rank("nckuh1_final") + rank("cych_final")) / 4
        group = {"patients": int(table.patient_alias.nunique()), "kidneys": len(table),
                 "prevalence": float(table.label.mean()), "models": {}}
        for title, column in MODELS.items():
            result = metric(table.label, table[column]) | cluster_interval(table.patient_alias, table.label, table[column])
            group["models"][column] = result
            rows.append({"probe": probe, "model": title, "auroc": result["auroc"], "auroc_low": result["auroc_95pct"][0],
                         "auroc_high": result["auroc_95pct"][1], "ap": result["ap"], "ap_low": result["ap_95pct"][0],
                         "ap_high": result["ap_95pct"][1]})
        best = max(group["models"], key=lambda n: min(group["models"][n]["auroc"], group["models"][n]["ap"]))
        group["highest_model_post_hoc"] = best
        group["target_met_by_any_model"] = bool(min(group["models"][best]["auroc"], group["models"][best]["ap"]) > 0.8)
        group["any_upper_interval_reaches_0.8_auroc"] = bool(max(m["auroc_95pct"][1] for m in group["models"].values()) > 0.8)
        summary["groups"][probe] = group
        table.reset_index().to_csv(P2_OUT / f"all_models_{probe}_kidney_oof_scores.csv", index=False)
    summary["target_met_in_both_groups"] = all(g["target_met_by_any_model"] for g in summary["groups"].values())
    out = pd.DataFrame(rows)
    out.to_csv(P2_OUT / "summary_table.csv", index=False)
    (P2_OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.set_option("display.width", 200)
    print(out.round(3).to_string(index=False))
    for probe, group in summary["groups"].items():
        print(probe, {k: group[k] for k in ("patients", "kidneys", "prevalence", "highest_model_post_hoc",
                                           "target_met_by_any_model", "any_upper_interval_reaches_0.8_auroc")})
    print("target met in both groups:", summary["target_met_in_both_groups"])


if __name__ == "__main__":
    main()
