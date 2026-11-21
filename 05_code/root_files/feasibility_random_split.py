"""
Feasibility design: patient-level random hold-out against a pre-specified target.

The study question is not "what is the cross-validated performance" but "can this be
made to work at all on this cohort" -- a feasibility assessment with a success criterion
fixed in advance: AUROC >= 0.80 AND AUPRC >= 0.80 on a randomly sampled hold-out set.

What this script does, and why it does it that way
--------------------------------------------------
One draw is exactly the design asked for: split patients at random (default 80 / 20,
stratified on label, no patient in both halves), train on the 80, carve an inner
validation split from it for early stopping and threshold selection, score the 20 once.

It then repeats that draw R times and reports the DISTRIBUTION of the result, plus the
fraction of draws that met the target.  This is not cross-validation by another name and
it does not change the design; it is what keeps a single-hold-out feasibility claim
honest.  A single draw is one sample from that distribution, so reporting one draw
without knowing the spread cannot distinguish "the method works" from "this partition
was lucky".  Reporting the draw that happened to clear the bar, out of several run, is
selection on the outcome -- and partition dependence of exactly this kind is what an
earlier reviewer of this project's related manuscript objected to.  Pre-specify the
target, report the spread, and the claim holds up.

Read the AUPRC target against the prevalence floor
--------------------------------------------------
AUPRC has no fixed chance level: a random classifier scores at the positive rate.  The
two cohorts differ sharply, so "AUPRC >= 0.80" is a very different bar at each:

    NCKUH   67.1 % positive images  ->  chance AUPRC ~ 0.67, target is +0.13 over chance
    CYCH    51.8 % positive images  ->  chance AUPRC ~ 0.52, target is +0.28 over chance

The script prints the floor next to every AUPRC and also reports the AUPRC LIFT over
prevalence, which is comparable between sites in a way the raw number is not.

Usage
    python feasibility_random_split.py --site NCKUH --manifest yolo_crops/manifest.csv \
        --out results_feasibility/NCKUH --draws 20
    python feasibility_random_split.py --site CYCH --out results_feasibility/CYCH --draws 20
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit

import single_center_cv as S

ARMS = ["clinical", "cnn", "fused", "svm"]
TARGET_AUROC = 0.80
TARGET_AUPRC = 0.80


def harness_args(a) -> argparse.Namespace:
    """A full single_center_cv namespace, so run_fold sees every knob it expects."""
    ns = S.build_parser().parse_args(["--site", a.site, "--out", str(a.out)])
    for k, v in dict(
        manifest=a.manifest, folds=2, seeds=1, inner_folds=a.inner_folds, arch=a.arch,
        epochs=a.epochs, patience=a.patience, batch_size=a.batch_size, lr=a.lr,
        weight_decay=a.weight_decay, monitor=a.monitor, patient_agg="mean",
        pca_components=a.pca_components, svm_C=a.svm_C, clinical_C=a.clinical_C,
        fusion_C=a.fusion_C, workers=a.workers, amp=a.amp, strict_determinism=False,
        keep_first_exam_only=True, smoke=False,
    ).items():
        setattr(ns, k, v)
    return ns


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--site", required=True, choices=["CYCH", "NCKUH"])
    p.add_argument("--out", required=True)
    p.add_argument("--manifest", default=str(S.DEFAULT_MANIFEST))
    p.add_argument("--test-frac", type=float, default=0.20)
    p.add_argument("--draws", type=int, default=20)
    p.add_argument("--start-seed", type=int, default=0)
    p.add_argument("--inner-folds", type=int, default=4)
    p.add_argument("--arch", default="densenet121")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--monitor", default="patient_auc", choices=["patient_ba", "patient_auc"])
    p.add_argument("--pca-components", type=int, default=50)
    p.add_argument("--svm-C", type=float, default=1.0)
    p.add_argument("--clinical-C", type=float, default=1.0)
    p.add_argument("--fusion-C", type=float, default=1.0)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--device", default=None)
    p.add_argument("--smoke", action="store_true", help="2 draws, 3 epochs.")
    a = p.parse_args(argv)
    if a.smoke:
        a.draws, a.epochs, a.patience, a.bootstrap = 2, 3, 3, 200

    t0 = time.time()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(a.device) if a.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    S._selftest_stats(verbose=False)

    img_df, notes = S.load_cohort(Path(a.manifest), a.site, True)
    pat = S.build_patient_table(img_df)
    prev_img = float((img_df["label"] == 1).mean())
    prev_pat = float((pat["label"] == 1).mean())

    print(f"{a.site}: {len(pat)} patients, {len(img_df)} images")
    print(f"  positive rate -- patients {prev_pat:.3f}, images {prev_img:.3f}")
    print(f"  AUPRC chance floor equals the positive rate; target {TARGET_AUPRC} is "
          f"+{TARGET_AUPRC - prev_pat:.3f} over it at patient level")
    print(f"  {a.draws} random draws, test fraction {a.test_frac}, device {device}\n")

    per_draw, pat_frames = [], []
    for d in range(a.draws):
        seed = a.start_seed + d
        S.set_all_seeds(seed)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=a.test_frac, random_state=seed)
        tr_idx, te_idx = next(sss.split(np.zeros(len(pat)), pat["label"].to_numpy()))

        ns = harness_args(a)
        try:
            pat_out, img_out, counts = S.run_fold(ns, device, seed, 0, pat, img_df, tr_idx, te_idx)
        except Exception as exc:                      # a degenerate draw must not kill the run
            print(f"  draw {d + 1}/{a.draws} (seed {seed}) FAILED: {type(exc).__name__}: {exc}")
            per_draw.append({"draw": d, "seed": seed, "failed": str(exc)})
            continue

        pat_out = pat_out.assign(draw=d)
        pat_frames.append(pat_out)
        y = pat_out["label"].to_numpy()
        rec = {"draw": d, "seed": seed,
               "n_test_patients": int(len(pat_out)),
               "n_test_pos": int(y.sum()), "n_test_neg": int((y == 0).sum()),
               "test_prevalence": float(y.mean())}
        for arm in ARMS:
            col = f"score_{arm}"
            if col not in pat_out.columns:
                continue
            s = pat_out[col].to_numpy()
            m = S.binary_metrics(y, s, (s >= 0.5).astype(int))
            rec[f"{arm}_auroc"] = m["auc"]
            rec[f"{arm}_auprc"] = m["average_precision"]
            rec[f"{arm}_auprc_lift"] = m["average_precision"] - float(y.mean())
            rec[f"{arm}_hit"] = bool(m["auc"] >= TARGET_AUROC
                                     and m["average_precision"] >= TARGET_AUPRC)
        per_draw.append(rec)
        hits = " ".join(f"{arm}{'*' if rec.get(f'{arm}_hit') else ''}="
                        f"{rec.get(f'{arm}_auroc', float('nan')):.3f}/"
                        f"{rec.get(f'{arm}_auprc', float('nan')):.3f}" for arm in ARMS)
        print(f"  draw {d + 1}/{a.draws} (seed {seed}, {rec['n_test_patients']} test pt)  "
              f"AUROC/AUPRC  {hits}")

    df = pd.DataFrame(per_draw)
    ok = df[~df.get("failed", pd.Series([np.nan] * len(df))).notna()] if "failed" in df else df
    if not len(ok):
        print("\nEvery draw failed; nothing to summarise.")
        return 1

    # ---- distribution across draws, and the pre-specified target ---------------------
    summary = {"site": a.site, "manifest": a.manifest, "draws_run": int(len(ok)),
               "test_frac": a.test_frac, "prevalence_patient": prev_pat,
               "prevalence_image": prev_img,
               "target": {"auroc": TARGET_AUROC, "auprc": TARGET_AUPRC}, "arms": {}}
    lines = []
    for arm in ARMS:
        ca, cp = f"{arm}_auroc", f"{arm}_auprc"
        if ca not in ok:
            continue
        ro, pr = ok[ca].to_numpy(float), ok[cp].to_numpy(float)
        hit = ok[f"{arm}_hit"].to_numpy(bool)
        st = {
            "auroc_median": float(np.median(ro)), "auroc_min": float(ro.min()),
            "auroc_max": float(ro.max()),
            "auroc_iqr": [float(np.percentile(ro, 25)), float(np.percentile(ro, 75))],
            "auprc_median": float(np.median(pr)), "auprc_min": float(pr.min()),
            "auprc_max": float(pr.max()),
            "auprc_lift_median": float(np.median(pr) - prev_pat),
            "draws_meeting_target": int(hit.sum()),
            "fraction_meeting_target": float(hit.mean()),
            "auroc_ge_080": int((ro >= TARGET_AUROC).sum()),
            "auprc_ge_080": int((pr >= TARGET_AUPRC).sum()),
        }
        summary["arms"][arm] = st
        lines.append(
            f"  {arm:<9} AUROC med {st['auroc_median']:.3f} "
            f"[{st['auroc_min']:.3f}, {st['auroc_max']:.3f}]   "
            f"AUPRC med {st['auprc_median']:.3f} (lift {st['auprc_lift_median']:+.3f})   "
            f"target met {st['draws_meeting_target']}/{len(ok)}")

    print("\n" + "=" * 92)
    print(f"FEASIBILITY -- {a.site}, {len(ok)} random patient-level draws, "
          f"{int(round(a.test_frac * 100))}% held out")
    print(f"Pre-specified target: AUROC >= {TARGET_AUROC} AND AUPRC >= {TARGET_AUPRC}")
    print(f"AUPRC chance floor at this site: {prev_pat:.3f} (patient level)")
    print("-" * 92)
    print("\n".join(lines))
    print("-" * 92)

    best = max(summary["arms"].items(), key=lambda kv: kv[1]["fraction_meeting_target"])
    if best[1]["draws_meeting_target"] == 0:
        verdict = (f"TARGET NOT MET in any of {len(ok)} draws by any arm. "
                   f"Best median AUROC was {best[1]['auroc_median']:.3f} ({best[0]}).")
    elif best[1]["fraction_meeting_target"] >= 0.8:
        verdict = (f"TARGET MET in {best[1]['draws_meeting_target']}/{len(ok)} draws "
                   f"by {best[0]}; the result is stable across partitions.")
    else:
        verdict = (f"TARGET MET in only {best[1]['draws_meeting_target']}/{len(ok)} draws "
                   f"by {best[0]}. Partition-dependent -- reporting a single passing draw "
                   f"would not be defensible; report the fraction.")
    summary["verdict"] = verdict
    print(verdict)
    print("=" * 92)

    df.to_csv(out / "per_draw.csv", index=False, encoding="utf-8")
    if pat_frames:
        pd.concat(pat_frames, ignore_index=True).to_csv(
            out / "per_draw_patient_predictions.csv", index=False, encoding="utf-8")
    summary["elapsed_min"] = round((time.time() - t0) / 60, 1)
    summary["cohort_notes"] = {k: v for k, v in notes.items() if not k.endswith("_files")}
    (out / "feasibility_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"elapsed {summary['elapsed_min']} min -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
