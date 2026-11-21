"""
Two controls that decide how to read the near-chance single-center result.

N1  NEGATIVE CONTROL  (--control permute)
    Patient labels are permuted within the site before any split is drawn, so the
    images carry no information about the (shuffled) target.  The whole pipeline --
    patient-level StratifiedGroupKFold, inner validation, early stopping, threshold
    selection -- then runs unchanged.  An honest pipeline must return AUC ~ 0.50.
    Anything meaningfully above 0.50 here is leakage in the harness, and it would
    invalidate the real run's numbers as well.

P1  POSITIVE CONTROL  (--control site)
    Same pipeline, same pixels, both hospitals pooled, but the target is replaced by
    WHICH HOSPITAL the image came from.  Trivial low-level features (white-caliper
    fraction, channel statistics, crop geometry) already separate the two sites at
    AUC ~0.99 under patient-grouped CV, so a working pipeline must score high here.

    Read the two together:
      N1 ~ 0.50 and P1 high  ->  the harness is sound and learns from these pixels;
                                 the absent inflammation signal is a property of the
                                 data (label, crop, or reference standard), not a bug.
      N1 ~ 0.50 and P1 ~ 0.50 ->  the pixels or the training recipe are the problem.
      N1 well above 0.50      ->  stop; the harness leaks and every number is suspect.

Patient ids are hospital-local, so the pooled control prefixes them with the site
before grouping.  (Measured intersection is currently empty, but a future extraction
could collide, and a silent collision would merge two children into one group.)

Usage
    python controls_negative_positive.py --control permute --site CYCH  --out results_controls/permute_CYCH
    python controls_negative_positive.py --control permute --site NCKUH --out results_controls/permute_NCKUH
    python controls_negative_positive.py --control site                 --out results_controls/site
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import single_center_cv as S


def _cnn_only_args(base):
    """train_cnn / predict_probs read their knobs off an argparse namespace."""
    ns = argparse.Namespace(
        arch=base.arch, epochs=base.epochs, patience=base.patience,
        batch_size=base.batch_size, lr=base.lr, weight_decay=base.weight_decay,
        monitor="patient_auc", patient_agg="mean",
        workers=base.workers, amp=base.amp, strict_determinism=False,
    )
    return ns


def build_cohort(args):
    """Return (img_df, pat, tag_notes) for the requested control."""
    manifest = Path(args.manifest)
    if args.control == "permute":
        img_df, notes = S.load_cohort(manifest, args.site, True)
        pat = S.build_patient_table(img_df)
        rng = np.random.default_rng(args.permute_seed)
        # Permute the PATIENT label vector, then push it back onto the images.  Doing it
        # at patient level keeps labels patient-constant (as in the real data) and keeps
        # the class balance exactly, so the folds look identical to the real run.
        shuffled = pat["label"].to_numpy().copy()
        rng.shuffle(shuffled)
        pat = pat.assign(label=shuffled)
        lut = dict(zip(pat["patient_id"], pat["label"]))
        img_df = img_df.assign(label=img_df["patient_id"].map(lut).astype(int))
        notes["control"] = "N1 negative: patient labels permuted"
        notes["permute_seed"] = args.permute_seed
        notes["label_agreement_with_truth"] = float(
            (shuffled == S.build_patient_table(
                S.load_cohort(manifest, args.site, True)[0])["label"].to_numpy()).mean())
        return img_df, pat, notes

    parts, notes = [], {"control": "P1 positive: target = hospital of origin"}
    for site, y in (("NCKUH", 1), ("CYCH", 0)):
        d, n = S.load_cohort(manifest, site, True)
        d = d.assign(label=y, patient_id=site + "_" + d["patient_id"].astype(str))
        parts.append(d)
        notes[f"{site}_images"] = int(len(d))
    img_df = pd.concat(parts, ignore_index=True)
    img_df = img_df.sort_values(["patient_id", "filename"]).reset_index(drop=True)
    img_df["img_idx"] = np.arange(len(img_df))
    pat = S.build_patient_table(img_df)
    notes["patients"] = int(len(pat))
    return img_df, pat, notes


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--control", required=True, choices=["permute", "site"])
    p.add_argument("--site", choices=["CYCH", "NCKUH"],
                   help="Required for --control permute; ignored for --control site.")
    p.add_argument("--out", required=True)
    p.add_argument("--manifest", default=str(S.DEFAULT_MANIFEST))
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--permute-seed", type=int, default=12345)
    p.add_argument("--inner-folds", type=int, default=4)
    p.add_argument("--arch", default="densenet121")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)

    if args.control == "permute" and not args.site:
        p.error("--control permute requires --site")

    t0 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    img_df, pat, notes = build_cohort(args)
    inner_args = _cnn_only_args(args)

    print(f"control={args.control}  patients={len(pat)}  images={len(img_df)}  "
          f"prevalence={pat['label'].mean():.3f}  device={device}")

    rows, fold_aucs = [], []
    for fold, (tr_idx, te_idx) in enumerate(S.outer_splits(pat, args.folds, args.seed)):
        S.set_all_seeds(args.seed * 100 + fold)
        pat_tr, pat_te = pat.iloc[tr_idx], pat.iloc[te_idx]
        i_tr, i_va = S.inner_split(pat_tr, args.inner_folds, args.seed)
        pids_tr = pat_tr.iloc[i_tr]["patient_id"].tolist()
        pids_va = pat_tr.iloc[i_va]["patient_id"].tolist()
        pids_te = pat_te["patient_id"].tolist()
        S.assert_patient_disjoint(pids_tr, pids_va, pids_te, f"control fold {fold}")

        tr_img = img_df[img_df["patient_id"].isin(pids_tr)]
        va_img = img_df[img_df["patient_id"].isin(pids_va)]
        te_img = img_df[img_df["patient_id"].isin(pids_te)]

        model, hist = S.train_cnn(args.arch, tr_img, va_img, inner_args, device,
                                  args.seed, f"c{fold}")
        probs = S.predict_probs(model, te_img, device, args.batch_size,
                                args.workers, args.amp)
        # aggregate_patient_scores already carries the label through from the image rows.
        merged = S.aggregate_patient_scores(te_img, probs)
        truth = dict(zip(pat_te["patient_id"], pat_te["label"]))
        mismatch = [p for p, l in zip(merged["patient_id"], merged["label"])
                    if truth.get(p) != l]
        S._require(not mismatch,
                   f"patient label disagreement between image rows and patient table: {mismatch[:3]}")
        auc = S.fast_auc(merged["label"].to_numpy(), merged["score_mean"].to_numpy())
        fold_aucs.append(auc)
        merged["fold"] = fold
        rows.append(merged)
        print(f"  fold {fold+1}/{args.folds}  best epoch {hist.get('best_epoch')}  "
              f"test patients {len(merged)}  patient AUC {auc:.3f}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    oof = pd.concat(rows, ignore_index=True)
    y = oof["label"].to_numpy()
    s = oof["score_mean"].to_numpy()
    pooled_auc = S.fast_auc(y, s)

    # Patient cluster bootstrap.  Every row here is already one patient, so resampling
    # rows with replacement IS the patient bootstrap; stratify by class to keep both
    # classes present in every draw.
    rng = np.random.default_rng(20260910)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    draws = []
    for _ in range(args.bootstrap):
        idx = np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)])
        a = S.fast_auc(y[idx], s[idx])
        if np.isfinite(a):
            draws.append(a)
    draws = np.asarray(draws)
    lo, hi = np.percentile(draws, [2.5, 97.5]) if len(draws) else (np.nan, np.nan)

    if args.control == "permute":
        verdict = ("PASS - consistent with a leak-free harness" if lo <= 0.5 <= hi else
                   "FAIL - the 95% CI excludes 0.50 on permuted labels; the harness leaks")
    else:
        verdict = ("PASS - the pipeline learns from these pixels" if lo > 0.70 else
                   "INCONCLUSIVE - site is not recovered; suspect the pixels or the recipe")

    result = {
        "control": args.control, "site": args.site, "notes": notes,
        "patients": int(len(pat)), "images": int(len(img_df)),
        "prevalence": float(pat["label"].mean()),
        "per_fold_patient_auc": [float(a) for a in fold_aucs],
        "pooled_patient_auc": float(pooled_auc),
        "ci95": [float(lo), float(hi)], "bootstrap_draws": int(len(draws)),
        "verdict": verdict,
        "config": vars(args), "elapsed_min": round((time.time() - t0) / 60, 1),
    }
    (out / "control_result.json").write_text(json.dumps(result, indent=2, default=str),
                                             encoding="utf-8")
    oof.to_csv(out / "oof_patient.csv", index=False, encoding="utf-8")

    print("\n" + "=" * 78)
    print(f"{args.control.upper()} control"
          + (f" ({args.site})" if args.site else " (both sites pooled)"))
    print(f"  per-fold patient AUC : {[round(a, 3) for a in fold_aucs]}")
    print(f"  pooled patient AUC   : {pooled_auc:.3f}   95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"  VERDICT: {verdict}")
    print(f"  elapsed {result['elapsed_min']} min -> {out}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
