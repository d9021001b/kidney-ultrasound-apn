#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
pooled_vs_federated.py
======================
Centralized pooled-training upper bound vs. FedAvg, under a valid patient-level
evaluation design.  This script exists to answer the co-author's MAJOR 1 and
MAJOR 4 on the federated-learning manuscript.

WHAT THIS REPLACES
------------------
`fedavg_corrected.py` and `leave_one_site_out_baseline.py` contain **no
train/validation/test split at all**: both call `get_file_label_items()` /
`KidneyDataset()` on the entire site directory and then `evaluate()` on that
same entire directory.  Consequently the manuscript's AUC 0.985 (NCKUH) /
0.979 (CYCH) for the federated global model, and the LOSO *within-site* rows
(0.987 / 0.980), are **training-set numbers**, and federated round 203 was
selected on those same images.  The only genuinely held-out numbers in the
current manuscript are the LOSO *cross-site* rows (0.492 and 0.495).

This script re-runs the whole comparison on frozen, patient-level,
StratifiedGroupKFold test folds that are **shared by all four arms**, so the
arms are directly comparable and nothing is ever selected on a test fold.

THE FOUR ARMS
-------------
ARM 1  local      train on site X's training patients only  -> evaluate on X's test fold
                  (the honest within-site number the manuscript currently lacks)
ARM 2  loso       train on site X's training patients only  -> evaluate on Y's test fold
                  (the cross-site domain-shift number).  ARM 1 and ARM 2 are the SAME
                  trained model scored on two different hospitals, so the contrast
                  between them is exactly paired and contains no training variance.
ARM 3  pooled     train on the union of both sites' training patients
                  -> evaluate on each site's test fold  (the centralized upper bound)
ARM 4  federated  FedAvg between the two sites using ONLY their training patients,
                  with the round selected on the VALIDATION patients of both sites
                  -> evaluate on each site's test fold

HONEST READING OF WHAT THESE ARMS CAN AND CANNOT SHOW
-----------------------------------------------------
With only TWO participating sites, this design supports two distinct claims and
no more, and the manuscript must not blur them:

  * **ARM 2 is the only arm that measures generalization to an unseen hospital.**
    Every other arm evaluates a model whose training set included patients from
    the very hospital it is being tested on.  The manuscript's headline
    "CYCH 0.492 -> 0.979" therefore contrasts an *unseen-hospital* model with a
    *seen-hospital* model; it measures the benefit of multi-site data access,
    not federated generalization to a new hospital.  No arm here can estimate
    performance at a third, never-seen hospital, because no third hospital
    exists in this dataset.

  * **ARM 3 versus ARM 4 is what isolates the cost of the federated algorithm.**
    Both arms see exactly the same training patients and differ only in whether
    the data were pooled centrally or kept on-site and combined by FedAvg.  The
    pooled-minus-federated gap is therefore the price paid for privacy
    preservation, and it is the only contrast in this study attributable to the
    federated *algorithm* rather than to data access.  If ARM 4 is close to
    ARM 3, the defensible claim is **near-lossless privacy preservation**, NOT
    that FedAvg beats centralized training.

  * ARM 1 versus ARM 3/ARM 4 mixes both effects (more data AND, for ARM 4, a
    different optimizer) and must not be reported as a federated-algorithm effect.

DEFECTS IN fedavg_corrected.py THAT ARE FIXED HERE
--------------------------------------------------
1. The round-selection criterion S was computed on the full site directories,
   i.e. on training data.  Here S is computed on held-out VALIDATION patients
   of both sites and never on any test patient.
2. `federated_average()` did `(stateA[k] + stateB[k]) / 2` over *every*
   state_dict key with equal weights.  Here the aggregation weights are
   selectable with `--agg {equal,nk}` (co-author MINOR 2): `nk` weights each
   client by its number of training images, which is what McMahan et al.
   specify; the two sites hold 1027 vs 349 images, so equal weighting silently
   upweights the smaller site roughly threefold per image.
3. `num_batches_tracked` (an int64 counter) was averaged as if it were a
   parameter.  Here BatchNorm buffers are handled explicitly: integer counters
   are combined by `max` and never averaged, and the float running statistics
   are averaged with the same client weights (`--bn-stats average`, default) or
   left at the previous global model's values (`--bn-stats keep_global`).
4. No seeding anywhere, so every number in the manuscript is a single
   unreproducible draw.  Here every RNG is seeded and logged.

RELATIONSHIP TO single_center_cv.py  (anti-drift)
--------------------------------------------------
The cohort loader, the patient-level splitting, the transforms, the model
builder, the sample weighting, the metric functions, the Youden threshold, the
patient-score aggregation and the DeLong test are **imported directly from
`single_center_cv.py`** - not reimplemented.  If that module changes, this
script changes with it, and the single-centre paper and the federated paper
cannot report numbers computed by different code.  The import is mandatory;
this script refuses to run without it.

On top of that, THIS script freezes its own partitions in
`partitions/{site}_seed{r}_assignments.csv` and reuses that file verbatim on
every later run, after checking that the patient set and labels still match, so
repeated runs of *this* script cannot silently re-randomise the folds.

Be precise about what that does NOT cover: `single_center_cv.py` does not read
or write the `partitions/` directory.  It writes its own
`<out>/fold_assignments.csv` (columns site/seed/fold/patient_id/split) and
re-derives its folds from `outer_splits` + `inner_split` each run.  The two
scripts therefore agree because they call the same splitting functions with the
same seeds, not because they share a file.  Verified by direct comparison at
seed 0: identical train/val/test patient sets in all 5 folds at both sites
(NCKUH test folds 53/53/53/52/53, CYCH 20/20/20/19/19).  If you ever change
`outer_splits`, `inner_split` or `--inner-folds` in one script, delete
`partitions/` and re-run both, or the two papers will silently diverge.

COST OF A FULL RUN  (measured on this machine: RTX A6000, Windows 10, torch
2.5.1+cu121, densenet121, --batch-size 16, --workers 0, AMP on)
------------------------------------------------------------------------
Steady-state timings, taken as the mean of epochs 2-3 so that cuDNN autotune and
the first-touch page cache are excluded (epoch 1 is 15-30% slower):

    NCKUH train epoch      613 images    21.4 s
    CYCH  train epoch      211 images     7.7 s
    pooled train epoch     824 images    36.7 s
    both-site validation   281 images     4.9 s   (charged once per epoch to the
                                                   pooled and federated arms,
                                                   half that to arms 1 and 2)

Per (seed, outer fold), with nothing early-stopping, that is ~110 min, so the
5 seeds x 5 folds ceiling is **~46 h**.  If every arm stops around epoch/round 25
it is ~19 h.  Plan for 20-45 h, not 16 h.  The federated arm is the one least
likely to stop early: at --local-epochs 1 the global model improves slowly, and
--fed-patience 20 out of --rounds 60 tolerates a long plateau.

The GPU is NOT the constraint - peak allocation is 1.3 GiB of the card's 48 GiB.
The bottleneck is single-threaded PIL decode plus the torchvision transform
stack, which is why --workers > 0 makes it worse on Windows (see --workers) and
why raising --batch-size buys much less than the memory headroom suggests.

To cut cost: `--seeds 0 1 2` is 3/5 of the above; a single seed is ~1/5 and is
enough to see whether ARM 3 and ARM 4 are close, which is the whole point of the
comparison.  The analysis stage after training is cheap and fixed: ~2.5 min for
the full 5-seed, B=2000 bootstrap, plus ~3 s for the tables and figures.

Usage
-----
    python pooled_vs_federated.py --smoke --out results_pooled/_smoke
    python pooled_vs_federated.py --out results_pooled/equal --agg equal
    python pooled_vs_federated.py --out results_pooled/nk    --agg nk
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import sys
import time
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("PYTHONHASHSEED", "0")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Mandatory shared import.  Everything listed here is used verbatim so that the
# single-centre manuscript and the federated manuscript cannot report numbers
# produced by two different implementations.
# ---------------------------------------------------------------------------
try:
    from single_center_cv import (  # noqa: F401
        DEFAULT_MANIFEST,
        EVAL_TRANSFORM,
        TRAIN_TRANSFORM,
        KidneyImageDataset,
        _ba_from_pred,
        aggregate_patient_scores,
        assert_patient_disjoint,
        binary_metrics,
        build_model,
        build_patient_table,
        delong_test,
        fast_auc,
        inner_split,
        load_cohort,
        make_loader,
        outer_splits,
        predict_probs,
        sample_weights,
        set_all_seeds,
        split_counts,
        youden_threshold,
    )
except ImportError as _e:  # pragma: no cover
    raise SystemExit(
        "pooled_vs_federated.py requires single_center_cv.py in the same directory.\n"
        "The splitting, transforms, model and metric code is imported from it on\n"
        "purpose, so that the single-centre and federated papers cannot drift apart.\n"
        f"Import failed with: {_e}"
    )

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

SITES: Tuple[str, str] = ("NCKUH", "CYCH")
ARMS: Tuple[str, ...] = ("local", "loso", "pooled", "federated")
ARM_LABEL = {
    "local": "ARM 1 local (train own site)",
    "loso": "ARM 2 loso (train other site)",
    "pooled": "ARM 3 pooled (centralized)",
    "federated": "ARM 4 federated (FedAvg)",
}
CONTRASTS = (("pooled", "federated"), ("loso", "federated"),
             ("local", "federated"), ("pooled", "local"))

# Metric keys as produced by single_center_cv.binary_metrics
METRIC_KEYS = ("accuracy", "tpr_sensitivity", "tnr_specificity", "balanced_accuracy",
               "auc", "average_precision", "min_tpr_tnr")
METRIC_LABEL = {
    "auc": "AUROC", "average_precision": "AP", "balanced_accuracy": "Balanced acc.",
    "tpr_sensitivity": "Sensitivity", "tnr_specificity": "Specificity",
    "accuracy": "Accuracy", "min_tpr_tnr": "min(Se,Sp)",
}

RECORDS_META: List[dict] = []


# ===========================================================================
# 1. Config
# ===========================================================================
@dataclass
class Config:
    """Attribute names deliberately mirror single_center_cv's argparse namespace
    (lr, weight_decay, amp, batch_size, workers, epochs, patience, arch,
    patient_agg, monitor) so the imported helpers accept this object directly."""
    out: Path
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    folds: int = 5
    inner_folds: int = 4
    max_folds: Optional[int] = None
    arch: str = "densenet121"
    batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 60
    patience: int = 12
    rounds: int = 60
    local_epochs: int = 1
    fed_patience: int = 20
    agg: str = "equal"
    bn_stats: str = "average"
    monitor: str = "patient_auc"
    patient_agg: str = "max_kidney"
    workers: int = 0
    amp: bool = True
    n_boot: int = 2000
    boot_seed: int = 20260909
    smoke: bool = False
    smoke_patients: int = 24
    figures: bool = True
    strict_determinism: bool = False
    keep_first_exam_only: bool = True
    manifest: Path = DEFAULT_MANIFEST
    partitions_dir: Path = PROJECT_ROOT / "partitions"
    reuse_partitions: bool = True
    device: str = "auto"

    @property
    def agg_col(self) -> str:
        """Primary patient score column, per protocol section 2.2 (mean over the
        views of each kidney, then MAX over the two kidneys)."""
        return "score_mean" if self.patient_agg == "mean" else "score_maxkidney"


# ===========================================================================
# 2. Cohort for BOTH sites
# ===========================================================================
def load_both_sites(cfg: Config) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], dict]:
    """Call the shared single-site loader once per site and assert that patient
    identifiers do not collide, because the pooled and federated arms concatenate
    the two sites and the imported helpers group by `patient_id` alone."""
    img, pat, notes = {}, {}, {}
    for s in SITES:
        d, n = load_cohort(cfg.manifest, s, cfg.keep_first_exam_only)
        d = d.copy()
        d["site"] = s
        img[s] = d
        pat[s] = build_patient_table(d)
        pat[s]["site"] = s
        notes[s] = n
    a, b = set(pat[SITES[0]]["patient_id"]), set(pat[SITES[1]]["patient_id"])
    overlap = a & b
    if overlap:
        raise AssertionError(
            f"patient_id collides across sites ({len(overlap)} ids, e.g. "
            f"{sorted(overlap)[:5]}); pooled aggregation would silently merge "
            "two different children. Namespace the ids before continuing.")
    notes["cross_site_patient_id_collisions"] = 0
    return img, pat, notes


def input_fingerprint(img: Dict[str, pd.DataFrame]) -> str:
    h = hashlib.sha256()
    for s in SITES:
        for p in sorted(img[s]["abs_path"]):
            try:
                sz = os.path.getsize(p)
            except OSError:
                sz = -1
            h.update(os.path.basename(p).encode("utf-8"))
            h.update(str(sz).encode("utf-8"))
    return h.hexdigest()[:16]


# ===========================================================================
# 3. Frozen partitions
# ===========================================================================
def build_assignments(pat: pd.DataFrame, site: str, seed: int, cfg: Config) -> pd.DataFrame:
    """One row per (patient, outer_fold) with role in {train, val, test}.

    Fold construction and inner-split seeding replicate single_center_cv's main
    loop exactly: outer_splits(pat, folds, seed), then inner_split(pat_tr_all,
    inner_folds, seed) inside every outer fold.
    """
    rows = []
    splits = outer_splits(pat, cfg.folds, seed)
    for k, (tr_idx, te_idx) in enumerate(splits):
        pat_tr_all = pat.iloc[tr_idx].reset_index(drop=True)
        itr, iva = inner_split(pat_tr_all, cfg.inner_folds, seed)
        val_pids = set(pat_tr_all.iloc[iva]["patient_id"])
        test_pids = set(pat.iloc[te_idx]["patient_id"])
        for _, r in pat.iterrows():
            pid = r["patient_id"]
            role = "test" if pid in test_pids else ("val" if pid in val_pids else "train")
            rows.append({"site": site, "seed": seed, "outer_fold": k, "patient_id": pid,
                         "label": int(r["label"]), "n_images": int(r["n_images"]), "role": role})
    return pd.DataFrame(rows)


def get_or_freeze_assignments(pat: pd.DataFrame, site: str, seed: int, cfg: Config) -> pd.DataFrame:
    cfg.partitions_dir.mkdir(parents=True, exist_ok=True)
    fp = cfg.partitions_dir / f"{site}_seed{seed}_assignments.csv"
    if cfg.reuse_partitions and fp.is_file():
        got = pd.read_csv(fp, dtype={"patient_id": str, "site": str})
        ref = pat[["patient_id", "label"]].sort_values("patient_id").reset_index(drop=True)
        cur = (got.drop_duplicates("patient_id")[["patient_id", "label"]]
               .sort_values("patient_id").reset_index(drop=True))
        if cur.equals(ref) and sorted(got["outer_fold"].unique()) == list(range(cfg.folds)):
            print(f"[partitions] reusing frozen folds {fp.name} ({len(cur)} patients)")
            # Reuse is keyed on patient ids + labels ONLY, because the folds must
            # stay fixed even if the image set changes.  But that means a file
            # frozen under a different --keep-all-exams / manifest still matches,
            # so its cached n_images can be stale.  Say so loudly; the reported
            # image counts are recomputed from the live cohort in main().
            live = pat.set_index("patient_id")["n_images"]
            cached = got.drop_duplicates("patient_id").set_index("patient_id")["n_images"]
            if not cached.reindex(live.index).astype("Int64").equals(live.astype("Int64")):
                print(f"[partitions] WARNING: {fp.name} was frozen under a different "
                      "image set (n_images differs). The patient-level folds are still "
                      "valid and are reused; image counts are recomputed from the "
                      "current cohort.")
            return got
        print(f"[partitions] {fp.name} does not match the current cohort - regenerating")
    a = build_assignments(pat, site, seed, cfg)
    a.to_csv(fp, index=False, encoding="utf-8")
    print(f"[partitions] froze folds -> {fp}")
    return a


def assert_fold_integrity(assign: pd.DataFrame, site: str, seed: int, cfg: Config) -> None:
    """Patient-level integrity of the frozen folds.

    Raised explicitly rather than with the `assert` statement: `python -O`
    strips `assert`, and these checks (plus the ARM 3/4 ones in run_fold) are
    the entire evidence for the manuscript's no-leakage claim.  A flag that
    silently deletes the leakage guard must not exist.
    """
    for k in range(cfg.folds):
        sub = assign[assign["outer_fold"] == k]
        tr = sub.loc[sub["role"] == "train", "patient_id"]
        va = sub.loc[sub["role"] == "val", "patient_id"]
        te = sub.loc[sub["role"] == "test", "patient_id"]
        assert_patient_disjoint(tr, va, te, f"{site} seed{seed} fold{k}")
        if not (len(tr) and len(va) and len(te)):
            raise AssertionError(
                f"{site} seed{seed} fold{k}: empty split "
                f"(train {len(tr)}, val {len(va)}, test {len(te)})")
    n_test = assign[assign["role"] == "test"].groupby("patient_id").size()
    if not (n_test == 1).all():
        raise AssertionError(
            f"{site} seed{seed}: a patient is tested in more than one fold "
            f"(e.g. {sorted(n_test[n_test != 1].index)[:5]}); the out-of-fold "
            "predictions would double-count that patient inside one AUC.")


# ===========================================================================
# 4. FedAvg aggregation  (the piece fedavg_corrected.py got wrong)
# ===========================================================================
def _is_integer_buffer(name: str, t: torch.Tensor) -> bool:
    return (not t.is_floating_point()) or name.endswith("num_batches_tracked")


def _is_bn_running_stat(name: str) -> bool:
    return name.endswith("running_mean") or name.endswith("running_var")


def federated_aggregate(
    states: Sequence[Dict[str, torch.Tensor]],
    weights: Sequence[float],
    bn_stats: str = "average",
    prev_global: Optional[Dict[str, torch.Tensor]] = None,
) -> "OrderedDict[str, torch.Tensor]":
    """Weighted FedAvg over client state_dicts, with BatchNorm handled explicitly.

    Three deliberate departures from `fedavg_corrected.federated_average`:

    * **Real weights.**  `--agg equal` gives 1/K (what the manuscript did);
      `--agg nk` gives n_k / sum(n_k) over training images, per McMahan et al.

    * **`num_batches_tracked` is a counter, not a parameter.**  The original code
      averaged it; with int64 tensors that is both meaningless and truncating.
      Here every non-floating buffer is combined by `max` across clients, which
      keeps monotone integer semantics.  (These BatchNorm layers use
      momentum=0.1, so the counter never enters the running-statistic update
      arithmetic; taking the max simply keeps a valid integer.)

    * **BN running statistics are buffers estimated from each client's own data
      distribution**, not gradients.  `bn_stats='average'` (default) averages
      them with the same client weights, which is what makes one global model
      usable at both hospitals.  `bn_stats='keep_global'` instead leaves them at
      the previous global model's values, isolating the effect of transporting
      each site's activation statistics.  The choice is recorded in the manifest.

    Accumulation is done in float64 on CPU so that repeated rounds do not
    accumulate float16/float32 rounding drift.
    """
    if not states:
        raise ValueError("no client states to aggregate")
    w = np.asarray(weights, dtype=np.float64)
    if not np.isfinite(w).all() or w.sum() <= 0:
        raise ValueError(f"invalid aggregation weights: {weights}")
    w = w / w.sum()

    ref = states[0]
    out: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for key, ref_t in ref.items():
        tensors = [s[key] for s in states]
        if any(t.shape != ref_t.shape for t in tensors):
            raise ValueError(f"shape mismatch across clients for {key}")

        if _is_integer_buffer(key, ref_t):
            out[key] = torch.stack([t.detach().cpu() for t in tensors]).max(dim=0).values.to(ref_t.dtype)
            continue

        if _is_bn_running_stat(key) and bn_stats == "keep_global":
            if prev_global is None:
                raise ValueError("bn_stats='keep_global' requires prev_global")
            out[key] = prev_global[key].detach().clone()
            continue

        acc = torch.zeros(ref_t.shape, dtype=torch.float64)
        for t, wi in zip(tensors, w):
            acc += float(wi) * t.detach().to(device="cpu", dtype=torch.float64)
        out[key] = acc.to(ref_t.dtype)
    return out


# ===========================================================================
# 5. Training
# ===========================================================================
def _cpu_state(model) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


# Every degenerate operating point met during the run, so that a fold which
# silently produced a trivial classifier can never be averaged into a published
# balanced accuracy without appearing in the manifest and on stderr.
DEGENERATE_THRESHOLDS: List[dict] = []


def _checked_youden(y: np.ndarray, s: np.ndarray, tag: str) -> Tuple[float, dict]:
    """`single_center_cv.youden_threshold` with the degenerate cases made loud.

    Two silent failure modes are being caught here, both of which fired in the
    delivered smoke run and both of which produce a *plausible-looking* number
    rather than a nan:

    * **Single-class validation set.**  youden_threshold returns a hard-coded
      0.5 whenever `len(unique(y)) < 2`.  That 0.5 is not an estimate of
      anything, but it flows straight into `patient_pred` and into the reported
      sensitivity/specificity.

    * **The trivial operating point.**  sklearn's `roc_curve` prepends an `inf`
      threshold representing "call everything negative", where Youden's J = 0.
      When no real operating point achieves J > 0 - i.e. the validation ROC is
      at or below chance - `argmax(J)` selects that first point, and
      youden_threshold maps `inf` to 1.0.  Thresholding at 1.0 predicts every
      case negative, giving sensitivity 0, specificity 1 and balanced accuracy
      exactly 0.5.  Observed in `results_pooled/_smoke`: arm `pooled`, site
      NCKUH, fold 0 got tau = 1.000000 while the largest probability in the
      fold was 0.661, so every prediction was negative.

    The returned threshold is left exactly as single_center_cv computes it (this
    script must not fork the shared thresholding rule); what changes is that the
    condition is *reported*.  This function only describes the threshold - it does
    NOT register it.  Registration happens in `_taus_for`, i.e. once per threshold
    that is actually used, because `_validation_S` runs every epoch and most of
    those epochs are discarded by early stopping; registering here would fill the
    manifest with degeneracies belonging to models that were thrown away.
    """
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    try:
        thr = float(youden_threshold(y, s, tag=tag))   # tag arg is newer than this call
    except TypeError:
        thr = float(youden_threshold(y, s))
    reason = ""
    if len(np.unique(y)) < 2:
        reason = "single_class_validation(tau forced to 0.5)"
    elif len(s) and thr > np.max(s):
        reason = "trivial_all_negative(tau above every score)"
    elif len(s) and thr <= np.min(s):
        reason = "trivial_all_positive(tau at or below every score)"
    info = {"reason": reason, "tau": thr, "n": int(len(y)), "n_pos": int(y.sum()),
            "score_min": float(np.min(s)) if len(s) else float("nan"),
            "score_max": float(np.max(s)) if len(s) else float("nan")}
    return thr, info


def _validation_S(model, val_img: Dict[str, pd.DataFrame], cfg: Config, device
                  ) -> Tuple[float, Dict[str, dict]]:
    """Selection criterion S, computed on VALIDATION patients only.

    S = min over participating sites of patient-level AUROC (default) or of
    patient-level balanced accuracy at that site's own Youden threshold.
    The `min` is deliberate: a global model that is only good at the larger
    hospital is not a useful federated model.

    Two thresholds are returned per site, both fitted on validation only:
    `tau` on the aggregated PATIENT scores and `tau_img` on the raw IMAGE
    probabilities.  They are not interchangeable - averaging a patient's image
    probabilities shrinks the score distribution, so a patient-level Youden
    point sits at a different quantile of the image distribution.  This mirrors
    single_center_cv, which likewise fits cnn_patient_youden and cnn_image_youden
    separately.
    """
    per = {}
    for s, fr in val_img.items():
        if fr is None or len(fr) == 0:
            continue
        p = predict_probs(model, fr, device, cfg.batch_size, cfg.workers, cfg.amp)
        vp = aggregate_patient_scores(fr, p)
        y = vp["label"].to_numpy()
        sc = vp[cfg.agg_col].to_numpy()
        thr, thr_info = _checked_youden(y, sc, f"{s}/patient")
        y_img = fr["label"].to_numpy()
        thr_img, thr_img_info = _checked_youden(y_img, p, f"{s}/image")
        per[s] = {"auc": fast_auc(y, sc),
                  "ba": _ba_from_pred(y, (sc >= thr).astype(int)),
                  "tau": float(thr),
                  "tau_degenerate": thr_info["reason"],
                  "tau_info": thr_info,
                  "auc_img": fast_auc(y_img, p),
                  "tau_img": float(thr_img),
                  "tau_img_degenerate": thr_img_info["reason"],
                  "tau_img_info": thr_img_info,
                  "n_val_pat": int(len(y)), "n_val_pat_pos": int(y.sum()),
                  "n_val_img": int(len(y_img)), "n_val_img_pos": int(y_img.sum())}
    if not per:
        return float("-inf"), {}
    key = "ba" if cfg.monitor == "patient_ba" else "auc"
    vals = [v[key] for v in per.values() if np.isfinite(v[key])]
    return (float(min(vals)) if vals else float("-inf")), per


def train_centralized(train_img: pd.DataFrame, val_img: Dict[str, pd.DataFrame],
                      cfg: Config, device, seed: int, tag: str) -> Tuple[torch.nn.Module, dict]:
    """Arms 1/2 (single site) and arm 3 (pooled) share this loop.

    It mirrors `single_center_cv.train_cnn` exactly - same model builder, same
    weighted cross-entropy, same optimiser, same AMP, same early-stopping
    semantics - and generalises only the monitored quantity to a MINIMUM over
    several validation sites, so that the pooled arm and the federated arm are
    selected by an identical criterion.  With a single site the criterion
    reduces exactly to train_cnn's.
    """
    model = build_model(cfg.arch).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
    w = sample_weights(train_img)
    ds = KidneyImageDataset(train_img["abs_path"].tolist(), train_img["label"].tolist(),
                            w, TRAIN_TRANSFORM)

    best = {"S": -np.inf, "epoch": -1, "state": None, "per_site": {}}
    history = []
    bar = tqdm(range(cfg.epochs), desc=f"{tag}", leave=False, ncols=88, ascii=True)
    for ep in bar:
        model.train()
        loader = make_loader(ds, cfg.batch_size, True, cfg.workers, seed * 1000 + ep)
        tot, nb = 0.0, 0
        for imgs, labels, wts, _ in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            wts = wts.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=cfg.amp and device.type == "cuda"):
                out = model(imgs)
                per = F.cross_entropy(out, labels, reduction="none")
                loss = (per * wts).sum() / wts.sum().clamp_min(1e-8)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += float(loss.detach()) * len(labels)
            nb += len(labels)

        S, per_site = _validation_S(model, val_img, cfg, device)
        history.append({"epoch": ep + 1, "train_loss": tot / max(1, nb), "S": S,
                        **{f"val_auc_{s}": v["auc"] for s, v in per_site.items()}})
        bar.set_postfix(loss=f"{tot/max(1,nb):.3f}", S=f"{S:.3f}")
        if S > best["S"] + 1e-6:
            best = {"S": float(S), "epoch": ep + 1, "state": _cpu_state(model),
                    "per_site": per_site}
        elif ep + 1 - best["epoch"] >= cfg.patience:
            break

    if best["state"] is None:
        # Never reached with a well-formed validation split; reachable when every
        # epoch's S was non-finite (e.g. a single-class validation set at every
        # participating site).  Silently returning the last epoch's weights would
        # publish an unselected model whose reported best_S is -inf.
        raise RuntimeError(
            f"{tag}: the selection criterion S was non-finite at every one of "
            f"{len(history)} epochs, so no model was ever selected. This means at "
            f"least one validation split has a single class. Inspect the fold "
            f"before trusting anything downstream.")
    model.load_state_dict(best["state"])
    return model, {"best_epoch": best["epoch"], "best_S": best["S"],
                   "epochs_run": len(history), "history": history,
                   "val_per_site": best["per_site"]}


def train_federated(train_img: Dict[str, pd.DataFrame], val_img: Dict[str, pd.DataFrame],
                    cfg: Config, device, seed: int) -> Tuple[torch.nn.Module, dict]:
    """FedAvg over the two sites' TRAINING patients only.

    Per round the global model is broadcast to both clients, each client runs
    `--local-epochs` epoch(s) of exactly the same weighted cross-entropy used by
    the centralized arms, and the client states are aggregated.  The round is
    then scored on the VALIDATION patients of both sites; no test patient is
    ever involved, which is the specific defect this replaces.
    """
    sites = [s for s in SITES if len(train_img.get(s, [])) > 0]
    global_model = build_model(cfg.arch).to(device)
    global_state = _cpu_state(global_model)

    n_k = np.array([len(train_img[s]) for s in sites], dtype=float)
    if cfg.agg == "equal":
        wts = [1.0 / len(sites)] * len(sites)
    elif cfg.agg == "nk":
        wts = (n_k / n_k.sum()).tolist()
    else:
        raise ValueError(f"unknown --agg {cfg.agg}")

    # per-client fixed training assets
    ds = {}
    for s in sites:
        w = sample_weights(train_img[s])
        ds[s] = KidneyImageDataset(train_img[s]["abs_path"].tolist(),
                                   train_img[s]["label"].tolist(), w, TRAIN_TRANSFORM)

    client = build_model(cfg.arch).to(device)   # reused shell, weights overwritten each round
    best = {"S": -np.inf, "round": -1, "state": None, "per_site": {}}
    history = []
    bar = tqdm(range(cfg.rounds), desc="fedavg rounds", leave=False, ncols=88, ascii=True)
    for rnd in bar:
        client_states = []
        for s in sites:
            client.load_state_dict(global_state)
            opt = torch.optim.Adam(client.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
            scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
            client.train()
            for le in range(cfg.local_epochs):
                # rnd * 10 + le collided for --local-epochs > 10
                # (rnd=0,le=10 == rnd=1,le=0), repeating shuffle orders across
                # rounds.  1000 leaves room for both.
                loader = make_loader(ds[s], cfg.batch_size, True, cfg.workers,
                                     seed * 1_000_000 + rnd * 1000 + le)
                for imgs, labels, wts_b, _ in loader:
                    imgs = imgs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    wts_b = wts_b.to(device, non_blocking=True)
                    opt.zero_grad(set_to_none=True)
                    with torch.amp.autocast("cuda", enabled=cfg.amp and device.type == "cuda"):
                        out = client(imgs)
                        per = F.cross_entropy(out, labels, reduction="none")
                        loss = (per * wts_b).sum() / wts_b.sum().clamp_min(1e-8)
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
            client_states.append(_cpu_state(client))

        global_state = federated_aggregate(client_states, wts, cfg.bn_stats, global_state)
        global_model.load_state_dict(global_state)

        S, per_site = _validation_S(global_model, val_img, cfg, device)
        history.append({"round": rnd + 1, "S": S,
                        **{f"val_auc_{s}": v["auc"] for s, v in per_site.items()}})
        bar.set_postfix(S=f"{S:.3f}")
        if S > best["S"] + 1e-6:
            best = {"S": float(S), "round": rnd + 1,
                    "state": copy.deepcopy(global_state), "per_site": per_site}
        elif rnd + 1 - best["round"] >= cfg.fed_patience:
            break

    if best["state"] is None:
        raise RuntimeError(
            f"fedavg: the selection criterion S was non-finite at every one of "
            f"{len(history)} rounds, so no round was ever selected. At least one "
            f"site's validation split has a single class.")
    global_model.load_state_dict(best["state"])
    del client
    return global_model, {"best_round": best["round"], "best_S": best["S"],
                          "rounds_run": len(history), "agg": cfg.agg,
                          "agg_weights": wts, "n_train_images": n_k.tolist(),
                          "bn_stats": cfg.bn_stats, "history": history,
                          "val_per_site": best["per_site"]}


def _taus_for(info: dict, site: str, tag: str) -> Tuple[float, float]:
    """Thresholds for `site` from a training run's selected-epoch validation record.

    The previous form was `info["val_per_site"].get(site, {}).get("tau", 0.5)`.
    That default is unreachable in a healthy run - and precisely for that reason
    it must not be a default: if the site really is absent (an empty validation
    frame, or a selected epoch whose per-site record was never populated), 0.5 is
    a fabricated operating point that would be published as though it had been
    fitted on validation data.  Fail instead.
    """
    vps = info.get("val_per_site", {}).get(site)
    if not vps or "tau" not in vps or "tau_img" not in vps:
        raise RuntimeError(
            f"{tag}: no validation threshold was recorded for site {site} at the "
            f"selected epoch/round (val_per_site keys = "
            f"{sorted(info.get('val_per_site', {}))}). Refusing to substitute 0.5.")
    # Register degeneracy for the SELECTED epoch/round only - these are the two
    # thresholds that will actually be applied to the test fold.
    for which, key in (("patient", "tau_info"), ("image", "tau_img_info")):
        nfo = vps.get(key) or {}
        if nfo.get("reason"):
            rec = {"where": f"{tag} [{which}]", "tau": nfo["tau"],
                   "reason": nfo["reason"], "n": nfo["n"], "n_pos": nfo["n_pos"],
                   "score_min": nfo["score_min"], "score_max": nfo["score_max"]}
            DEGENERATE_THRESHOLDS.append(rec)
            print(f"    [degenerate threshold] {rec['where']}: tau={rec['tau']:.6f} "
                  f"({rec['reason']}); validation scores span "
                  f"[{rec['score_min']:.4f}, {rec['score_max']:.4f}], n={rec['n']} "
                  f"({rec['n_pos']} positive). The threshold-dependent metrics of "
                  f"this fold are NOT interpretable; AUROC and AP are unaffected.")
    return float(vps["tau"]), float(vps["tau_img"])


def _tau_flag(info: dict, site: str) -> str:
    """Degeneracy suffix for the `tau_source` column of the prediction CSVs, so a
    trivial operating point is visible per row and not only in the manifest."""
    vps = info.get("val_per_site", {}).get(site, {})
    bits = [b for b in (vps.get("tau_degenerate", ""), vps.get("tau_img_degenerate", "")) if b]
    return ("!DEGENERATE:" + "+".join(sorted(set(bits)))) if bits else ""


# ===========================================================================
# 6. One (seed, outer fold): all four arms on the same test folds
# ===========================================================================
def run_fold(img: Dict[str, pd.DataFrame], assign: Dict[str, pd.DataFrame],
             seed: int, fold: int, cfg: Config, device) -> List[pd.DataFrame]:
    roles: Dict[str, Dict[str, set]] = {}
    fr: Dict[str, Dict[str, pd.DataFrame]] = {}
    for s in SITES:
        a = assign[s]
        a = a[a["outer_fold"] == fold]
        roles[s] = {r: set(a.loc[a["role"] == r, "patient_id"]) for r in ("train", "val", "test")}
        d = img[s]
        fr[s] = {r: d[d["patient_id"].isin(roles[s][r])].reset_index(drop=True)
                 for r in ("train", "val", "test")}
        assert_patient_disjoint(roles[s]["train"], roles[s]["val"], roles[s]["test"],
                                f"{s} seed{seed} fold{fold}")

    # ARM 3 / ARM 4 must never see a test patient from EITHER hospital.
    all_test = set().union(*[roles[s]["test"] for s in SITES])
    pooled_train = set().union(*[roles[s]["train"] for s in SITES])
    pooled_val = set().union(*[roles[s]["val"] for s in SITES])
    # Explicit raises, not `assert`: `python -O` strips assert statements, and
    # these three lines are the only thing standing between the pooled/federated
    # arms and the exact defect this script was written to remove.
    for _bad, _msg in (
            (pooled_train & all_test, "ARM 3/4 TRAINING set contains a test patient"),
            (pooled_val & all_test, "ARM 3/4 VALIDATION set contains a test patient"),
            (pooled_train & pooled_val, "ARM 3/4 train/val overlap")):
        if _bad:
            raise AssertionError(
                f"{_msg} at seed{seed} fold{fold}: {len(_bad)} patient(s), e.g. "
                f"{sorted(_bad)[:5]} -- aborting; every number after this point "
                "would be invalid.")

    records: List[pd.DataFrame] = []

    def emit(arm: str, model, test_site: str, tau: float, tau_img: float, tau_source: str,
             train_scope: str, meta: dict) -> None:
        rows = fr[test_site]["test"]
        if len(rows) == 0:
            return
        probs = predict_probs(model, rows, device, cfg.batch_size, cfg.workers, cfg.amp)
        pat = aggregate_patient_scores(rows, probs)
        m_max = dict(zip(pat["patient_id"], pat["score_maxkidney"]))
        m_mean = dict(zip(pat["patient_id"], pat["score_mean"]))
        rec = rows[["patient_id", "label", "view", "side", "filename"]].copy()
        rec.insert(0, "seed", seed)
        rec.insert(1, "fold", fold)
        rec.insert(2, "arm", arm)
        rec.insert(3, "test_site", test_site)
        rec["train_scope"] = train_scope
        rec["image_prob"] = probs
        rec["patient_score_maxkidney"] = rec["patient_id"].map(m_max)
        rec["patient_score_mean"] = rec["patient_id"].map(m_mean)
        rec["patient_score"] = rec["patient_id"].map(
            m_max if cfg.agg_col == "score_maxkidney" else m_mean)
        rec["tau"] = tau
        rec["tau_img"] = tau_img
        rec["tau_source"] = tau_source
        # Image predictions use the IMAGE-level validation threshold, not the
        # patient-level one: patient scores are means/maxes of image
        # probabilities and are far less dispersed, so thresholding raw image
        # probabilities at the patient tau put the image operating point at an
        # arbitrary quantile (measured on the smoke run: LOSO@CYCH image
        # sensitivity 0.06, and one pooled fold where tau=1.0 predicted every
        # image negative).
        rec["image_pred"] = (rec["image_prob"] >= tau_img).astype(int)
        rec["patient_pred"] = (rec["patient_score"] >= tau).astype(int)
        for k, v in meta.items():
            rec[f"meta_{k}"] = v
        records.append(rec)

    # -------- ARMS 1 and 2: one single-site model, scored at both hospitals --
    site_model = {}
    for s in SITES:
        set_all_seeds(seed * 100 + fold, strict=cfg.strict_determinism)
        model, info = train_centralized(fr[s]["train"], {s: fr[s]["val"]}, cfg, device,
                                        seed, tag=f"local[{s}] s{seed}f{fold}")
        tau, tau_img = _taus_for(info, s, f"local[{s}] s{seed}f{fold}")
        site_model[s] = (model, tau, tau_img, info)

    for s in SITES:
        model, tau, tau_img, info = site_model[s]
        other = SITES[1] if s == SITES[0] else SITES[0]
        meta = {"best_epoch": info["best_epoch"], "best_S": round(info["best_S"], 5)}
        emit("local", model, s, tau, tau_img, f"val[{s}]" + _tau_flag(info, s),
             f"train[{s}]", meta)
        # ARM 2: the OTHER hospital's test fold.  Both thresholds necessarily come
        # from the training site, because a model deployed at an unseen hospital
        # has no calibration data there.  AUROC and AP are threshold-free and are
        # therefore unaffected by this.
        emit("loso", model, other, tau, tau_img, f"val[{s}]" + _tau_flag(info, s),
             f"train[{s}]", meta)

    # -------- ARM 3: centralized pooled upper bound -------------------------
    set_all_seeds(seed * 100 + fold, strict=cfg.strict_determinism)
    pooled_tr = pd.concat([fr[s]["train"] for s in SITES], ignore_index=True)
    pooled_va = {s: fr[s]["val"] for s in SITES}
    pooled_model, pinfo = train_centralized(pooled_tr, pooled_va, cfg, device, seed,
                                            tag=f"pooled s{seed}f{fold}")
    for s in SITES:
        p_tau, p_tau_img = _taus_for(pinfo, s, f"pooled s{seed}f{fold} @ {s}")
        emit("pooled", pooled_model, s, p_tau, p_tau_img,
             f"val[{s}]" + _tau_flag(pinfo, s), "train[NCKUH+CYCH]",
             {"best_epoch": pinfo["best_epoch"], "best_S": round(pinfo["best_S"], 5)})

    # -------- ARM 4: FedAvg -------------------------------------------------
    set_all_seeds(seed * 100 + fold, strict=cfg.strict_determinism)
    fed_model, finfo = train_federated({s: fr[s]["train"] for s in SITES}, pooled_va,
                                       cfg, device, seed)
    for s in SITES:
        f_tau, f_tau_img = _taus_for(finfo, s, f"fedavg s{seed}f{fold} @ {s}")
        emit("federated", fed_model, s, f_tau, f_tau_img,
             f"val[{s}]" + _tau_flag(finfo, s), "fedavg[NCKUH+CYCH]",
             {"best_round": finfo["best_round"], "best_S": round(finfo["best_S"], 5),
              "agg": finfo["agg"], "bn_stats": finfo["bn_stats"]})

    RECORDS_META.append({
        "seed": seed, "fold": fold,
        "counts": {s: [split_counts(
            pd.DataFrame({"patient_id": sorted(roles[s][r]),
                          "label": [int(img[s].loc[img[s].patient_id == p, "label"].iloc[0])
                                    for p in sorted(roles[s][r])]}),
            img[s], sorted(roles[s][r]), r) for r in ("train", "val", "test")] for s in SITES},
        "local": {s: {"best_epoch": site_model[s][3]["best_epoch"],
                      "best_S": site_model[s][3]["best_S"],
                      "tau_J": site_model[s][1],
                      "tau_J_image": site_model[s][2]} for s in SITES},
        "pooled": {"best_epoch": pinfo["best_epoch"], "best_S": pinfo["best_S"]},
        "federated": {k: finfo[k] for k in ("best_round", "best_S", "rounds_run",
                                            "agg", "agg_weights", "n_train_images", "bn_stats")},
    })

    del site_model, pooled_model, fed_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return records


# ===========================================================================
# 7. Analysis: metrics, cluster bootstrap, DeLong
# ===========================================================================
def _holm(p: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni step-down adjusted p-values (valid under any dependence)."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p, kind="mergesort")
    adj = np.empty(n, dtype=float)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (n - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def patient_frame(preds: pd.DataFrame) -> pd.DataFrame:
    cols = ["seed", "fold", "arm", "test_site", "patient_id", "label",
            "patient_score", "patient_score_maxkidney", "patient_score_mean",
            "patient_pred", "tau"]
    return preds[cols].drop_duplicates(
        subset=["seed", "arm", "test_site", "patient_id"]).reset_index(drop=True)


def analyse(preds: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Point estimates (mean over seeds of the pooled out-of-fold estimate),
    stratified patient cluster-bootstrap CIs with repetition randomisation, and
    paired DeLong tests.

    Two rules from the protocol are enforced here:
      * a patient never appears twice inside a single AUC computation - seeds
        are combined by averaging AUCs and by the bootstrap's repetition draw,
        never by concatenating score vectors;
      * patients are the resampling unit at BOTH levels, so drawing a patient
        drags in all of that patient's images (four views of one child on one
        day are nearly perfectly correlated; a naive image bootstrap would be
        about half as wide as it should be).
    """
    pat = patient_frame(preds)
    seeds = sorted(preds["seed"].unique().tolist())
    rows: List[dict] = []
    boot_auc: Dict[Tuple[str, str], np.ndarray] = {}
    # patient-level point AUROC per (site, arm), so the contrast rows can report
    # the plug-in difference of exactly the numbers Table 4 prints.
    point_auc: Dict[Tuple[str, str], float] = {}

    for site in sorted(preds["test_site"].unique()):
        canon = (pat[pat.test_site == site][["patient_id", "label"]]
                 .drop_duplicates("patient_id").sort_values("patient_id").reset_index(drop=True))
        if canon["label"].nunique() < 2:
            continue
        pid_pos = {p: i for i, p in enumerate(canon["patient_id"])}
        y_pat = canon["label"].to_numpy()

        ci = (preds[preds.test_site == site][["patient_id", "filename", "label"]]
              .drop_duplicates(["patient_id", "filename"])
              .sort_values(["patient_id", "filename"]).reset_index(drop=True))
        img_pos = {(p, f): i for i, (p, f) in enumerate(zip(ci["patient_id"], ci["filename"]))}
        y_img = ci["label"].to_numpy()
        img_of_pat: Dict[int, List[int]] = {}
        for (p, _f), i in img_pos.items():
            img_of_pat.setdefault(pid_pos[p], []).append(i)
        img_of_pat = {k: np.asarray(sorted(v), dtype=int) for k, v in img_of_pat.items()}

        # ---- one set of bootstrap draws, shared by every arm at this site ----
        # NOTE: the site offset MUST come from a stable digest, not Python's
        # builtin hash().  str.__hash__ is salted per process (PYTHONHASHSEED is
        # read at interpreter start-up, so setting it from inside the process has
        # no effect), which silently made --boot-seed inert: re-running analyse()
        # on identical predictions produced different 95% CIs every time.
        site_offset = int(hashlib.sha256(site.encode("utf-8")).hexdigest()[:8], 16) % 10007
        rng = np.random.default_rng(cfg.boot_seed + site_offset)
        pos = np.flatnonzero(y_pat == 1)
        neg = np.flatnonzero(y_pat == 0)
        rep_idx = rng.integers(0, max(len(seeds), 1), size=cfg.n_boot)
        draws = [np.concatenate([rng.choice(pos, len(pos), True),
                                 rng.choice(neg, len(neg), True)]) for _ in range(cfg.n_boot)]

        aligned: Dict[Tuple[str, int], dict] = {}
        for arm in ARMS:
            for si, sd in enumerate(seeds):
                sp = pat[(pat.test_site == site) & (pat.arm == arm) & (pat.seed == sd)]
                si_ = preds[(preds.test_site == site) & (preds.arm == arm) & (preds.seed == sd)]
                if len(sp) == 0:
                    continue
                ps = np.full(len(canon), np.nan); pp = np.full(len(canon), np.nan)
                for p, s_, pr in zip(sp.patient_id, sp.patient_score, sp.patient_pred):
                    ps[pid_pos[p]] = s_; pp[pid_pos[p]] = pr
                isc = np.full(len(ci), np.nan); ipr = np.full(len(ci), np.nan)
                for p, f, s_, pr in zip(si_.patient_id, si_.filename,
                                        si_.image_prob, si_.image_pred):
                    j = img_pos.get((p, f))
                    if j is not None:
                        isc[j] = s_; ipr[j] = pr
                aligned[(arm, si)] = {"ps": ps, "pp": pp, "is": isc, "ip": ipr}

        # ---- COMMON evaluation set, so that every arm at this site is scored on
        # exactly the same patients and the same bootstrap replicates.
        # Previously each arm chose its own fallback seed and applied its OWN
        # finite-score mask inside the bootstrap, so if any arm was missing at a
        # seed the paired delta below subtracted AUCs computed on different seeds
        # and/or different patient subsets - i.e. it was not paired at all,
        # contradicting the "identical bootstrap draws for both arms" caption.
        arms_here = [a_ for a_ in ARMS if any((a_, i) in aligned for i in range(len(seeds)))]
        shared_seeds = [i for i in range(len(seeds))
                        if all((a_, i) in aligned for a_ in arms_here)]
        if not shared_seeds:                      # degenerate; fall back to per-arm
            shared_seeds = sorted({i for (a_, i) in aligned})
        # patient / image positions scored by EVERY arm at a given seed
        ok_pat = {si: np.all([np.isfinite(aligned[(a_, si)]["ps"])
                              for a_ in arms_here if (a_, si) in aligned], axis=0)
                  for si in shared_seeds}
        ok_img = {si: np.all([np.isfinite(aligned[(a_, si)]["is"])
                              for a_ in arms_here if (a_, si) in aligned], axis=0)
                  for si in shared_seeds}
        # one frozen (seed, patient index, image index) plan per replicate
        boot_plan: List[Tuple[int, np.ndarray, np.ndarray]] = []
        for b in range(cfg.n_boot):
            si = int(rep_idx[b])
            if si not in shared_seeds:
                si = shared_seeds[b % len(shared_seeds)]
            idx = draws[b][ok_pat[si][draws[b]]]
            if len(idx) < 4 or len(np.unique(y_pat[idx])) < 2:
                boot_plan.append((si, np.zeros(0, dtype=int), np.zeros(0, dtype=int)))
                continue
            parts = [img_of_pat[j] for j in idx if j in img_of_pat]
            iidx = np.concatenate(parts) if parts else np.zeros(0, dtype=int)
            iidx = iidx[ok_img[si][iidx]] if len(iidx) else iidx
            boot_plan.append((si, idx, iidx))

        for arm in ARMS:
            present = [i for i in range(len(seeds)) if (arm, i) in aligned]
            if not present:
                continue

            per_seed = {"patient": [], "image": []}
            for si in present:
                a = aligned[(arm, si)]
                ok = ok_pat.get(si, np.isfinite(a["ps"]))
                per_seed["patient"].append(binary_metrics(y_pat[ok], a["ps"][ok], a["pp"][ok]))
                ok = ok_img.get(si, np.isfinite(a["is"]))
                per_seed["image"].append(binary_metrics(y_img[ok], a["is"][ok], a["ip"][ok]))

            bp = {k: np.full(cfg.n_boot, np.nan) for k in METRIC_KEYS}
            bi = {k: np.full(cfg.n_boot, np.nan) for k in METRIC_KEYS}
            for b in range(cfg.n_boot):
                si, idx, iidx = boot_plan[b]
                if (arm, si) not in aligned or len(idx) == 0:
                    continue
                a = aligned[(arm, si)]
                mp = binary_metrics(y_pat[idx], a["ps"][idx], a["pp"][idx])
                for k in METRIC_KEYS:
                    bp[k][b] = mp[k]
                # iidx comes from boot_plan: same images for every arm at this
                # replicate (a patient drawn twice contributes its images twice).
                if len(iidx) >= 4 and len(np.unique(y_img[iidx])) >= 2:
                    mi = binary_metrics(y_img[iidx], a["is"][iidx], a["ip"][iidx])
                    for k in METRIC_KEYS:
                        bi[k][b] = mi[k]
            boot_auc[(site, arm)] = bp["auc"].copy()

            for level, boots in (("patient", bp), ("image", bi)):
                ms = per_seed[level]
                row = {"test_site": site, "arm": arm, "arm_label": ARM_LABEL[arm],
                       "level": level, "n_seeds": len(present),
                       "n": int(np.nanmean([m["n"] for m in ms])),
                       "n_pos": int(np.nanmean([m["n_pos"] for m in ms])),
                       "n_neg": int(np.nanmean([m["n_neg"] for m in ms]))}
                # A percentile interval needs enough usable replicates for its
                # 2.5th/97.5th order statistics to mean anything.  The old floor
                # of 10 would happily publish a "95% CI" whose tails were single
                # draws.  Require at least half of B, and never fewer than 200,
                # and record B_used so a thin interval is auditable.
                min_b = max(200, cfg.n_boot // 2) if cfg.n_boot >= 400 else max(
                    50, cfg.n_boot // 2)
                for k in METRIC_KEYS:
                    vals = np.array([m[k] for m in ms], dtype=float)
                    row[k] = float(np.nanmean(vals))
                    row[f"{k}_seed_sd"] = (float(np.nanstd(vals, ddof=1))
                                           if len(vals) > 1 else float("nan"))
                    v = boots[k][np.isfinite(boots[k])]
                    row[f"{k}_B_used"] = int(len(v))
                    ok_b = len(v) >= min_b
                    row[f"{k}_lo"] = float(np.percentile(v, 2.5)) if ok_b else float("nan")
                    row[f"{k}_hi"] = float(np.percentile(v, 97.5)) if ok_b else float("nan")
                    if not ok_b:
                        print(f"    [bootstrap] {site}/{arm}/{level}/{k}: only "
                              f"{len(v)}/{cfg.n_boot} usable replicates "
                              f"(need {min_b}); CI suppressed rather than reported "
                              f"from a handful of draws.")
                if level == "patient":
                    point_auc[(site, arm)] = row["auc"]
                    t = pat[(pat.test_site == site) & (pat.arm == arm)]["tau"]
                    row["tau_mean"] = float(t.mean())
                    row["tau_sd"] = float(t.std(ddof=1)) if len(t) > 1 else float("nan")
                rows.append(row)

        # ---- paired bootstrap deltas, identical draws for both arms ---------
        for a1, a2 in CONTRASTS:
            if (site, a1) not in boot_auc or (site, a2) not in boot_auc:
                continue
            d = boot_auc[(site, a1)] - boot_auc[(site, a2)]
            dv = d[np.isfinite(d)]
            if len(dv) < 10:
                continue
            # Add-one (Davison & Hinkley) correction.  Without it, a contrast
            # whose bootstrap deltas are all strictly positive - which is exactly
            # what loso-vs-federated will look like at 0.49 vs 0.98 - reports
            # "p = 0.0000", a number that cannot be produced by B resamples.
            # The reportable floor is now 2/(B+1).
            nb_ = len(dv)
            p = 2.0 * min((float(np.sum(dv <= 0)) + 1.0) / (nb_ + 1.0),
                          (float(np.sum(dv >= 0)) + 1.0) / (nb_ + 1.0))
            # Point estimate = the PLUG-IN difference of the two AUROCs printed in
            # Table 4, not the mean of the bootstrap deltas.  The bootstrap mean
            # is a resampling average and differs from the plug-in value by the
            # bootstrap bias (~0.002 on synthetic data here), so reporting it made
            # the contrast table inconsistent with the arm table it sits under: a
            # reader subtracting two Table 4 rows got a different number from the
            # one in the "Delta AUROC" column.  The bootstrap mean is kept
            # alongside so the bias is visible.
            pt = {a_: point_auc.get((site, a_), float("nan")) for a_ in (a1, a2)}
            plug_in = pt[a1] - pt[a2]
            rows.append({"test_site": site, "arm": f"delta:{a1}-{a2}",
                         "arm_label": f"delta AUROC: {a1} - {a2}", "level": "patient",
                         "auc": float(plug_in),
                         "boot_delta_mean": float(np.mean(dv)),
                         "boot_bias": float(np.mean(dv) - plug_in),
                         "auc_lo": float(np.percentile(dv, 2.5)),
                         "auc_hi": float(np.percentile(dv, 97.5)),
                         "auc_B_used": int(nb_),
                         "boot_p": min(p, 1.0), "n_seeds": len(seeds)})

    res = pd.DataFrame(rows)

    # ---- paired DeLong, patient level, per seed + conservative max-p rule ---
    # The imported single_center_cv.delong_test was cross-checked against its
    # definition before this was written: AUCs reproduce sklearn exactly under
    # heavy ties (mid-ranks), the Sun & Xu covariance reproduces a stratified
    # paired bootstrap SD to within 1% over six random problems, the statistic is
    # antisymmetric under argument swap, the p-value is two-sided, and empirical
    # type-I error over 2000 null replicates at n=120 was 0.0500 against a
    # nominal 0.05.  The implementation is sound; what needed guarding is the
    # boundary it hands back on degenerate input (see MIN_CLASS below).
    MIN_CLASS = 5
    dl: List[dict] = []
    for site in sorted(pat.test_site.unique()):
        for a1, a2 in CONTRASTS:
            ps = []
            for sd in seeds:
                s1 = pat[(pat.test_site == site) & (pat.arm == a1) & (pat.seed == sd)]
                s2 = pat[(pat.test_site == site) & (pat.arm == a2) & (pat.seed == sd)]
                if len(s1) == 0 or len(s2) == 0:
                    continue
                m = s1.merge(s2, on=["patient_id", "label"], suffixes=("_1", "_2"))
                yv = m["label"].to_numpy()
                n_pos, n_neg = int((yv == 1).sum()), int((yv == 0).sum())
                # DeLong's variance is a covariance over the per-case placement
                # values, estimated with np.cov(ddof=1).  With one positive (or
                # one negative) that covariance is 0/0: delong_test then takes the
                # `not isfinite(var)` branch and returns z = +/-inf, p = 0.0 for
                # any non-zero delta.  The old guard was `len(m) < 10`, which a
                # 10-patient fold with 1 positive passes -- so a completely
                # uninformative comparison could be published as p = 0.
                if n_pos < MIN_CLASS or n_neg < MIN_CLASS:
                    dl.append({"test_site": site, "comparison": f"{a1} vs {a2}",
                               "seed": str(sd), "n_patients": int(len(m)),
                               "auc1": np.nan, "auc2": np.nan, "delta": np.nan,
                               "se_delta": np.nan, "z": np.nan, "p": np.nan,
                               "skipped": f"too few per class ({n_pos}+/{n_neg}-); "
                                          f"DeLong variance undefined below {MIN_CLASS}"})
                    continue
                r = delong_test(yv, m["patient_score_1"].to_numpy(),
                                m["patient_score_2"].to_numpy())
                # Second guard: a zero/non-finite SE with a non-zero delta is the
                # same degeneracy arriving by another route (e.g. every placement
                # value identical).  p = 0.0 there is an artefact, not evidence.
                if (not np.isfinite(r["se_delta"])) or (
                        r["se_delta"] <= 0 and abs(r["delta"]) > 1e-12):
                    r["p"], r["z"] = np.nan, np.nan
                    r["skipped"] = ("degenerate DeLong variance (se=0 with non-zero "
                                    "delta); p suppressed instead of reported as 0")
                else:
                    r["skipped"] = ""
                r.update({"test_site": site, "comparison": f"{a1} vs {a2}",
                          "seed": str(sd), "n_patients": int(len(m))})
                dl.append(r)
                if np.isfinite(r["p"]):
                    ps.append(r["p"])
            if ps:
                dl.append({"test_site": site, "comparison": f"{a1} vs {a2}",
                           "seed": "MAX-RULE", "n_patients": np.nan, "auc1": np.nan,
                           "auc2": np.nan, "delta": np.nan, "se_delta": np.nan,
                           "z": np.nan, "p": float(np.max(ps)),
                           "skipped": "" if len(ps) == len(seeds) else
                                      f"only {len(ps)}/{len(seeds)} seeds usable"})
    dl_df = pd.DataFrame(dl)

    # ---- multiplicity ---------------------------------------------------------
    # CONTRASTS x sites is a family of 8 tests (and the bootstrap contrasts are a
    # second family of 8 on the same data).  Reporting eight unadjusted p-values
    # and quoting whichever cleared 0.05 is exactly the "detailed description of
    # the statistical tests" failure the earlier rejection (ki.txt) called out.
    # Holm is used because it is valid under arbitrary dependence, which these
    # contrasts certainly have -- they reuse the same patients and the same arms.
    if len(dl_df):
        dl_df["p_holm"] = np.nan
        mask = (dl_df["seed"] == "MAX-RULE") & dl_df["p"].notna()
        dl_df.loc[mask, "p_holm"] = _holm(dl_df.loc[mask, "p"].to_numpy())
    if len(res) and "boot_p" in res.columns:
        # `boot_p` only exists once at least one contrast survived; without this
        # guard a run where every contrast was skipped raised KeyError here.
        dmask = res["arm"].str.startswith("delta:") & res["boot_p"].notna()
        res["boot_p_holm"] = np.nan
        if dmask.any():
            res.loc[dmask, "boot_p_holm"] = _holm(res.loc[dmask, "boot_p"].to_numpy())
    return res, dl_df


# ===========================================================================
# 8. Table 4
# ===========================================================================
def fmt_ci(v, lo, hi, nd: int = 3) -> str:
    if v is None or not np.isfinite(v):
        return "-"
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return f"{v:.{nd}f}"
    return f"{v:.{nd}f} ({lo:.{nd}f}-{hi:.{nd}f})"


def build_table4(res: pd.DataFrame) -> pd.DataFrame:
    if len(res) == 0:
        return pd.DataFrame()
    base = res[~res["arm"].str.startswith("delta:")]
    pt = base[base.level == "patient"]
    im = {(r.test_site, r.arm): r for r in base[base.level == "image"].itertuples()}
    order = {a: i for i, a in enumerate(ARMS)}
    out = []
    for r in sorted(pt.itertuples(), key=lambda x: (x.test_site, order.get(x.arm, 99))):
        i = im.get((r.test_site, r.arm))
        out.append(OrderedDict([
            ("Test site", r.test_site),
            ("Arm", ARM_LABEL.get(r.arm, r.arm)),
            ("Test patients (n)", int(r.n)),
            ("Test images (n)", int(i.n) if i is not None else np.nan),
            ("Patient AUROC (95% CI)", fmt_ci(r.auc, r.auc_lo, r.auc_hi)),
            ("Patient AP (95% CI)",
             fmt_ci(r.average_precision, r.average_precision_lo, r.average_precision_hi)),
            ("Patient balanced acc. (95% CI)",
             fmt_ci(r.balanced_accuracy, r.balanced_accuracy_lo, r.balanced_accuracy_hi)),
            ("Patient sensitivity (95% CI)",
             fmt_ci(r.tpr_sensitivity, r.tpr_sensitivity_lo, r.tpr_sensitivity_hi)),
            ("Patient specificity (95% CI)",
             fmt_ci(r.tnr_specificity, r.tnr_specificity_lo, r.tnr_specificity_hi)),
            ("Patient accuracy (95% CI)", fmt_ci(r.accuracy, r.accuracy_lo, r.accuracy_hi)),
            ("Patient min(Se,Sp) (95% CI)",
             fmt_ci(r.min_tpr_tnr, r.min_tpr_tnr_lo, r.min_tpr_tnr_hi)),
            ("Image AUROC (95% CI)",
             fmt_ci(i.auc, i.auc_lo, i.auc_hi) if i is not None else "-"),
            ("Image balanced acc. (95% CI)",
             fmt_ci(i.balanced_accuracy, i.balanced_accuracy_lo, i.balanced_accuracy_hi)
             if i is not None else "-"),
            ("Patient AUROC across seeds (mean +/- SD)",
             f"{r.auc:.3f} +/- {r.auc_seed_sd:.3f}" if np.isfinite(r.auc_seed_sd)
             else f"{r.auc:.3f} +/- n/a"),
            ("tau_J (mean +/- SD)",
             f"{r.tau_mean:.3f} +/- {r.tau_sd:.3f}" if np.isfinite(r.tau_sd)
             else f"{r.tau_mean:.3f}"),
        ]))
    return pd.DataFrame(out)


def df_to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(r[c]) else str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def write_table4(res: pd.DataFrame, delong: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    t4 = build_table4(res)
    t4.to_csv(cfg.out / "table4_pooled_vs_federated.csv", index=False, encoding="utf-8-sig")

    md = [
        "# Table 4 (replacement). Patient-level performance of four training regimes",
        "",
        f"All four arms share the same frozen patient-level StratifiedGroupKFold test "
        f"folds ({cfg.folds} outer folds x {len(cfg.seeds)} seed(s)); no epoch, round, "
        f"threshold or model was selected on a test fold. Point estimate = mean over "
        f"seeds of the pooled out-of-fold estimate. 95% CI = label-stratified PATIENT "
        f"cluster bootstrap (B={cfg.n_boot}) with repetition randomisation; drawing a "
        f"patient draws all of that patient's images. Every arm at a site is scored "
        f"on the same patients and the same bootstrap replicates, so the paired "
        f"contrasts below are exactly paired. Balanced accuracy = (Se+Sp)/2, "
        f"not plain accuracy. Patient score = {cfg.agg_col} (mean over each kidney's "
        f"views, then max over kidneys), thresholded at a Youden cut-point fitted on "
        f"validation PATIENT scores; image-level predictions use a separate Youden "
        f"cut-point fitted on validation IMAGE probabilities. FedAvg client weighting "
        f"= {cfg.agg}; BatchNorm running statistics = {cfg.bn_stats}; "
        f"num_batches_tracked combined by max, never averaged.",
        "",
        df_to_markdown(t4),
        "",
    ]

    if DEGENERATE_THRESHOLDS:
        md += ["## Degenerate operating points detected (threshold-dependent rows only)",
               "",
               "A Youden cut-point fitted on a validation split can land outside the "
               "range of the scores it is applied to - most often the trivial "
               "'call everything negative' point that sklearn's roc_curve returns as "
               "an infinite threshold, which yields sensitivity 0, specificity 1 and "
               "balanced accuracy exactly 0.5. Those folds are listed below. AUROC and "
               "AP are threshold-free and are unaffected; sensitivity, specificity, "
               "accuracy, balanced accuracy and min(Se,Sp) for the affected folds are "
               "NOT interpretable and must not be quoted without this caveat.", "",
               df_to_markdown(pd.DataFrame(DEGENERATE_THRESHOLDS)), ""]

    d = res[res["arm"].str.startswith("delta:")] if len(res) else res
    if len(d):
        dd = pd.DataFrame([{
            "Test site": r.test_site,
            "Contrast (patient AUROC)": r.arm.replace("delta:", ""),
            "Delta AUROC (95% CI)": fmt_ci(r.auc, r.auc_lo, r.auc_hi),
            "Bootstrap p": (f"{r.boot_p:.4f}"
                            if np.isfinite(getattr(r, "boot_p", np.nan)) else "-"),
            "Bootstrap p (Holm)": (f"{r.boot_p_holm:.4f}"
                                   if np.isfinite(getattr(r, "boot_p_holm", np.nan))
                                   else "-"),
        } for r in d.itertuples()])
        md += ["## Paired contrasts (identical bootstrap draws for both arms)", "",
               f"Point estimate = the plug-in difference of the two AUROCs in Table 4. "
               f"CI = percentile interval of the paired bootstrap deltas. The smallest "
               f"reportable two-sided bootstrap p is 2/(B+1); a value at that floor "
               f"means 'no resample crossed zero', not 'p = 0'. Holm adjustment is over "
               f"the family of {len(d)} contrasts shown here and is valid under the "
               f"arbitrary dependence these contrasts have (same patients, same arms).",
               "", df_to_markdown(dd), ""]

    if len(delong):
        show = pd.DataFrame([{
            "Test site": r["test_site"], "Comparison": r["comparison"], "Seed": r["seed"],
            "AUC 1": "-" if not np.isfinite(r.get("auc1", np.nan)) else f"{r['auc1']:.4f}",
            "AUC 2": "-" if not np.isfinite(r.get("auc2", np.nan)) else f"{r['auc2']:.4f}",
            "Delta": "-" if not np.isfinite(r.get("delta", np.nan)) else f"{r['delta']:+.4f}",
            "DeLong p": ("-" if not np.isfinite(r.get("p", np.nan))
                         else f"{r['p']:.4g}"),
            "DeLong p (Holm)": ("-" if not np.isfinite(r.get("p_holm", np.nan))
                                else f"{r['p_holm']:.4g}"),
            "Note": str(r.get("skipped", "") or ""),
        } for _, r in delong.iterrows()])
        md += ["## Paired DeLong test (patient level, per seed)", "",
               "The MAX-RULE row is the pre-specified conservative decision rule: a "
               "difference counts as significant only if p < 0.05 at EVERY seed, which "
               "makes seed cherry-picking impossible. Holm adjustment is applied across "
               "the MAX-RULE family (one row per contrast per site).", "",
               "A blank DeLong p means the comparison was refused rather than "
               "estimated: DeLong's variance is a covariance over per-case placement "
               "values and is undefined when either class has fewer than 5 patients, "
               "where the estimator would otherwise return z = +/-inf and p = 0 for any "
               "non-zero difference. The `Note` column gives the reason.", "",
               "Caveat to state in the manuscript: these are pooled out-of-fold "
               "predictions, so different patients were scored by different fold "
               "models. DeLong's variance treats them as one independent sample and "
               "therefore does not carry the cross-validation partition variability; "
               "the bootstrap CI above, which randomises the repetition, does.", "",
               df_to_markdown(show), ""]

    md += [
        "## How to read this table",
        "",
        "* **ARM 2 (loso) is the only row that measures generalisation to an unseen "
        "hospital.** Every other arm was trained on patients from the hospital it is "
        "scored on. With only two participating sites, nothing here estimates "
        "performance at a third, never-seen hospital.",
        "* **ARM 3 versus ARM 4 is the only contrast attributable to the federated "
        "algorithm.** Both use exactly the same training patients and differ only in "
        "whether the data were pooled centrally or combined by FedAvg. If ARM 4 is "
        "close to ARM 3, the defensible claim is near-lossless privacy preservation, "
        "not that FedAvg outperforms centralised training.",
        "* The ARM 2 -> ARM 4 improvement mixes two causes - access to the other "
        "hospital's data, and the aggregation algorithm - and must not be attributed "
        "to federation alone. This is the co-author's MAJOR 4.",
        "* ARM 2 thresholds come from the training site's validation patients, because "
        "a model deployed at an unseen hospital has no calibration data there. AUROC "
        "and AP are threshold-free and so are unaffected by this.",
        "* Image-level columns are descriptive only. Labels are patient-level and acute "
        "pyelonephritis is frequently unilateral, so images of a normal contralateral "
        "kidney carry a positive label; image-level discrimination is bounded by that "
        "label noise rather than by model capacity, and no image-level number may be "
        "read as per-kidney accuracy.",
        "",
    ]
    (cfg.out / "table4_pooled_vs_federated.md").write_text("\n".join(md), encoding="utf-8")
    return t4


# ===========================================================================
# 9. Figures (600 dpi)
# ===========================================================================
def make_figures(preds: pd.DataFrame, res: pd.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    pat = patient_frame(preds)
    sites = sorted(pat.test_site.unique())
    colors = {"local": "#1f77b4", "loso": "#d62728", "pooled": "#2ca02c", "federated": "#9467bd"}

    fig, axes = plt.subplots(1, max(len(sites), 1), figsize=(5.4 * max(len(sites), 1), 5.1))
    for ax, site in zip(np.atleast_1d(axes), sites):
        sd0 = sorted(pat.seed.unique())[0]
        for arm in ARMS:
            sub = pat[(pat.test_site == site) & (pat.arm == arm) & (pat.seed == sd0)]
            if len(sub) == 0 or sub["label"].nunique() < 2:
                continue
            fpr, tpr, _ = roc_curve(sub["label"], sub["patient_score"])
            a = fast_auc(sub["label"].to_numpy(), sub["patient_score"].to_numpy())
            ax.plot(fpr, tpr, color=colors[arm], lw=1.8, label=f"{arm} (AUC {a:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_xlabel("1 - specificity"); ax.set_ylabel("Sensitivity")
        ax.set_title(f"{site} test folds\n(patient level, seed {sd0})", fontsize=10)
        ax.legend(loc="lower right", fontsize=8); ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(figdir / "roc_patient_level.png", dpi=600)
    fig.savefig(figdir / "roc_patient_level.pdf")
    plt.close(fig)

    pt = res[(res.level == "patient") & (~res.arm.str.startswith("delta:"))]
    if len(pt):
        rr = sorted(pt.itertuples(),
                    key=lambda x: (x.test_site, ARMS.index(x.arm) if x.arm in ARMS else 99))
        fig, ax = plt.subplots(figsize=(7.6, 0.52 * len(rr) + 2.0))
        for i, r in enumerate(rr):
            ax.plot([r.auc_lo, r.auc_hi], [i, i], color=colors.get(r.arm, "k"), lw=2.2)
            ax.plot([r.auc], [i], "o", color=colors.get(r.arm, "k"), ms=5.5)
        ax.axvline(0.5, color="k", ls="--", lw=0.8)
        ax.set_yticks(range(len(rr)))
        ax.set_yticklabels([f"{r.test_site} / {r.arm}" for r in rr], fontsize=8)
        ax.set_xlabel("Patient-level AUROC (95% patient cluster bootstrap CI)")
        ax.set_xlim(0.25, 1.02); ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(figdir / "forest_patient_auroc.png", dpi=600)
        fig.savefig(figdir / "forest_patient_auroc.pdf")
        plt.close(fig)


# ===========================================================================
# 10. CLI
# ===========================================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Local / LOSO / pooled-centralized / FedAvg on shared patient-level "
                    "test folds - the comparison the co-author's MAJOR 1 and MAJOR 4 ask for.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--out", default="results_pooled/main")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--inner-folds", type=int, default=4)
    p.add_argument("--max-folds", type=int, default=None,
                   help="run only the first N outer folds (debugging)")
    p.add_argument("--arch", default="densenet121",
                   choices=["densenet121", "mobilenet_v2", "efficientnet_b0", "resnet50"])
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=60, help="max epochs for arms 1-3")
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--rounds", type=int, default=60, help="max FedAvg rounds")
    p.add_argument("--local-epochs", type=int, default=1,
                   help="client epochs per federated round")
    p.add_argument("--fed-patience", type=int, default=20)
    p.add_argument("--agg", choices=["equal", "nk"], default="equal",
                   help="FedAvg client weighting: equal (1/K, what the manuscript did) or "
                        "nk (training-image count, McMahan et al.) - co-author MINOR 2")
    p.add_argument("--bn-stats", choices=["average", "keep_global"], default="average",
                   help="how BN running_mean/var are combined; the integer "
                        "num_batches_tracked counter is never averaged (max is used)")
    p.add_argument("--monitor", choices=["patient_auc", "patient_ba"], default="patient_auc",
                   help="quantity minimised over sites to form the selection criterion S, "
                        "computed on VALIDATION patients only")
    p.add_argument("--patient-agg", choices=["max_kidney", "mean"], default="max_kidney",
                   help="patient score rule; max_kidney = protocol section 2.2 (mean over "
                        "each kidney's views, then max over kidneys)")
    p.add_argument("--workers", type=int, default=0,
                   help="KEEP AT 0 on this Windows machine: single_center_cv.make_loader "
                        "rebuilds the DataLoader every epoch, so worker processes are "
                        "respawned each time and >0 workers measured 5-10x SLOWER. "
                        "See COST OF A FULL RUN in the module docstring for the "
                        "measured per-epoch times at 0 workers.")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--boot-seed", type=int, default=20260909)
    p.add_argument("--keep-all-exams", action="store_true",
                   help="sensitivity analysis: keep both exams of the two repeat-exam patients")
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    p.add_argument("--partitions-dir", default=str(PROJECT_ROOT / "partitions"))
    p.add_argument("--no-reuse-partitions", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--strict-determinism", action="store_true")
    p.add_argument("--smoke", action="store_true",
                   help="tiny end-to-end run: 1 seed, 2 folds, 24 patients/site, "
                        "mobilenet_v2, 2 epochs, 2 rounds")
    p.add_argument("--smoke-patients", type=int, default=24)
    return p


def cfg_from_args(a: argparse.Namespace) -> Config:
    cfg = Config(
        out=Path(a.out), seeds=list(a.seeds), folds=a.folds, inner_folds=a.inner_folds,
        max_folds=a.max_folds, arch=a.arch, batch_size=a.batch_size, lr=a.lr,
        weight_decay=a.weight_decay, epochs=a.epochs, patience=a.patience, rounds=a.rounds,
        local_epochs=a.local_epochs, fed_patience=a.fed_patience, agg=a.agg,
        bn_stats=a.bn_stats, monitor=a.monitor, patient_agg=a.patient_agg,
        workers=a.workers, amp=not a.no_amp, n_boot=a.bootstrap, boot_seed=a.boot_seed,
        smoke=a.smoke, smoke_patients=a.smoke_patients, figures=not a.no_figures,
        strict_determinism=a.strict_determinism, keep_first_exam_only=not a.keep_all_exams,
        manifest=Path(a.manifest), partitions_dir=Path(a.partitions_dir),
        reuse_partitions=not a.no_reuse_partitions, device=a.device,
    )
    if cfg.smoke:
        cfg.seeds = cfg.seeds[:1]
        cfg.max_folds = 2
        cfg.arch = "mobilenet_v2"
        cfg.batch_size = 16
        cfg.epochs = 2
        cfg.patience = 1
        cfg.rounds = 2
        cfg.fed_patience = 2
        cfg.n_boot = min(cfg.n_boot, 200)
        # never write smoke folds into the real partitions directory
        cfg.partitions_dir = cfg.out / "partitions"
    return cfg


def subsample_for_smoke(img: Dict[str, pd.DataFrame], pat: Dict[str, pd.DataFrame],
                        n_per_site: int, seed: int):
    """Subsample by PATIENT, never by image, so grouping semantics survive."""
    rng = np.random.default_rng(seed)
    for s in SITES:
        keep = []
        for lab in (0, 1):
            ids = pat[s].loc[pat[s]["label"] == lab, "patient_id"].to_numpy()
            keep.extend(rng.choice(ids, min(len(ids), max(n_per_site // 2, 8)),
                                   replace=False).tolist())
        keep = set(keep)
        img[s] = img[s][img[s]["patient_id"].isin(keep)].reset_index(drop=True)
        img[s]["img_idx"] = np.arange(len(img[s]))
        pat[s] = pat[s][pat[s]["patient_id"].isin(keep)].reset_index(drop=True)
    return img, pat


def divergence_report(cfg: Config) -> List[str]:
    """Every place this script's settings differ from single_center_cv.py's defaults.

    Printed at startup and stored in the run manifest.  These are choices, not
    accidents, but they must be visible: if the single-centre paper and the
    federated paper quote numbers produced under different selection rules, a
    reviewer will find it.  Resolve them deliberately before submission.
    """
    # These MUST be kept in sync with single_center_cv.build_parser().  Every
    # hyperparameter that reaches the shared training loop belongs here: an entry
    # missing from this dict is a divergence that never gets reported, which is
    # strictly worse than no report at all, because the manifest then asserts a
    # sameness that does not hold.
    scv_defaults = {"monitor": "patient_ba", "patient_agg": "mean",
                    "batch_size": 16, "patience": 12, "folds": 5, "arch": "densenet121",
                    "weight_decay": 0.0, "lr": 1e-4, "epochs": 60, "inner_folds": 4}
    notes = []
    if cfg.weight_decay != scv_defaults["weight_decay"]:
        notes.append(
            f"weight_decay = {cfg.weight_decay} (single_center_cv default: "
            f"{scv_defaults['weight_decay']}). This is a TRAINING difference, not a "
            "reporting one: the optimiser here applies L2 regularisation that the "
            "single-centre paper's CNN does not, so the two papers' 'identical shared "
            "code' claim does not hold for the arms trained by this script. Either "
            "pass --weight-decay 0.0 here or change single_center_cv's default; do "
            "not leave them different.")
    if cfg.monitor != scv_defaults["monitor"]:
        notes.append(
            f"monitor = {cfg.monitor} (single_center_cv default: "
            f"{scv_defaults['monitor']}). The federated round-selection criterion S "
            "must be threshold-free, otherwise round choice is entangled with the "
            "Youden threshold and the pooled/federated contrast is confounded. "
            "Pass --monitor patient_ba to match the single-centre paper exactly.")
    if cfg.patient_agg != scv_defaults["patient_agg"]:
        notes.append(
            f"patient_agg = {cfg.patient_agg} (single_center_cv default: "
            f"{scv_defaults['patient_agg']}). max_kidney is the PRE-SPECIFIED primary "
            "rule in protocol section 2.2; single_center_cv currently defaults to mean. "
            "Both scores are written to the prediction CSVs, so either can be "
            "recomputed without retraining, but the two papers should agree.")
    for k in ("batch_size", "patience", "folds", "arch", "lr", "epochs", "inner_folds"):
        if getattr(cfg, k) != scv_defaults[k]:
            notes.append(f"{k} = {getattr(cfg, k)} (single_center_cv default: "
                         f"{scv_defaults[k]})")
    if not notes:
        notes.append("no divergence from single_center_cv.py defaults")
    return notes


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return _finite(float(o))
    if isinstance(o, (np.ndarray,)):
        return _sanitise_json(o.tolist())
    return str(o)


def _finite(x):
    """NaN/inf -> None.  json.dumps happily writes bare NaN/Infinity tokens, which
    are not valid JSON: R's jsonlite and most strict parsers reject the manifest.
    best_S is -inf whenever a run degenerates, so this is reachable in practice."""
    return None if (isinstance(x, float) and not np.isfinite(x)) else x


def _sanitise_json(o):
    if isinstance(o, dict):
        return {k: _sanitise_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitise_json(v) for v in o]
    if isinstance(o, float):
        return _finite(o)
    if isinstance(o, (np.floating,)):
        return _finite(float(o))
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


def main(argv: Optional[List[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    cfg = cfg_from_args(a)
    cfg.out.mkdir(parents=True, exist_ok=True)
    (cfg.out / "predictions").mkdir(exist_ok=True)
    t0 = time.time()

    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if cfg.device == "auto" else cfg.device)

    print("=" * 78)
    print("pooled_vs_federated.py  -  centralized upper bound vs FedAvg")
    print("=" * 78)
    print(f"device={device}  arch={cfg.arch}  agg={cfg.agg}  bn_stats={cfg.bn_stats}  "
          f"monitor={cfg.monitor}  patient_agg={cfg.patient_agg}")
    print(f"seeds={cfg.seeds}  folds={cfg.folds}  inner_folds={cfg.inner_folds}  "
          f"smoke={cfg.smoke}")
    print("[shared] splitting / transforms / model / metrics imported from single_center_cv.py")
    for line in divergence_report(cfg):
        print(f"[diverges] {line}")

    img, pat, notes = load_both_sites(cfg)
    if cfg.smoke:
        img, pat = subsample_for_smoke(img, pat, cfg.smoke_patients, cfg.seeds[0])
        print(f"[smoke] subsampled to "
              f"{ {s: len(pat[s]) for s in SITES} } patients / "
              f"{ {s: len(img[s]) for s in SITES} } images")

    print("\n[cohort]")
    for s in SITES:
        p, d = pat[s], img[s]
        print(f"  {s:6s} {len(p):4d} patients ({int((p.label == 1).sum())} pos / "
              f"{int((p.label == 0).sum())} neg)   {len(d):5d} images "
              f"({int((d.label == 1).sum())} pos / {int((d.label == 0).sum())} neg)")

    folds_to_run = list(range(cfg.folds if cfg.max_folds is None
                              else min(cfg.max_folds, cfg.folds)))
    split_rows, all_preds = [], []

    # Image counts for split_counts.csv are recomputed from the cohort that is
    # actually being trained on, NOT from the n_images column of the frozen
    # partition file.  A partition file is reused whenever the patient ids and
    # labels match, which they still do after --keep-all-exams changes the image
    # set, so trusting its cached n_images would misreport the very train/val/test
    # image table the co-author's MAJOR 2 asks for.
    n_img_by_pid = {s: img[s].groupby("patient_id").size() for s in SITES}

    def _n_images(site: str, pids) -> int:
        return int(n_img_by_pid[site].reindex(list(pids)).fillna(0).sum())

    for seed in cfg.seeds:
        set_all_seeds(seed, strict=cfg.strict_determinism)
        assign = {}
        for s in SITES:
            assign[s] = get_or_freeze_assignments(pat[s], s, seed, cfg)
            assert_fold_integrity(assign[s], s, seed, cfg)
            aa = assign[s]
            for k in range(cfg.folds):
                sub = aa[aa.outer_fold == k]
                for role in ("train", "val", "test"):
                    r = sub[sub.role == role]
                    split_rows.append({
                        "site": s, "seed": seed, "outer_fold": k, "role": role,
                        "patients": int(len(r)),
                        "patients_pos": int((r.label == 1).sum()),
                        "patients_neg": int((r.label == 0).sum()),
                        "patient_prevalence": round(float((r.label == 1).mean()), 4) if len(r) else np.nan,
                        "images": _n_images(s, r.patient_id),
                        "images_cached_in_partition_file": int(r.n_images.sum()),
                    })

        for fold in folds_to_run:
            print(f"\n--- seed {seed} / outer fold {fold} " + "-" * 42)
            for s in SITES:
                sub = assign[s][assign[s].outer_fold == fold]
                for role in ("train", "val", "test"):
                    r = sub[sub.role == role]
                    print(f"    {s:6s} {role:5s} {len(r):3d} patients "
                          f"({int((r.label == 1).sum())}+/{int((r.label == 0).sum())}-) "
                          f"{_n_images(s, r.patient_id):4d} images")
            all_preds.extend(run_fold(img, assign, seed, fold, cfg, device))

    if not all_preds:
        print("!! no predictions produced")
        return 1

    preds = pd.concat(all_preds, ignore_index=True)
    for arm in preds.arm.unique():
        for site in preds.test_site.unique():
            for sd in preds.seed.unique():
                sub = preds[(preds.arm == arm) & (preds.test_site == site) & (preds.seed == sd)]
                if len(sub):
                    sub.to_csv(cfg.out / "predictions" / f"{arm}_{site}_seed{sd}.csv",
                               index=False, encoding="utf-8")
    preds.to_csv(cfg.out / "predictions_all.csv", index=False, encoding="utf-8")
    pd.DataFrame(split_rows).to_csv(cfg.out / "split_counts.csv", index=False,
                                    encoding="utf-8-sig")

    print("\n[analysis] metrics, patient cluster bootstrap, DeLong ...")
    res, delong = analyse(preds, cfg)
    res.to_csv(cfg.out / "metrics_long.csv", index=False, encoding="utf-8-sig")
    if len(delong):
        delong.to_csv(cfg.out / "stats_delong.csv", index=False, encoding="utf-8-sig")
    t4 = write_table4(res, delong, cfg)

    if cfg.figures:
        try:
            make_figures(preds, res, cfg.out)
        except Exception as e:  # noqa: BLE001
            print(f"[figures] skipped: {type(e).__name__}: {e}")

    manifest = {
        "script": "pooled_vs_federated.py",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "wall_clock_sec": round(time.time() - t0, 1),
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in cfg.__dict__.items()},
        "patient_score_column": cfg.agg_col,
        "shared_code": "single_center_cv.py (hard import: splitting, transforms, model, "
                       "sample weights, metrics, Youden threshold, patient aggregation, DeLong)",
        "partition_sharing": ("this script freezes and reuses partitions/{site}_seed{r}_"
                              "assignments.csv; single_center_cv.py does NOT read or write "
                              "that directory - it re-derives folds from the same "
                              "outer_splits/inner_split calls, so the two agree by code "
                              "equivalence, not by a shared file"),
        "image_threshold": ("image-level predictions use a separate Youden threshold fitted "
                            "on validation IMAGE probabilities (tau_img); patient-level "
                            "predictions use tau fitted on validation PATIENT scores"),
        "degenerate_thresholds": DEGENERATE_THRESHOLDS,
        "n_degenerate_thresholds": len(DEGENERATE_THRESHOLDS),
        "divergences_from_single_center_cv": divergence_report(cfg),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
        "cohort_notes": notes,
        "input_fingerprint": input_fingerprint(img),
        "n_images": {s: int(len(img[s])) for s in SITES},
        "n_patients": {s: int(len(pat[s])) for s in SITES},
        "folds_run": folds_to_run,
        "fold_records": RECORDS_META,
    }
    (cfg.out / "run_manifest.json").write_text(
        json.dumps(_sanitise_json(manifest), indent=2, ensure_ascii=False,
                   default=_json_default, allow_nan=False),
        encoding="utf-8")

    print("\n" + "=" * 78)
    print("TABLE 4 (replacement)")
    print("=" * 78)
    if len(t4):
        with pd.option_context("display.width", 250, "display.max_columns", 40):
            print(t4[["Test site", "Arm", "Test patients (n)", "Patient AUROC (95% CI)",
                      "Patient balanced acc. (95% CI)",
                      "Image AUROC (95% CI)"]].to_string(index=False))
    else:
        print("(empty)")

    d = res[res["arm"].str.startswith("delta:")] if len(res) else res
    if len(d):
        print("\nPaired contrasts (patient AUROC, identical bootstrap draws):")
        for r in d.itertuples():
            print(f"  {r.test_site:6s} {r.arm.replace('delta:',''):20s} "
                  f"{fmt_ci(r.auc, r.auc_lo, r.auc_hi)}  boot p = {r.boot_p:.4f}")
    if len(delong):
        mx = delong[delong.seed == "MAX-RULE"]
        if len(mx):
            print("\nDeLong max-p rule (significant only if p < 0.05 at EVERY seed):")
            for _, r in mx.iterrows():
                ph = r.get("p_holm", np.nan)
                print(f"  {r['test_site']:6s} {r['comparison']:24s} "
                      f"max p = {r['p']:.4g}"
                      + (f"   Holm-adjusted = {ph:.4g}" if np.isfinite(ph) else ""))

    if DEGENERATE_THRESHOLDS:
        print(f"\n!! {len(DEGENERATE_THRESHOLDS)} degenerate operating point(s) detected "
              f"(trivial or single-class Youden thresholds). AUROC/AP are unaffected; "
              f"the threshold-dependent metrics of those folds are not interpretable. "
              f"See run_manifest.json -> degenerate_thresholds.")
        for _r in DEGENERATE_THRESHOLDS[:10]:
            print(f"   {_r['where']}: tau={_r['tau']:.6f} ({_r['reason']})")

    print(f"\nOutputs -> {cfg.out.resolve()}")
    print(f"Wall clock: {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
