#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
single_center_cv.py
===================

Single-center, patient-level, nested cross-validated diagnostic-accuracy pipeline for
ultrasound-based detection of kidney inflammation (acute pyelonephritis) in children.

Implements the pre-specified single-center protocol for ONE hospital at a time:

    python single_center_cv.py --site CYCH  --out results_single_center/CYCH
    python single_center_cv.py --site NCKUH --out results_single_center/NCKUH
    python single_center_cv.py --site CYCH  --smoke --out results_single_center/_smoke

WHAT THIS SCRIPT GUARANTEES (and the previous federated code did not)
---------------------------------------------------------------------
1.  A real train / validation / test separation exists, it is PATIENT-LEVEL, and the
    absence of patient overlap is asserted at run time (the run aborts if violated).
2.  Nothing is ever selected on the test fold.  The stopping epoch, every decision
    threshold and the fusion-layer coefficients are chosen on an inner validation split
    of the training patients.  The test fold is scored exactly once, at the end of the
    fold, with those frozen choices.
3.  Every arm (clinical-only, image-only CNN, fused, PCA+SVM) is trained on the *same*
    inner-training patients and tuned on the *same* inner-validation patients, so the
    head-to-head comparisons are paired and fair.
4.  Uncertainty is reported: patient-level cluster bootstrap 95% CIs, mean +/- SD across
    seeds (partition stability), and a paired DeLong test implemented from scratch
    (statsmodels is not installed in this environment).
5.  Out-of-fold predictions are written to CSV at both image and patient level, so every
    downstream table, figure and statistical test can be recomputed without retraining.

UNIT OF ANALYSIS
----------------
Labels in this dataset are PER PATIENT and patient-constant; acute pyelonephritis is
frequently unilateral, so a positive patient contributes images of a possibly normal
contralateral kidney.  Training is therefore image-level (with the patient label
propagated, a known and quantified noise source), while every headline number is
PATIENT-LEVEL.  Image-level metrics are computed and saved, but they are descriptive
only and must never be presented as per-kidney accuracy.

Author: generated for the kidney-inflammation single-center manuscript.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import random
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------------
# Determinism must be requested before torch initialises CUDA-side state.
# ----------------------------------------------------------------------------------
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from torchvision import models, transforms  # noqa: E402
from PIL import Image  # noqa: E402

from scipy import stats as sp_stats  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm is installed in this environment
    def tqdm(x, **kwargs):  # type: ignore
        return x

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = PROJECT_ROOT / "patient_characteristics" / "used_ultrasound_image_filenames.csv"

CLINICAL_FEATURES = ["age_years", "gender_code", "renal_measure_cm"]
ARMS = ["clinical", "cnn", "fused", "svm"]
ARM_LABELS = {
    "clinical": "Clinical-only (LR: age, sex, renal measurement)",
    "cnn": "Image-only CNN",
    "fused": "Fused (CNN probability + clinical)",
    "svm": "Conventional (PCA + RBF-SVM on CNN features)",
}
IMAGE_ARMS = ["cnn", "svm"]  # clinical/fused are patient-level constructs by design


# ==================================================================================
# 1.  Reproducibility
# ==================================================================================
def set_all_seeds(seed: int, strict: bool = False) -> None:
    """Seed python, numpy and torch; make cuDNN deterministic."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if strict:
        # Can raise on ops without a deterministic implementation; opt-in only.
        torch.use_deterministic_algorithms(True, warn_only=True)


# ==================================================================================
# 2.  Cohort assembly
# ==================================================================================
def _parse_exam_date(raw) -> float:
    """Sortable numeric exam date.  One NCKUH patient has a malformed 9-digit date
    (chart id redacted); it is a single-exam patient so ordering is unaffected,
    but we degrade gracefully instead of crashing."""
    s = str(raw)
    try:
        return float(pd.Timestamp(pd.to_datetime(s, format="%Y%m%d")).value)
    except Exception:
        digits = "".join(ch for ch in s if ch.isdigit())[:8]
        try:
            return float(pd.Timestamp(pd.to_datetime(digits, format="%Y%m%d")).value)
        except Exception:
            return float("inf")


def load_cohort(manifest: Path, site: str, keep_first_exam_only: bool) -> Tuple[pd.DataFrame, dict]:
    """Load the image manifest for one site and return (image_df, notes)."""
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")
    df = pd.read_csv(manifest, encoding="utf-8-sig")
    missing_cols = [c for c in
                    ["site", "label", "filename", "relative_path", "patient_id", "exam_date_raw",
                     "gender_code", "age_years", "renal_measure_cm", "view"] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Manifest is missing required columns: {missing_cols}")

    df = df[df["site"].astype(str).str.upper() == site.upper()].copy()
    if df.empty:
        raise ValueError(f"No rows for site={site} in {manifest}")

    df["patient_id"] = df["patient_id"].astype(str)
    df["label"] = df["label"].astype(int)
    df["_exam_sort"] = df["exam_date_raw"].map(_parse_exam_date)

    notes = {
        "images_before_exam_filter": int(len(df)),
        "patients": int(df["patient_id"].nunique()),
        "exams_before_exam_filter": int(df.groupby(["patient_id", "exam_date_raw"]).ngroups),
    }

    # --- repeat-exam rule -----------------------------------------------------
    repeat = (df.groupby("patient_id")["exam_date_raw"].nunique() > 1)
    repeat_pids = sorted(repeat[repeat].index.tolist())
    notes["repeat_exam_patients"] = repeat_pids
    if keep_first_exam_only and repeat_pids:
        first_exam = df.groupby("patient_id")["_exam_sort"].transform("min")
        dropped = df[df["_exam_sort"] > first_exam]
        notes["dropped_second_exam_images"] = int(len(dropped))
        notes["dropped_second_exam_files"] = sorted(dropped["filename"].tolist())
        df = df[df["_exam_sort"] <= first_exam].copy()
    else:
        notes["dropped_second_exam_images"] = 0
        notes["dropped_second_exam_files"] = []

    # --- label must be patient-constant --------------------------------------
    conflicts = df.groupby("patient_id")["label"].nunique()
    bad = conflicts[conflicts > 1]
    if len(bad):
        raise AssertionError(f"Patient label conflicts at {site}: {bad.index.tolist()}")

    # --- resolve image paths --------------------------------------------------
    df["abs_path"] = df["relative_path"].map(lambda p: str((PROJECT_ROOT / str(p)).resolve()))
    missing = [p for p in df["abs_path"] if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} image files listed in the manifest do not exist, e.g. {missing[:3]}")

    # --- kidney side, for the protocol's secondary max-over-kidneys aggregation --
    df["side"] = df["view"].astype(str).str.upper().str[0].map({"L": "L", "R": "R"}).fillna("U")

    df = df.sort_values(["patient_id", "filename"]).reset_index(drop=True)
    df["img_idx"] = np.arange(len(df))

    notes["images_after_exam_filter"] = int(len(df))
    notes["exams_after_exam_filter"] = int(df.groupby(["patient_id", "exam_date_raw"]).ngroups)
    notes["images_pos"] = int((df["label"] == 1).sum())
    notes["images_neg"] = int((df["label"] == 0).sum())
    return df, notes


def build_patient_table(img_df: pd.DataFrame) -> pd.DataFrame:
    """One row per patient: label + clinical covariates taken from the index (earliest) exam."""
    rows = []
    for pid, g in img_df.groupby("patient_id", sort=True):
        g = g.sort_values("_exam_sort")
        idx_exam = g[g["_exam_sort"] == g["_exam_sort"].min()]
        rows.append({
            "patient_id": pid,
            "label": int(g["label"].iloc[0]),
            "age_years": float(idx_exam["age_years"].iloc[0]),
            "gender_code": float(idx_exam["gender_code"].iloc[0]),
            "renal_measure_cm": float(idx_exam["renal_measure_cm"].iloc[0]),
            "n_images": int(len(g)),
            "n_exams": int(g["exam_date_raw"].nunique()),
        })
    pat = pd.DataFrame(rows).sort_values("patient_id").reset_index(drop=True)
    return pat


# ==================================================================================
# 3.  Splitting  (patient-level, asserted disjoint)
# ==================================================================================
def outer_splits(pat: pd.DataFrame, n_folds: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """StratifiedGroupKFold over the PATIENT table (groups = patient_id, one row per
    patient, y = patient label).  Because each group is a single patient, group
    disjointness is structural and stratification is exact at the patient level."""
    y = pat["label"].to_numpy()
    groups = pat["patient_id"].to_numpy()
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(sgkf.split(np.zeros(len(pat)), y, groups))


def inner_split(pat_train: pd.DataFrame, n_inner: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Carve ONE patient-level inner validation split (1/n_inner of the training
    patients) out of the outer-training patients.  Used only for early stopping,
    threshold selection and fitting the fusion layer."""
    y = pat_train["label"].to_numpy()
    groups = pat_train["patient_id"].to_numpy()
    n_inner = max(2, min(n_inner, int(min(np.bincount(y)))))
    sgkf = StratifiedGroupKFold(n_splits=n_inner, shuffle=True, random_state=seed + 1000)
    tr_idx, va_idx = next(iter(sgkf.split(np.zeros(len(pat_train)), y, groups)))
    return tr_idx, va_idx


def _require(condition: bool, message: str) -> None:
    """Guard that survives `python -O`.  Every integrity check in this file uses this
    instead of a bare `assert`, because `assert` statements are compiled out under -O and
    a silently disabled leakage/alignment guard is worse than no guard at all."""
    if not condition:
        raise AssertionError(message)


def assert_patient_disjoint(train_pids, val_pids, test_pids, tag: str) -> None:
    """Fail loudly on any patient overlap between the three splits."""
    s_tr, s_va, s_te = set(train_pids), set(val_pids), set(test_pids)
    problems = []
    if s_tr & s_va:
        problems.append(f"train n val = {sorted(s_tr & s_va)[:10]}")
    if s_tr & s_te:
        problems.append(f"train n test = {sorted(s_tr & s_te)[:10]}")
    if s_va & s_te:
        problems.append(f"val n test = {sorted(s_va & s_te)[:10]}")
    if problems:
        raise AssertionError(
            f"PATIENT LEAKAGE DETECTED in {tag}: " + "; ".join(problems) +
            "  -- aborting: every number produced after this point would be invalid.")


def split_counts(pat: pd.DataFrame, img_df: pd.DataFrame,
                 pids: Sequence[str], name: str) -> dict:
    p = pat[pat["patient_id"].isin(set(pids))]
    i = img_df[img_df["patient_id"].isin(set(pids))]
    return {
        "split": name,
        "patients": int(len(p)),
        "patients_pos": int((p["label"] == 1).sum()),
        "patients_neg": int((p["label"] == 0).sum()),
        "patient_prevalence": round(float((p["label"] == 1).mean()) if len(p) else float("nan"), 4),
        "images": int(len(i)),
        "images_pos": int((i["label"] == 1).sum()),
        "images_neg": int((i["label"] == 0).sum()),
        "image_prevalence": round(float((i["label"] == 1).mean()) if len(i) else float("nan"), 4),
    }


# ==================================================================================
# 4.  Dataset / transforms / model
# ==================================================================================
# Transforms are copied verbatim from fedavg_corrected.py so that single-center and
# federated numbers are directly comparable.  (The pre-specified protocol asks for a
# milder augmentation set; see the DEVIATIONS note in the run summary.)
TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class KidneyImageDataset(Dataset):
    """Images addressed by absolute path, with a per-sample training weight."""

    def __init__(self, paths: Sequence[str], labels: Sequence[int],
                 weights: Optional[Sequence[float]] = None, transform=None):
        self.paths = list(paths)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.weights = (np.ones(len(self.paths), dtype=np.float32)
                        if weights is None else np.asarray(weights, dtype=np.float32))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        img = Image.open(self.paths[i]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[i]), float(self.weights[i]), i


ARCH_HEADS = {
    "densenet121": "classifier",
    "mobilenet_v2": "classifier",
    "efficientnet_b0": "classifier",
    "resnet50": "fc",
}


def build_model(arch: str) -> nn.Module:
    """ImageNet-pretrained backbone with a fresh 2-output head."""
    if arch not in ARCH_HEADS:
        raise ValueError(f"--arch must be one of {sorted(ARCH_HEADS)}")
    if arch == "densenet121":
        m = models.densenet121(weights="IMAGENET1K_V1")
        in_f = m.classifier.in_features
        m.classifier = nn.Linear(in_f, 2)
    elif arch == "mobilenet_v2":
        m = models.mobilenet_v2(weights="IMAGENET1K_V1")
        in_f = m.classifier[-1].in_features
        m.classifier = nn.Linear(in_f, 2)   # dropout dropped so head->Identity gives features
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights="IMAGENET1K_V1")
        in_f = m.classifier[-1].in_features
        m.classifier = nn.Linear(in_f, 2)
    else:  # resnet50
        m = models.resnet50(weights="IMAGENET1K_V1")
        in_f = m.fc.in_features
        m.fc = nn.Linear(in_f, 2)
    return m


def make_loader(ds: Dataset, batch_size: int, shuffle: bool, workers: int, seed: int,
                persistent: bool = False) -> DataLoader:
    """Build a DataLoader.

    `persistent` keeps the worker processes alive between epochs and is only meaningful
    when the SAME loader object is iterated more than once (see train_cnn).  Windows has
    no fork(), so every worker is a fresh `spawn`ed interpreter that re-imports this
    module -- and therefore torch, torchvision and sklearn.  That costs seconds per
    worker, per DataLoader construction.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    kw = {}
    if workers > 0:
        kw["persistent_workers"] = bool(persistent)
        kw["prefetch_factor"] = 4
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), generator=g if shuffle else None,
                      drop_last=False, **kw)


def sample_weights(train_img: pd.DataFrame) -> np.ndarray:
    """w_i = (1 / n_images(patient_i)) * (N_pat / (2 * N_pat_of_class(y_i))).

    First factor: every patient contributes equally regardless of how many views were
    acquired (patients hold 1-4 images; incomplete studies would otherwise be
    under-weighted).  Second factor: class balance at the PATIENT level, which is the
    level the model is evaluated at."""
    n_img_per_pat = train_img.groupby("patient_id")["img_idx"].transform("size").to_numpy()
    pat = train_img.drop_duplicates("patient_id")[["patient_id", "label"]]
    n_pat = len(pat)
    n_pat_cls = pat["label"].value_counts().to_dict()
    cls_w = train_img["label"].map(
        lambda y: n_pat / (2.0 * max(1, n_pat_cls.get(int(y), 1)))).to_numpy()
    w = (1.0 / n_img_per_pat) * cls_w
    return (w / w.mean()).astype(np.float32)


@torch.no_grad()
def predict_probs(model: nn.Module, img_rows: pd.DataFrame, device, batch_size: int,
                  workers: int, amp: bool) -> np.ndarray:
    """P(inflammation) per image, in the row order of img_rows."""
    model.eval()
    ds = KidneyImageDataset(img_rows["abs_path"].tolist(), img_rows["label"].tolist(),
                            None, EVAL_TRANSFORM)
    # workers is deliberately forced to 0 for evaluation passes.  These are short
    # single-shot loops (a few hundred images) called 5x per fold; on Windows the
    # process-spawn cost of even one worker exceeds the whole pass.  EVAL_TRANSFORM is
    # deterministic and shuffle is off, so this changes no number.
    loader = make_loader(ds, batch_size, False, 0, 0)
    out = np.zeros(len(ds), dtype=np.float64)
    pos = 0
    for imgs, _, _, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
            logits = model(imgs)
        p = torch.softmax(logits.float(), dim=1)[:, 1].detach().cpu().numpy()
        out[pos:pos + len(p)] = p
        pos += len(p)
    return out


@torch.no_grad()
def extract_features(model: nn.Module, arch: str, img_rows: pd.DataFrame, device,
                     batch_size: int, workers: int, amp: bool) -> np.ndarray:
    """Penultimate-layer embedding per image (head temporarily replaced by Identity)."""
    head_name = ARCH_HEADS[arch]
    head = getattr(model, head_name)
    setattr(model, head_name, nn.Identity())
    try:
        model.eval()
        ds = KidneyImageDataset(img_rows["abs_path"].tolist(), img_rows["label"].tolist(),
                                None, EVAL_TRANSFORM)
        loader = make_loader(ds, batch_size, False, 0, 0)   # see predict_probs: workers=0
        feats = []
        for imgs, _, _, _ in loader:
            imgs = imgs.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
                f = model(imgs)
            feats.append(f.float().detach().cpu().numpy())
        return np.concatenate(feats, axis=0) if feats else np.zeros((0, 1))
    finally:
        setattr(model, head_name, head)


# ==================================================================================
# 5.  Metrics
# ==================================================================================
def _midrank(x: np.ndarray) -> np.ndarray:
    """Mid-ranks with ties, returned in the original order (used by AUC and DeLong)."""
    J = np.argsort(x, kind="mergesort")
    Z = np.asarray(x, dtype=float)[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def fast_auc(y: np.ndarray, s: np.ndarray) -> float:
    """Rank-based AUROC (ties handled by mid-ranks).  Identical to sklearn but faster
    inside the bootstrap loop."""
    y = np.asarray(y).astype(int)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    s = np.asarray(s, dtype=float)
    # Tripwire: a NaN score would be sorted to the end by _midrank and silently produce a
    # finite but meaningless AUC.  A NaN here means a missing out-of-fold prediction.
    if not np.isfinite(s).all():
        raise ValueError("fast_auc: non-finite score(s) -- a missing/NaN out-of-fold "
                         "prediction would silently corrupt this AUC.")
    r = _midrank(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def binary_metrics(y: np.ndarray, score: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(int)
    tp = int(np.sum((y == 1) & (pred == 1)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    tpr = tp / (tp + fn) if (tp + fn) else float("nan")
    tnr = tn / (tn + fp) if (tn + fp) else float("nan")
    # PPV is UNDEFINED when nothing is predicted positive (0/0).  Reporting it as 0.0
    # makes an undefined quantity look like a measured failure and drags the across-seed
    # mean PPV downwards; NaN lets that seed drop out and be counted as dropped.
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    if np.isnan(prec) or np.isnan(tpr):
        f1 = 0.0 if (tp + fn) else float("nan")   # no positive called -> F1 is 0 by convention
    elif (prec + tpr) == 0:
        f1 = 0.0
    else:
        f1 = 2 * prec * tpr / (prec + tpr)
    try:
        auc = float(roc_auc_score(y, score))
        ap = float(average_precision_score(y, score))
    except Exception:
        auc, ap = float("nan"), float("nan")
    return {
        "n": int(len(y)), "n_pos": int(y.sum()), "n_neg": int(len(y) - y.sum()),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": (tp + tn) / len(y) if len(y) else float("nan"),
        "tpr_sensitivity": tpr,
        "tnr_specificity": tnr,
        "balanced_accuracy": (tpr + tnr) / 2.0,          # (TPR+TNR)/2, NOT plain accuracy
        "precision_ppv": prec,
        "npv": tn / (tn + fn) if (tn + fn) else float("nan"),
        "f1": f1,
        "auc": auc,
        "average_precision": ap,
        "min_tpr_tnr": min(tpr, tnr) if not (np.isnan(tpr) or np.isnan(tnr)) else float("nan"),
    }


def youden_threshold(y: np.ndarray, s: np.ndarray, tag: str = "") -> float:
    """Threshold maximising Youden's J on the validation scores."""
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        # A single-class selection set cannot define an operating point.  Returning 0.5
        # silently would inject an arbitrary threshold into a published confusion matrix.
        warnings.warn(
            f"youden_threshold{(' [' + tag + ']') if tag else ''}: the selection set has a "
            "single class, so no operating point exists; falling back to 0.5.  Confusion-"
            "matrix metrics derived from this threshold are NOT interpretable.",
            RuntimeWarning, stacklevel=2)
        return 0.5
    fpr, tpr, thr = roc_curve(y, s)
    thr = np.where(np.isinf(thr), 1.0, thr)
    j = tpr - fpr
    return float(thr[int(np.argmax(j))])


def sensitivity_threshold(y: np.ndarray, s: np.ndarray, target: float = 0.90,
                          tag: str = "") -> float:
    """Highest threshold whose validation sensitivity is still >= target (rule-out point)."""
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        warnings.warn(
            f"sensitivity_threshold{(' [' + tag + ']') if tag else ''}: the selection set has "
            "a single class, so the sensitivity>=target point does not exist; falling back "
            "to 0.5.  Rule-out metrics from this fold are NOT interpretable.",
            RuntimeWarning, stacklevel=2)
        return 0.5
    fpr, tpr, thr = roc_curve(y, s)
    thr = np.where(np.isinf(thr), 1.0, thr)
    ok = tpr >= target
    if not ok.any():
        return float(np.min(thr))
    return float(np.max(thr[ok]))


def cv_oof_threshold(X: np.ndarray, y: np.ndarray, seed: int, n_splits: int = 5,
                     C: float = 1.0) -> Tuple[float, float]:
    """Threshold for a cheap (logistic) model chosen from out-of-fold predictions inside
    the data the model was fitted on -- avoids an in-sample threshold for the fused arm,
    whose fitting set (the inner-validation patients) is also its only honest data.

    `C` MUST match the regularisation of the model the threshold will be applied to.
    A mismatch puts the threshold on a different probability scale than the deployed
    model, so the reported sensitivity/specificity would belong to no fitted model."""
    y = np.asarray(y).astype(int)
    n_splits = max(2, min(n_splits, int(np.bincount(y).min())))
    oof = np.full(len(y), np.nan)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        if len(np.unique(y[tr])) < 2:
            continue
        m = LogisticRegression(max_iter=2000, C=C)
        sc = StandardScaler().fit(X[tr])
        m.fit(sc.transform(X[tr]), y[tr])
        oof[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    ok = ~np.isnan(oof)
    if ok.sum() < 4 or len(np.unique(y[ok])) < 2:
        return 0.5, 0.5
    return youden_threshold(y[ok], oof[ok]), sensitivity_threshold(y[ok], oof[ok], 0.90)


# ==================================================================================
# 6.  DeLong's test (Sun & Xu 2014, fast implementation; statsmodels unavailable)
# ==================================================================================
def _fast_delong(preds_sorted: np.ndarray, m: int) -> Tuple[np.ndarray, np.ndarray]:
    """preds_sorted: (k, N) score matrix with the m positives first.  Returns
    (aucs, covariance matrix) following Sun & Xu, IEEE SPL 21(11):1389-1393, 2014."""
    n = preds_sorted.shape[1] - m
    k = preds_sorted.shape[0]
    positive = preds_sorted[:, :m]
    negative = preds_sorted[:, m:]
    tx = np.empty((k, m)); ty = np.empty((k, n)); tz = np.empty((k, m + n))
    for r in range(k):
        tx[r] = _midrank(positive[r])
        ty[r] = _midrank(negative[r])
        tz[r] = _midrank(preds_sorted[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    sx = np.atleast_2d(sx); sy = np.atleast_2d(sy)
    cov = sx / m + sy / n
    return aucs, cov


def delong_test(y: np.ndarray, s1: np.ndarray, s2: np.ndarray) -> Dict[str, float]:
    """Two-sided paired DeLong test for AUC(s1) - AUC(s2) on the SAME observations.
    Only ever called at the patient level: DeLong assumes independent observations,
    which images from one patient are not.

    Returns p = NaN (with degenerate_variance = True) when the covariance estimate is
    singular.  It NEVER returns p = 0: a variance that does not exist cannot support a
    significance claim."""
    y = np.asarray(y).astype(int)
    s1 = np.asarray(s1, dtype=float)
    s2 = np.asarray(s2, dtype=float)
    if len(np.unique(y)) < 2:
        return {"auc1": float("nan"), "auc2": float("nan"), "delta": float("nan"),
                "z": float("nan"), "p": float("nan"), "se_delta": float("nan"),
                "degenerate_variance": True}
    order = (-y).argsort(kind="mergesort")
    m = int(y.sum())
    preds = np.vstack((s1[order], s2[order]))
    aucs, cov = _fast_delong(preds, m)
    L = np.array([1.0, -1.0])
    var = float(L @ cov @ L)
    delta = float(aucs[0] - aucs[1])
    if not np.isfinite(var) or var <= 0:
        # A non-positive DeLong variance means the covariance estimate is DEGENERATE --
        # typically one arm separates this sample perfectly, so its placement values are
        # constant within class.  There is then no standard error, hence no p-value.
        # Returning 0.0 here (the previous behaviour) manufactured "p < 0.05 at every
        # seed" out of a variance that does not exist.  Return NaN instead; the caller
        # must treat a comparison with any NaN seed as UNDECIDED, not as significant.
        if abs(delta) < 1e-12:
            # Identical AUCs: the difference is exactly zero, which is a genuine result.
            return {"auc1": float(aucs[0]), "auc2": float(aucs[1]), "delta": delta,
                    "se_delta": 0.0, "z": 0.0, "p": 1.0, "degenerate_variance": False}
        return {"auc1": float(aucs[0]), "auc2": float(aucs[1]), "delta": delta,
                "se_delta": float("nan"), "z": float("nan"), "p": float("nan"),
                "degenerate_variance": True}
    se = float(np.sqrt(var))
    z = delta / se
    p = float(2.0 * sp_stats.norm.sf(abs(z)))
    return {"auc1": float(aucs[0]), "auc2": float(aucs[1]), "delta": delta,
            "se_delta": se, "z": float(z), "p": float(p), "degenerate_variance": False}


def _selftest_stats(verbose: bool = True) -> None:
    """Pin the AUC and DeLong implementations against sklearn before any real work."""
    rng = np.random.default_rng(12345)
    y = rng.integers(0, 2, 400)
    s1 = rng.normal(y * 0.9, 1.0)
    s2 = rng.normal(y * 0.5, 1.0)
    s2[:20] = s2[21:41]  # force ties
    def _check(cond: bool, msg: str) -> None:
        # NOT a bare `assert`: `python -O` strips assert statements, and these guards are
        # the only thing standing between a silent numerical error and a published number.
        if not cond:
            raise AssertionError(f"[selftest] {msg}")

    a_fast, a_sk = fast_auc(y, s1), roc_auc_score(y, s1)
    _check(abs(a_fast - a_sk) < 1e-9, f"fast_auc mismatch: {a_fast} vs {a_sk}")
    r = delong_test(y, s1, s2)
    _check(abs(r["auc1"] - roc_auc_score(y, s1)) < 1e-9, "DeLong auc1 mismatch")
    _check(abs(r["auc2"] - roc_auc_score(y, s2)) < 1e-9, "DeLong auc2 mismatch")
    r0 = delong_test(y, s1, s1.copy())
    _check(abs(r0["delta"]) < 1e-12 and r0["p"] >= 0.99, "DeLong self-comparison failed")
    # Degenerate-variance regression guard: a perfectly separating arm against an inverted
    # one has no DeLong SE.  This MUST come back as p = NaN, never as p = 0.
    yd = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    rd = delong_test(yd, np.array([.1, .2, .3, .4, .6, .7, .8, .9]),
                     np.array([.9, .8, .7, .6, .4, .3, .2, .1]))
    _check(np.isnan(rd["p"]) and rd["degenerate_variance"],
           f"degenerate DeLong variance must yield p = NaN, got p = {rd['p']}")
    # A percentile bootstrap can never resolve a p below 2/(B+1).
    _check(_bootstrap_p(np.ones(100), 100) >= 2.0 / 101.0 - 1e-12,
           "bootstrap p is not floored at its resolution limit")
    # DeLong SE must agree with a brute-force stratified bootstrap SE of the same delta.
    # This is the check that would catch a wrong covariance formula or a mishandled tie:
    # both would leave the AUCs correct while silently corrupting every p-value.
    _rg = np.random.default_rng(20260909)
    _y = _rg.integers(0, 2, 300)
    _z = _rg.normal(0, 1, 300)
    _a = 1.0 * _y + 0.7 * _z + 0.7 * _rg.normal(0, 1, 300)   # correlated arms, as in practice
    _b = 0.5 * _y + 0.7 * _z + 0.7 * _rg.normal(0, 1, 300)
    _se_delong = delong_test(_y, _a, _b)["se_delta"]
    _p1, _p0 = np.flatnonzero(_y == 1), np.flatnonzero(_y == 0)
    _br = np.random.default_rng(1)
    _d = np.empty(4000)
    for _i in range(4000):
        _tk = np.concatenate([_br.choice(_p1, _p1.size, True), _br.choice(_p0, _p0.size, True)])
        _d[_i] = fast_auc(_y[_tk], _a[_tk]) - fast_auc(_y[_tk], _b[_tk])
    _se_boot = float(_d.std(ddof=1))
    _check(abs(_se_delong / _se_boot - 1.0) < 0.10,
           f"DeLong SE {_se_delong:.4f} disagrees with the bootstrap SE {_se_boot:.4f} by "
           "more than 10%: the covariance formula or the midrank tie handling is wrong")
    # Holm must be monotone non-decreasing in sorted order and never below the raw p.
    # m = 3 after dropping the NaN: 3*0.01=0.03, 2*0.02=0.04, 1*0.60=0.60, running max.
    _h = holm_bonferroni({"a": 0.01, "b": 0.02, "c": 0.60, "d": float("nan")})
    _check(abs(_h["a"] - 0.03) < 1e-12 and abs(_h["b"] - 0.04) < 1e-12
           and abs(_h["c"] - 0.60) < 1e-12 and np.isnan(_h["d"]),
           f"Holm-Bonferroni adjustment is wrong: {_h}")
    # Monotonicity: a step-down adjustment must never decrease as raw p increases.
    _h2 = holm_bonferroni({"a": 0.04, "b": 0.041, "c": 0.9})
    _check(_h2["a"] <= _h2["b"] <= _h2["c"], f"Holm is not monotone: {_h2}")
    if verbose:
        print(f"[selftest] fast_auc == sklearn ({a_fast:.6f}); DeLong AUCs match sklearn; "
              f"identical-score p = {r0['p']:.3f}; degenerate-variance p = NaN; "
              f"bootstrap p floored at 2/(B+1); DeLong SE/bootstrap SE = "
              f"{_se_delong/_se_boot:.3f}; Holm step-down OK")


# ==================================================================================
# 7.  Cluster bootstrap
# ==================================================================================
@dataclass
class OOFStore:
    """Pooled out-of-fold predictions, aligned across seeds so a bootstrap draw can
    reuse the identical patient sample for every arm and every seed."""
    patient_ids: List[str]
    y_pat: np.ndarray
    pat_scores: Dict[int, Dict[str, np.ndarray]] = field(default_factory=dict)
    pat_preds: Dict[int, Dict[str, np.ndarray]] = field(default_factory=dict)
    img_index: List[Tuple[str, str]] = field(default_factory=list)
    y_img: np.ndarray = field(default_factory=lambda: np.zeros(0))
    img_scores: Dict[int, Dict[str, np.ndarray]] = field(default_factory=dict)
    img_preds: Dict[int, Dict[str, np.ndarray]] = field(default_factory=dict)
    img_of_patient: Dict[str, np.ndarray] = field(default_factory=dict)


def build_oof_store(pat_pred_df: pd.DataFrame, img_pred_df: pd.DataFrame,
                    seeds: Sequence[int]) -> OOFStore:
    pids = sorted(pat_pred_df["patient_id"].unique().tolist())
    pid_pos = {p: i for i, p in enumerate(pids)}
    y_pat = np.zeros(len(pids), dtype=int)
    for p, g in pat_pred_df.groupby("patient_id"):
        y_pat[pid_pos[p]] = int(g["label"].iloc[0])

    img_keys = sorted(set(zip(img_pred_df["patient_id"], img_pred_df["filename"])))
    img_pos = {k: i for i, k in enumerate(img_keys)}
    y_img = np.zeros(len(img_keys), dtype=int)
    for (p, f), lab in zip(zip(img_pred_df["patient_id"], img_pred_df["filename"]),
                           img_pred_df["label"]):
        y_img[img_pos[(p, f)]] = int(lab)

    img_of_patient: Dict[str, List[int]] = {p: [] for p in pids}
    for (p, f), i in img_pos.items():
        img_of_patient[p].append(i)
    img_of_patient = {p: np.asarray(sorted(v), dtype=int) for p, v in img_of_patient.items()}

    store = OOFStore(patient_ids=pids, y_pat=y_pat, img_index=img_keys, y_img=y_img,
                     img_of_patient=img_of_patient)
    for s in seeds:
        ps = pat_pred_df[pat_pred_df["seed"] == s]
        order = np.array([pid_pos[p] for p in ps["patient_id"]])
        store.pat_scores[s] = {}
        store.pat_preds[s] = {}
        for arm in ARMS:
            sc = np.full(len(pids), np.nan)
            pr = np.full(len(pids), np.nan)
            sc[order] = ps[f"score_{arm}"].to_numpy()
            pr[order] = ps[f"pred_{arm}"].to_numpy()
            # Any residual NaN means a patient has no out-of-fold prediction for this
            # seed.  Left in place it would sort to the end of the rank vector and yield
            # a finite but meaningless AUC, with nothing printed to say so.
            if not (np.isfinite(sc).all() and np.isfinite(pr).all()):
                raise AssertionError(
                    f"seed {s}, arm {arm}: {int((~np.isfinite(sc)).sum())} of {len(pids)} "
                    "patients have no out-of-fold prediction -- refusing to compute "
                    "metrics on a partially filled score vector.")
            store.pat_scores[s][arm] = sc
            store.pat_preds[s][arm] = pr
        isx = img_pred_df[img_pred_df["seed"] == s]
        iorder = np.array([img_pos[(p, f)] for p, f in zip(isx["patient_id"], isx["filename"])])
        store.img_scores[s] = {}
        store.img_preds[s] = {}
        for arm in IMAGE_ARMS:
            sc = np.full(len(img_keys), np.nan)
            pr = np.full(len(img_keys), np.nan)
            sc[iorder] = isx[f"prob_{arm}"].to_numpy()
            pr[iorder] = isx[f"pred_{arm}"].to_numpy()
            if not (np.isfinite(sc).all() and np.isfinite(pr).all()):
                raise AssertionError(
                    f"seed {s}, image arm {arm}: {int((~np.isfinite(sc)).sum())} of "
                    f"{len(img_keys)} images have no out-of-fold prediction.")
            store.img_scores[s][arm] = sc
            store.img_preds[s][arm] = pr
    return store


def _bootstrap_p(d: np.ndarray, b_used: int) -> float:
    """Two-sided percentile-bootstrap p for a difference, FLOORED at its resolution limit.

    With B resamples the smallest attainable one-sided tail mass is 1/B, so the smallest
    honest two-sided p is 2/(B+1).  Reporting the raw 2*min(...) prints `p = 0.0000`
    whenever no draw crosses zero, which is a precision the resampling cannot deliver."""
    p = 2.0 * min(float(np.mean(d <= 0)), float(np.mean(d >= 0)))
    return float(min(1.0, max(p, 2.0 / (b_used + 1))))


def holm_bonferroni(pvals: Dict[str, float]) -> Dict[str, float]:
    """Holm step-down adjusted p-values for a FAMILY of comparisons.

    Four paired comparisons are reported (cnn-vs-clinical, fused-vs-cnn, cnn-vs-svm,
    fused-vs-clinical).  Reading each raw p against 0.05 inflates the family-wise error
    rate to roughly 1 - 0.95^4 ~ 0.19; the previous draft was rejected in part for exactly
    this ("a detailed description of the statistical tests").  Holm is uniformly more
    powerful than Bonferroni and needs no independence assumption, which matters here
    because the four comparisons share arms and are strongly positively dependent.

    Non-finite p-values are not adjustable and are returned as NaN; they are also excluded
    from the family size, so an undecided comparison cannot make the others look stricter
    OR looser than they are -- it is simply reported as undecided.
    """
    items = sorted(((k, float(v)) for k, v in pvals.items() if np.isfinite(v)),
                   key=lambda kv: kv[1])
    m = len(items)
    adj: Dict[str, float] = {}
    running = 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))   # step-down monotonicity
        adj[k] = float(running)
    for k in pvals:
        adj.setdefault(k, float("nan"))
    return adj


def pooled_vs_per_fold_auc(pat_pred: pd.DataFrame, seeds: Sequence[int],
                           arms: Sequence[str]) -> dict:
    """Diagnostic for the pooled-out-of-fold AUC.

    Every headline AUC here is computed on out-of-fold scores POOLED across the outer
    folds, i.e. the ranking mixes scores produced by k different models.  If those models
    are calibrated differently the pooled AUC is a biased estimate of the per-fold AUC
    (Airola et al. 2011; Forman & Scholz 2010) -- it can be either optimistic or
    pessimistic.  Reporting the mean of the per-fold AUCs beside the pooled one makes the
    size of that artefact visible instead of leaving it as an unstated assumption.
    Per-fold AUCs on ~20 test patients are very noisy, which is why pooling is used for
    the headline; this is a diagnostic, not a replacement.
    """
    out: dict = {}
    for a in arms:
        pooled, foldmeans, allfolds = [], [], []
        for s in seeds:
            ps = pat_pred[pat_pred["seed"] == s]
            pooled.append(fast_auc(ps["label"].to_numpy(), ps[f"score_{a}"].to_numpy()))
            per_f = []
            for f, g in ps.groupby("fold"):
                if g["label"].nunique() < 2:
                    continue                     # single-class fold: AUC undefined
                per_f.append(fast_auc(g["label"].to_numpy(), g[f"score_{a}"].to_numpy()))
            allfolds.extend(per_f)
            foldmeans.append(float(np.mean(per_f)) if per_f else float("nan"))
        pooled = np.asarray(pooled, dtype=float)
        foldmeans = np.asarray(foldmeans, dtype=float)
        out[a] = {
            "pooled_oof_auc_mean_over_seeds": float(np.nanmean(pooled)),
            "mean_of_per_fold_auc_mean_over_seeds": float(np.nanmean(foldmeans)),
            "pooling_gap": float(np.nanmean(pooled) - np.nanmean(foldmeans)),
            "per_fold_auc_sd": float(np.std(allfolds, ddof=1)) if len(allfolds) > 1
                               else float("nan"),
            "n_folds_used": int(len(allfolds)),
        }
    return out


def _ba_from_pred(y: np.ndarray, pred: np.ndarray) -> float:
    tp = np.sum((y == 1) & (pred == 1)); fn = np.sum((y == 1) & (pred == 0))
    tn = np.sum((y == 0) & (pred == 0)); fp = np.sum((y == 0) & (pred == 1))
    tpr = tp / (tp + fn) if (tp + fn) else np.nan
    tnr = tn / (tn + fp) if (tn + fp) else np.nan
    return float((tpr + tnr) / 2.0)


def cluster_bootstrap(store: OOFStore, seeds: Sequence[int], B: int, rng_seed: int,
                      level: str = "patient", fixed_seed: Optional[int] = None
                      ) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Label-stratified patient cluster bootstrap.

    Patients are the resampling unit at BOTH levels: drawing a patient at image level
    drags in all of that patient's images, which is what keeps the interval honest
    (four views of one child on one day are nearly perfectly correlated; a naive image
    bootstrap would be roughly half as wide as it should be).

    If fixed_seed is None the repetition (CV seed) is randomised per bootstrap draw, so
    the interval carries both sampling variability and CV-partition variability.
    """
    rng = np.random.default_rng(rng_seed)
    y = store.y_pat
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    arms = ARMS if level == "patient" else IMAGE_ARMS
    draws = {a: {"auc": [], "balanced_accuracy": []} for a in arms}
    seed_list = list(seeds)

    for _ in range(B):
        s = fixed_seed if fixed_seed is not None else seed_list[rng.integers(len(seed_list))]
        take = np.concatenate([rng.choice(pos_idx, size=len(pos_idx), replace=True),
                               rng.choice(neg_idx, size=len(neg_idx), replace=True)])
        if level == "patient":
            yy = y[take]
            for a in arms:
                draws[a]["auc"].append(fast_auc(yy, store.pat_scores[s][a][take]))
                draws[a]["balanced_accuracy"].append(
                    _ba_from_pred(yy, store.pat_preds[s][a][take]))
        else:
            img_take = np.concatenate([store.img_of_patient[store.patient_ids[i]] for i in take])
            yy = store.y_img[img_take]
            for a in arms:
                draws[a]["auc"].append(fast_auc(yy, store.img_scores[s][a][img_take]))
                draws[a]["balanced_accuracy"].append(
                    _ba_from_pred(yy, store.img_preds[s][a][img_take]))

    # Observed statistic on the FULL sample, no resampling.  The percentile interval must be
    # quoted next to THIS, not next to the bootstrap mean: the bootstrap mean is a different
    # (bias-inflated) estimator, and quoting it as the point estimate makes the CI table
    # disagree with the mean-over-seeds AUC printed in the main results table.
    obs_seeds = [fixed_seed] if fixed_seed is not None else seed_list
    point: Dict[str, Dict[str, float]] = {}
    for a in arms:
        if level == "patient":
            au = [fast_auc(y, store.pat_scores[s][a]) for s in obs_seeds]
            bb = [_ba_from_pred(y, store.pat_preds[s][a]) for s in obs_seeds]
        else:
            au = [fast_auc(store.y_img, store.img_scores[s][a]) for s in obs_seeds]
            bb = [_ba_from_pred(store.y_img, store.img_preds[s][a]) for s in obs_seeds]
        point[a] = {"auc": float(np.nanmean(au)), "balanced_accuracy": float(np.nanmean(bb))}

    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for a in arms:
        out[a] = {}
        for met, vals in draws[a].items():
            v = np.asarray(vals, dtype=float)
            v = v[np.isfinite(v)]
            bm = float(np.mean(v)) if len(v) else float("nan")
            out[a][met] = {
                "point_estimate": point[a][met],       # report THIS with the interval
                "boot_mean": bm,
                "boot_bias": bm - point[a][met],       # |bias| >> 0 => percentile CI suspect
                "ci_lo": float(np.percentile(v, 2.5)) if len(v) else float("nan"),
                "ci_hi": float(np.percentile(v, 97.5)) if len(v) else float("nan"),
                "ci_method": "percentile (uncorrected); not BCa",
                "B_used": int(len(v)),
                "B_requested": int(B),
            }
    return out


def paired_bootstrap_delta(store: OOFStore, seeds: Sequence[int], arm_a: str, arm_b: str,
                           B: int, rng_seed: int) -> Dict[str, float]:
    """Paired bootstrap on AUC(arm_a) - AUC(arm_b) using identical patient draws."""
    rng = np.random.default_rng(rng_seed)
    y = store.y_pat
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    seed_list = list(seeds)
    # Observed difference on the full sample (mean over repetitions).  `delta_mean` below is
    # the mean of the RESAMPLED differences, which is a bootstrap mean, not the estimate.
    obs = float(np.nanmean([fast_auc(y, store.pat_scores[s][arm_a]) -
                            fast_auc(y, store.pat_scores[s][arm_b]) for s in seed_list]))
    deltas = []
    for _ in range(B):
        s = seed_list[rng.integers(len(seed_list))]
        take = np.concatenate([rng.choice(pos_idx, size=len(pos_idx), replace=True),
                               rng.choice(neg_idx, size=len(neg_idx), replace=True)])
        yy = y[take]
        deltas.append(fast_auc(yy, store.pat_scores[s][arm_a][take]) -
                      fast_auc(yy, store.pat_scores[s][arm_b][take]))
    d = np.asarray(deltas, dtype=float)
    d = d[np.isfinite(d)]
    if not len(d):
        return {"delta_observed": obs, "delta_mean": float("nan"), "ci_lo": float("nan"),
                "ci_hi": float("nan"), "p_bootstrap": float("nan"),
                "p_bootstrap_floor": float("nan"), "B_used": 0}
    p = _bootstrap_p(d, len(d))
    return {"delta_observed": obs, "delta_mean": float(np.mean(d)),
            "boot_bias": float(np.mean(d)) - obs,
            "ci_lo": float(np.percentile(d, 2.5)),
            "ci_hi": float(np.percentile(d, 97.5)), "p_bootstrap": p,
            "p_bootstrap_floor": float(2.0 / (len(d) + 1)), "B_used": int(len(d)),
            "ci_method": "percentile (uncorrected); not BCa"}


# ==================================================================================
# 8.  One outer fold
# ==================================================================================
def logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def aggregate_patient_scores(img_rows: pd.DataFrame, probs: np.ndarray) -> pd.DataFrame:
    """Patient score by (a) plain mean over the patient's images -- the primary rule in
    this script -- and (b) the protocol's mean-over-views-then-max-over-kidneys rule,
    stored alongside so both can be compared without retraining."""
    tmp = img_rows[["patient_id", "label", "side"]].copy()
    tmp["p"] = probs
    rows = []
    for pid, g in tmp.groupby("patient_id", sort=True):
        side_means = g.groupby("side")["p"].mean()
        rows.append({
            "patient_id": pid,
            "label": int(g["label"].iloc[0]),
            "n_images": int(len(g)),
            "score_mean": float(g["p"].mean()),
            "score_maxkidney": float(side_means.max()),
        })
    return pd.DataFrame(rows)


def train_cnn(arch: str, tr_img: pd.DataFrame, va_img: pd.DataFrame,
              args, device, seed: int, tag: str) -> Tuple[nn.Module, dict]:
    """Train on inner-training images; early stop on inner-VALIDATION patient-level
    balanced accuracy (at that epoch's Youden threshold).  The validation split contains
    no patient that appears in training, and no test patient is involved at any point."""
    model = build_model(arch).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    w = sample_weights(tr_img)
    ds = KidneyImageDataset(tr_img["abs_path"].tolist(), tr_img["label"].tolist(), w, TRAIN_TRANSFORM)

    best = {"metric": -np.inf, "epoch": -1, "state": None, "val_auc": float("nan")}
    history = []
    agg_col = "score_mean" if args.patient_agg == "mean" else "score_maxkidney"

    # Training-loader strategy.  With workers == 0 (the default) a fresh loader per epoch
    # is free and gives each epoch its own reproducible shuffle seed -- keep it, so the
    # default run stays bit-identical.  With workers > 0 a per-epoch loader is a trap on
    # Windows: every construction spawns fresh interpreters that re-import torch +
    # torchvision + sklearn.  Measured on this machine (NCKUH, 616 training images,
    # densenet121, batch 16): workers=0 -> 26 s/epoch; workers=4 rebuilt every epoch ->
    # 148 s/epoch (5.7x SLOWER); workers=4 built once and reused -> 16 s/epoch.  When the
    # user asks for workers we build ONE persistent loader for the whole fold; its
    # generator is seeded once and advances across epochs, which is still deterministic.
    persistent_loader = None
    if args.workers > 0:
        persistent_loader = make_loader(ds, args.batch_size, True, args.workers,
                                        seed * 1000, persistent=True)

    bar = tqdm(range(args.epochs), desc=f"{tag} train", leave=False, ncols=88, ascii=True)
    for ep in bar:
        model.train()
        loader = (persistent_loader if persistent_loader is not None
                  else make_loader(ds, args.batch_size, True, args.workers, seed * 1000 + ep))
        tot, nb = 0.0, 0
        for imgs, labels, wts, _ in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            wts = wts.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                out = model(imgs)
                per = F.cross_entropy(out, labels, reduction="none")
                loss = (per * wts).sum() / wts.sum().clamp_min(1e-8)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += float(loss.detach()) * len(labels)
            nb += len(labels)

        vp = predict_probs(model, va_img, device, args.batch_size, args.workers, args.amp)
        vpat = aggregate_patient_scores(va_img, vp)
        yv = vpat["label"].to_numpy()
        sv = vpat[agg_col].to_numpy()
        thr = youden_threshold(yv, sv)
        ba = _ba_from_pred(yv, (sv >= thr).astype(int))
        auc = fast_auc(yv, sv)
        monitored = ba if args.monitor == "patient_ba" else (auc if np.isfinite(auc) else -np.inf)
        history.append({"epoch": ep + 1, "train_loss": tot / max(1, nb),
                        "val_patient_ba": ba, "val_patient_auc": auc})
        bar.set_postfix(loss=f"{tot/max(1,nb):.3f}", val_ba=f"{ba:.3f}", val_auc=f"{auc:.3f}")

        if monitored > best["metric"] + 1e-6:
            best = {"metric": float(monitored), "epoch": ep + 1,
                    "state": copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()}),
                    "val_auc": float(auc), "val_ba": float(ba)}
        elif ep + 1 - best["epoch"] >= args.patience:
            break

    if best["state"] is None:
        # Every epoch's monitored value was NaN or -inf, so no epoch was ever selected and
        # the weights still in `model` are simply those of the LAST epoch.  Returning them
        # would present an unselected model as an early-stopped one.
        raise RuntimeError(
            f"{tag}: early stopping never selected an epoch (monitor={args.monitor} was not "
            "finite at any epoch, e.g. a single-class validation split).  Refusing to "
            "return last-epoch weights as if they had been validated.")
    model.load_state_dict(best["state"])
    info = {"best_epoch": best["epoch"], "best_monitor": best["metric"],
            "best_val_patient_auc": best.get("val_auc", float("nan")),
            "best_val_patient_ba": best.get("val_ba", float("nan")),
            "epochs_run": len(history), "history": history}
    return model, info


def run_fold(args, device, seed: int, fold: int, pat: pd.DataFrame, img_df: pd.DataFrame,
             tr_idx: np.ndarray, te_idx: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Train every arm on one outer fold and score the held-out test patients once."""
    pat_tr_all = pat.iloc[tr_idx].reset_index(drop=True)
    pat_te = pat.iloc[te_idx].reset_index(drop=True)

    itr, iva = inner_split(pat_tr_all, args.inner_folds, seed)
    pat_itr = pat_tr_all.iloc[itr].reset_index(drop=True)
    pat_iva = pat_tr_all.iloc[iva].reset_index(drop=True)

    # ---- HARD LEAKAGE ASSERTION --------------------------------------------------
    assert_patient_disjoint(pat_itr["patient_id"], pat_iva["patient_id"], pat_te["patient_id"],
                            f"seed {seed} fold {fold}")
    if set(pat_tr_all["patient_id"]) & set(pat_te["patient_id"]):
        raise AssertionError("outer train/test patient overlap")

    def rows_of(pids) -> pd.DataFrame:
        return img_df[img_df["patient_id"].isin(set(pids))].sort_values("img_idx").reset_index(drop=True)

    img_itr = rows_of(pat_itr["patient_id"])
    img_iva = rows_of(pat_iva["patient_id"])
    img_te = rows_of(pat_te["patient_id"])
    if not (len(img_itr) and len(img_iva) and len(img_te)):
        raise RuntimeError(f"empty split at seed {seed} fold {fold}")

    # ---- DEGENERATE-FOLD DETECTION -----------------------------------------------
    # A single-class inner-validation set makes every threshold arbitrary (youden_threshold
    # falls back to 0.5) and a single-class test fold makes AUC undefined (fast_auc returns
    # NaN, roc_auc_score raises inside a try/except that swallows it).  Either one would
    # travel silently into the pooled out-of-fold numbers, so detect it here instead.
    degenerate = []
    for _name, _tbl in (("inner_train", pat_itr), ("inner_val", pat_iva), ("test", pat_te)):
        _cls = sorted({int(v) for v in _tbl["label"]})
        if len(_cls) < 2:
            degenerate.append(f"{_name} contains only class {_cls} ({len(_tbl)} patients)")
    if degenerate:
        raise RuntimeError(
            f"DEGENERATE FOLD at seed {seed} fold {fold}: " + "; ".join(degenerate) +
            ".  Thresholds and AUC are undefined on a single-class split; lower --folds "
            "or --inner-folds rather than letting this fold emit a silent NaN or a "
            "default 0.5 threshold.")

    counts = {
        "seed": seed, "fold": fold,
        "splits": [
            split_counts(pat, img_df, pat_itr["patient_id"], "inner_train"),
            split_counts(pat, img_df, pat_iva["patient_id"], "inner_val"),
            split_counts(pat, img_df, pat_tr_all["patient_id"], "outer_train_total"),
            split_counts(pat, img_df, pat_te["patient_id"], "test"),
        ],
        "leakage_check": "passed: train/val/test patient sets are pairwise disjoint",
    }

    agg_col = "score_mean" if args.patient_agg == "mean" else "score_maxkidney"

    # ---------------------------------------------------------------- ARM 2: CNN --
    model, cnn_info = train_cnn(args.arch, img_itr, img_iva, args, device, seed,
                                f"s{seed}f{fold}")
    p_va = predict_probs(model, img_iva, device, args.batch_size, args.workers, args.amp)
    p_te = predict_probs(model, img_te, device, args.batch_size, args.workers, args.amp)
    pat_va_cnn = aggregate_patient_scores(img_iva, p_va)
    pat_te_cnn = aggregate_patient_scores(img_te, p_te)

    thr_cnn = youden_threshold(pat_va_cnn["label"].to_numpy(), pat_va_cnn[agg_col].to_numpy())
    thr_cnn90 = sensitivity_threshold(pat_va_cnn["label"].to_numpy(),
                                      pat_va_cnn[agg_col].to_numpy(), 0.90)
    thr_cnn_img = youden_threshold(img_iva["label"].to_numpy(), p_va)

    # ----------------------------------------------------------- ARM 1: clinical --
    # Fitted on the INNER-TRAINING patients only, exactly like the CNN, so that the two
    # arms see identical data and the inner-validation patients stay clean for both.
    Xc_tr = pat_itr[CLINICAL_FEATURES].to_numpy(float)
    Xc_va = pat_iva[CLINICAL_FEATURES].to_numpy(float)
    Xc_te = pat_te[CLINICAL_FEATURES].to_numpy(float)
    sc_c = StandardScaler().fit(Xc_tr)                 # scaler fitted on training data only
    clf_c = LogisticRegression(max_iter=2000, C=args.clinical_C)
    clf_c.fit(sc_c.transform(Xc_tr), pat_itr["label"].to_numpy())
    s_c_va = clf_c.predict_proba(sc_c.transform(Xc_va))[:, 1]
    s_c_te = clf_c.predict_proba(sc_c.transform(Xc_te))[:, 1]
    thr_clin = youden_threshold(pat_iva["label"].to_numpy(), s_c_va)
    thr_clin90 = sensitivity_threshold(pat_iva["label"].to_numpy(), s_c_va, 0.90)

    # -------------------------------------------------------------- ARM 3: fused --
    # The fusion layer is fitted on the INNER-VALIDATION patients, i.e. patients the CNN
    # never trained on.  Fitting it on the CNN's own training patients would feed the
    # fuser near-memorised probabilities and inflate the fused arm; that is exactly the
    # defect in the previous draft's Section 3.7 multimodal analysis.
    va_order = pat_iva.set_index("patient_id").loc[pat_va_cnn["patient_id"]].reset_index()
    te_order = pat_te.set_index("patient_id").loc[pat_te_cnn["patient_id"]].reset_index()
    # A silent row misalignment here would corrupt the fused arm without raising anything.
    # These are explicit raises, not `assert`: `python -O` strips assert statements, and
    # this script's contract is that it ABORTS rather than emit a corrupted number.
    _require(list(va_order["patient_id"]) == list(pat_va_cnn["patient_id"]),
             "val row misalignment between the patient table and the CNN patient scores")
    _require(list(te_order["patient_id"]) == list(pat_te_cnn["patient_id"]),
             "test row misalignment between the patient table and the CNN patient scores")
    _require(bool((va_order["label"].to_numpy() == pat_va_cnn["label"].to_numpy()).all()),
             "val label mismatch after reordering")
    _require(bool((te_order["label"].to_numpy() == pat_te_cnn["label"].to_numpy()).all()),
             "test label mismatch after reordering")
    # NOTE: the fusion features are the RAW clinical covariates, not the clinical arm's
    # predicted probability -- the fusion LR learns its own clinical weights on its own
    # fitting set.  s_c_va is therefore deliberately not a fusion input.
    s_c_te_aligned = pd.Series(s_c_te, index=pat_te["patient_id"]).loc[
        pat_te_cnn["patient_id"]].to_numpy()

    Xf_va = np.column_stack([logit(pat_va_cnn[agg_col].to_numpy()),
                             va_order[CLINICAL_FEATURES].to_numpy(float)])
    Xf_te = np.column_stack([logit(pat_te_cnn[agg_col].to_numpy()),
                             te_order[CLINICAL_FEATURES].to_numpy(float)])
    yf_va = pat_va_cnn["label"].to_numpy()
    if len(np.unique(yf_va)) < 2:
        s_f_te = pat_te_cnn[agg_col].to_numpy()
        thr_fused, thr_fused90 = thr_cnn, thr_cnn90
    else:
        sc_f = StandardScaler().fit(Xf_va)
        clf_f = LogisticRegression(max_iter=2000, C=args.fusion_C)
        clf_f.fit(sc_f.transform(Xf_va), yf_va)
        s_f_te = clf_f.predict_proba(sc_f.transform(Xf_te))[:, 1]
        # Threshold from out-of-fold predictions inside the fusion-fitting set, so it is
        # not an in-sample threshold.  C must match clf_f, or the threshold would sit on a
        # different probability scale than the model it is applied to.
        thr_fused, thr_fused90 = cv_oof_threshold(Xf_va, yf_va, seed, C=args.fusion_C)

    # ---------------------------------------------------------------- ARM 4: SVM --
    # 2022-style pipeline: penultimate CNN embedding -> standardise -> PCA -> RBF SVM.
    # PCA and SVM are fitted on the inner-TRAINING images only (the same images the CNN
    # saw), which mirrors the original design; note in the manuscript that the feature
    # extractor is not cross-fitted, so this arm's feature space is optimistic relative
    # to a fully nested embedding.  Its evaluation patients are still strictly held out.
    f_tr = extract_features(model, args.arch, img_itr, device, args.batch_size, args.workers, args.amp)
    f_va = extract_features(model, args.arch, img_iva, device, args.batch_size, args.workers, args.amp)
    f_te = extract_features(model, args.arch, img_te, device, args.batch_size, args.workers, args.amp)
    sc_s = StandardScaler().fit(f_tr)
    n_comp = int(max(2, min(args.pca_components, f_tr.shape[0] - 1, f_tr.shape[1])))
    pca = PCA(n_components=n_comp, random_state=seed).fit(sc_s.transform(f_tr))
    svm = SVC(kernel="rbf", C=args.svm_C, gamma="scale", probability=True,
              class_weight="balanced", random_state=seed)
    svm.fit(pca.transform(sc_s.transform(f_tr)), img_itr["label"].to_numpy())
    sv_va = svm.predict_proba(pca.transform(sc_s.transform(f_va)))[:, 1]
    sv_te = svm.predict_proba(pca.transform(sc_s.transform(f_te)))[:, 1]
    pat_va_svm = aggregate_patient_scores(img_iva, sv_va)
    pat_te_svm = aggregate_patient_scores(img_te, sv_te)
    _require(list(pat_te_svm["patient_id"]) == list(pat_te_cnn["patient_id"]),
             "svm row misalignment between the SVM and CNN patient score tables")
    thr_svm = youden_threshold(pat_va_svm["label"].to_numpy(), pat_va_svm[agg_col].to_numpy())
    thr_svm90 = sensitivity_threshold(pat_va_svm["label"].to_numpy(),
                                      pat_va_svm[agg_col].to_numpy(), 0.90)
    thr_svm_img = youden_threshold(img_iva["label"].to_numpy(), sv_va)

    # ------------------------------------------------------------------ assemble --
    pat_out = pd.DataFrame({
        "site": args.site, "seed": seed, "fold": fold,
        "patient_id": pat_te_cnn["patient_id"].to_numpy(),
        "label": pat_te_cnn["label"].to_numpy(),
        "n_images": pat_te_cnn["n_images"].to_numpy(),
        "age_years": te_order["age_years"].to_numpy(),
        "gender_code": te_order["gender_code"].to_numpy(),
        "renal_measure_cm": te_order["renal_measure_cm"].to_numpy(),
        "score_clinical": s_c_te_aligned,
        "score_cnn": pat_te_cnn[agg_col].to_numpy(),
        "score_fused": s_f_te,
        "score_svm": pat_te_svm.set_index("patient_id").loc[
            pat_te_cnn["patient_id"], agg_col].to_numpy(),
        "score_cnn_mean": pat_te_cnn["score_mean"].to_numpy(),
        "score_cnn_maxkidney": pat_te_cnn["score_maxkidney"].to_numpy(),
        "thr_clinical": thr_clin, "thr_cnn": thr_cnn, "thr_fused": thr_fused, "thr_svm": thr_svm,
        "thr_cnn_sens90": thr_cnn90, "thr_clinical_sens90": thr_clin90,
        "thr_fused_sens90": thr_fused90, "thr_svm_sens90": thr_svm90,
    })
    for arm in ARMS:
        pat_out[f"pred_{arm}"] = (pat_out[f"score_{arm}"] >= pat_out[f"thr_{arm}"]).astype(int)
        pat_out[f"pred_{arm}_sens90"] = (
            pat_out[f"score_{arm}"] >= pat_out[f"thr_{arm}_sens90"]).astype(int)

    img_out = img_te[["patient_id", "filename", "relative_path", "view", "side", "label"]].copy()
    img_out.insert(0, "fold", fold)
    img_out.insert(0, "seed", seed)
    img_out.insert(0, "site", args.site)
    img_out["prob_cnn"] = p_te
    img_out["prob_svm"] = sv_te
    img_out["thr_cnn_img"] = thr_cnn_img
    img_out["thr_svm_img"] = thr_svm_img
    img_out["pred_cnn"] = (p_te >= thr_cnn_img).astype(int)
    img_out["pred_svm"] = (sv_te >= thr_svm_img).astype(int)

    counts["cnn"] = {k: v for k, v in cnn_info.items() if k != "history"}
    counts["cnn_history"] = cnn_info["history"]
    counts["thresholds"] = {
        "cnn_patient_youden": thr_cnn, "cnn_patient_sens90": thr_cnn90,
        "cnn_image_youden": thr_cnn_img, "clinical_patient_youden": thr_clin,
        "fused_patient_youden": thr_fused, "svm_patient_youden": thr_svm,
        "svm_image_youden": thr_svm_img,
    }
    counts["svm_pca_components"] = n_comp
    # Selection-set optimism.  The stopping epoch was chosen to maximise the monitored
    # metric on the inner-validation patients, and the fusion layer plus every threshold
    # are then fitted/chosen on the CNN's scores for those SAME patients.  The test fold
    # is untouched, so the test numbers stay valid, but the CNN-score distribution the
    # fusion layer sees is optimistically selected and will not match the test one.
    # Recording both AUCs makes the size of that gap visible instead of assumed small.
    counts["selection_optimism"] = {
        "cnn_inner_val_patient_auc_at_selected_epoch": float(cnn_info.get(
            "best_val_patient_auc", float("nan"))),
        "cnn_test_patient_auc": float(fast_auc(pat_te_cnn["label"].to_numpy(),
                                               pat_te_cnn[agg_col].to_numpy())),
        "note": "inner-val AUC is a SELECTION statistic, not a performance estimate; the "
                "gap to the test AUC is the optimism carried into the fusion coefficients "
                "and into every decision threshold.",
    }
    # frozen per-patient split membership for this fold (written to fold_assignments.csv;
    # popped before metrics.json so the JSON stays readable)
    counts["_pids"] = {
        "inner_train": pat_itr["patient_id"].tolist(),
        "inner_val": pat_iva["patient_id"].tolist(),
        "test": pat_te["patient_id"].tolist(),
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return pat_out, img_out, counts


# ==================================================================================
# 9.  Reporting
# ==================================================================================
def per_seed_metrics(pat_pred: pd.DataFrame, img_pred: pd.DataFrame,
                     seeds: Sequence[int]) -> dict:
    res = {"patient_level": {}, "image_level": {}, "patient_level_sens90": {}}
    for s in seeds:
        ps = pat_pred[pat_pred["seed"] == s]
        isx = img_pred[img_pred["seed"] == s]
        res["patient_level"][str(s)] = {
            a: binary_metrics(ps["label"].to_numpy(), ps[f"score_{a}"].to_numpy(),
                              ps[f"pred_{a}"].to_numpy()) for a in ARMS}
        res["patient_level_sens90"][str(s)] = {
            a: binary_metrics(ps["label"].to_numpy(), ps[f"score_{a}"].to_numpy(),
                              ps[f"pred_{a}_sens90"].to_numpy()) for a in ARMS}
        res["image_level"][str(s)] = {
            a: binary_metrics(isx["label"].to_numpy(), isx[f"prob_{a}"].to_numpy(),
                              isx[f"pred_{a}"].to_numpy()) for a in IMAGE_ARMS}
    return res


def summarise_across_seeds(per_seed: dict, level: str, arms: Sequence[str]) -> dict:
    out = {}
    keys = ["auc", "average_precision", "accuracy", "balanced_accuracy", "tpr_sensitivity",
            "tnr_specificity", "precision_ppv", "npv", "f1", "min_tpr_tnr"]
    for a in arms:
        out[a] = {}
        for k in keys:
            raw = np.array([per_seed[level][s][a][k] for s in per_seed[level]], dtype=float)
            vals = raw[np.isfinite(raw)]
            out[a][k] = {
                "mean": float(np.mean(vals)) if len(vals) else float("nan"),
                # SD over a SINGLE seed is not "zero variability", it is UNESTIMABLE.
                # Printing 0.000 invites the summary's own rule ("SD < 0.02 supports
                # partition independence") to be claimed from one repetition.
                "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan"),
                # Non-finite seeds are dropped; say how many, so a mean over 2 of 5 seeds
                # cannot be read as a mean over 5.
                "n_seeds_total": int(len(raw)),
                "n_seeds_used": int(len(vals)),
                "per_seed": [float(v) for v in vals],
            }
    return out


def run_delong_comparisons(pat_pred: pd.DataFrame, seeds: Sequence[int]) -> dict:
    """Paired DeLong per seed.  A comparison is called significant only if it is
    significant at EVERY seed (max p rule) -- this makes seed cherry-picking impossible.

    CAVEAT (recorded in metrics.json and summary.txt): DeLong's variance assumes two
    FIXED models scored on independent observations.  Here the scores are pooled across
    outer folds, so five different models produced them and every patient's data helped
    train the models that scored the other patients.  The observations are therefore not
    independent and the DeLong SE is anti-conservative.  The paired patient cluster
    bootstrap is the primary inference; DeLong is a secondary, optimistic check."""
    # fused_vs_clinical is the protocol's primary endpoint (does imaging add anything to
    # the clinical variables?).  It had a paired bootstrap but no DeLong test, so the one
    # comparison the paper is built on was the only one without the standard AUC test.
    pairs = [("cnn", "clinical"), ("fused", "cnn"), ("cnn", "svm"), ("fused", "clinical")]
    seed_list = list(seeds)
    out = {}
    for a, b in pairs:
        per_seed = {}
        for s in seed_list:
            ps = pat_pred[pat_pred["seed"] == s]
            per_seed[str(s)] = delong_test(ps["label"].to_numpy(),
                                           ps[f"score_{a}"].to_numpy(),
                                           ps[f"score_{b}"].to_numpy())
        ps_vals = [v["p"] for v in per_seed.values() if np.isfinite(v["p"])]
        n_undecided = len(seed_list) - len(ps_vals)
        deltas = [v["delta"] for v in per_seed.values() if np.isfinite(v["delta"])]
        # A seed whose DeLong variance degenerated has NO p-value.  Dropping it and then
        # taking max() over what is left would claim "significant at EVERY seed" from a
        # strict subset of the seeds, which is the opposite of the stated rule.
        out[f"{a}_vs_{b}"] = {
            "comparison": f"AUC({ARM_LABELS[a]}) - AUC({ARM_LABELS[b]})",
            "per_seed": per_seed,
            "n_seeds": len(seed_list),
            "n_seeds_without_valid_p": int(n_undecided),
            "max_p_over_seeds": float(np.max(ps_vals)) if ps_vals else float("nan"),
            "min_p_over_seeds": float(np.min(ps_vals)) if ps_vals else float("nan"),
            "mean_delta": float(np.mean(deltas)) if deltas else float("nan"),
            "significant_at_every_seed_0.05": bool(
                n_undecided == 0 and ps_vals and max(ps_vals) < 0.05),
            "inference_note": "secondary; DeLong SE is anti-conservative on CV-pooled "
                              "predictions -- use the paired cluster bootstrap as primary",
        }
    return out


def make_figures(pat_pred: pd.DataFrame, img_pred: pd.DataFrame, across: dict,
                 out_dir: Path, site: str, ref_seed: int) -> None:
    colors = {"clinical": "#4C72B0", "cnn": "#C44E52", "fused": "#55A868", "svm": "#8172B2"}

    def _curves(df, arms, score_col, level, fname_roc, fname_pr, summary_key):
        # The curve drawn is ONE repetition (ref_seed).  Its legend must therefore report
        # THAT curve's own AUC/AP; labelling a single-seed curve with the across-seed mean
        # lets a reader measure the plotted curve and get a different number than the
        # caption.  The across-seed mean +/- SD goes in the title, where it belongs.
        d = df[df["seed"] == ref_seed]
        y = d["label"].to_numpy()
        n_seeds = len(df["seed"].unique())

        fig, ax = plt.subplots(figsize=(5.2, 5.0))
        means = []
        for a in arms:
            s = d[score_col.format(a)].to_numpy()
            fpr, tpr, _ = roc_curve(y, s)
            auc_here = float(roc_auc_score(y, s))
            m = across[summary_key][a]["auc"]
            means.append(f"{a} {m['mean']:.3f}+/-{m['sd']:.3f}")
            ax.plot(fpr, tpr, lw=1.6, color=colors[a],
                    label=f"{ARM_LABELS[a]}\nAUC {auc_here:.3f} (this curve, seed {ref_seed})")
        ax.plot([0, 1], [0, 1], ls="--", lw=0.9, color="0.5")
        ax.set_xlabel("1 - specificity"); ax.set_ylabel("Sensitivity")
        ax.set_title(f"{site}: {level}-level ROC, out-of-fold, seed {ref_seed} of {n_seeds}\n"
                     f"{n_seeds}-seed mean AUC +/- SD: " + "; ".join(means), fontsize=7.5)
        ax.legend(fontsize=6.5, loc="lower right", frameon=False)
        fig.tight_layout(); fig.savefig(out_dir / fname_roc, dpi=600); plt.close(fig)

        fig, ax = plt.subplots(figsize=(5.2, 5.0))
        means = []
        for a in arms:
            s = d[score_col.format(a)].to_numpy()
            pr, rc, _ = precision_recall_curve(y, s)
            ap_here = float(average_precision_score(y, s))
            m = across[summary_key][a]["average_precision"]
            means.append(f"{a} {m['mean']:.3f}+/-{m['sd']:.3f}")
            ax.plot(rc, pr, lw=1.6, color=colors[a],
                    label=f"{ARM_LABELS[a]}\nAP {ap_here:.3f} (this curve, seed {ref_seed})")
        ax.axhline(float(np.mean(y)), ls="--", lw=0.9, color="0.5", label="Prevalence")
        ax.set_xlabel("Recall (sensitivity)"); ax.set_ylabel("Precision (PPV)")
        ax.set_title(f"{site}: {level}-level precision-recall, out-of-fold, "
                     f"seed {ref_seed} of {n_seeds}\n"
                     f"{n_seeds}-seed mean AP +/- SD: " + "; ".join(means), fontsize=7.5)
        ax.legend(fontsize=6.5, loc="lower left", frameon=False)
        fig.tight_layout(); fig.savefig(out_dir / fname_pr, dpi=600); plt.close(fig)

    _curves(pat_pred, ARMS, "score_{}", "Patient", "roc_patient.png", "pr_patient.png",
            "patient_level")
    _curves(img_pred, IMAGE_ARMS, "prob_{}", "Image", "roc_image.png", "pr_image.png",
            "image_level")


def write_summary(path: Path, args, notes: dict, fold_records: List[dict], per_seed: dict,
                  across: dict, boot: dict, boot_img: dict, boot_per_seed: dict,
                  delong: dict, boot_delta: dict, elapsed: float, env: dict,
                  holm: Optional[dict] = None, pooling: Optional[dict] = None) -> None:
    L: List[str] = []
    add = L.append
    add("=" * 96)
    add(f"SINGLE-CENTER PATIENT-LEVEL CROSS-VALIDATION  --  SITE: {args.site}")
    add("=" * 96)
    add(f"Run finished           : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    add(f"Elapsed                : {elapsed/60:.1f} min")
    add(f"Backbone               : {args.arch}   (ImageNet weights, 2-output head)")
    add(f"Seeds x outer folds    : {args.seeds} x {args.folds}"
        f"   (inner validation = 1/{args.inner_folds} of the training patients)")
    add(f"Epochs / patience      : {args.epochs} / {args.patience}   monitor = {args.monitor}")
    add(f"Patient aggregation    : {args.patient_agg}"
        "  (both mean and max-over-kidneys scores are saved in the CSV)")
    add(f"Bootstrap              : B = {args.bootstrap}, label-stratified patient clusters,"
        " CV-seed randomised per draw")
    add(f"Environment            : {env['python']}, torch {env['torch']}, {env['gpu']}")
    add("")
    add("-" * 96)
    add("COHORT")
    add("-" * 96)
    add(f"Images before repeat-exam rule : {notes['images_before_exam_filter']}"
        f"   exams {notes['exams_before_exam_filter']}   patients {notes['patients']}")
    add(f"Repeat-exam patients           : {notes['repeat_exam_patients']}"
        f"   (keep_first_exam_only = {args.keep_first_exam_only})")
    add(f"Images dropped (2nd exam)      : {notes['dropped_second_exam_images']}")
    add(f"Images analysed                : {notes['images_after_exam_filter']}"
        f"  ({notes['images_pos']} positive / {notes['images_neg']} negative)")
    add("")
    add("-" * 96)
    add("SPLIT COUNTS PER FOLD  (patient-level; the leakage assertion passed for every fold)")
    add("-" * 96)
    add(f"{'seed':>4} {'fold':>4} {'split':<18} {'pat':>5} {'pat+':>5} {'pat-':>5} "
        f"{'prev':>6} {'img':>5} {'img+':>5} {'img-':>5}")
    for rec in fold_records:
        for sp in rec["splits"]:
            add(f"{rec['seed']:>4} {rec['fold']:>4} {sp['split']:<18} {sp['patients']:>5} "
                f"{sp['patients_pos']:>5} {sp['patients_neg']:>5} {sp['patient_prevalence']:>6.3f} "
                f"{sp['images']:>5} {sp['images_pos']:>5} {sp['images_neg']:>5}")
    add("")
    add("-" * 96)
    add("PRIMARY RESULTS -- PATIENT LEVEL, pooled out-of-fold, mean +/- SD across seeds")
    add("-" * 96)
    add(f"{'arm':<10}{'AUC (mean+/-SD)':>18}{'BalAcc (mean+/-SD)':>20}{'Sens':>9}{'Spec':>9}"
        f"{'PPV':>9}{'NPV':>9}{'F1':>9}{'Acc':>9}{'min(Se,Sp)':>12}")
    for a in ARMS:
        m = across["patient_level"][a]
        add(f"{a:<10}"
            + f"{m['auc']['mean']:.3f}+/-{m['auc']['sd']:.3f}".rjust(18)
            + f"{m['balanced_accuracy']['mean']:.3f}+/-{m['balanced_accuracy']['sd']:.3f}".rjust(20)
            + f"{m['tpr_sensitivity']['mean']:.3f}".rjust(9)
            + f"{m['tnr_specificity']['mean']:.3f}".rjust(9)
            + f"{m['precision_ppv']['mean']:.3f}".rjust(9)
            + f"{m['npv']['mean']:.3f}".rjust(9)
            + f"{m['f1']['mean']:.3f}".rjust(9)
            + f"{m['accuracy']['mean']:.3f}".rjust(9)
            + f"{m['min_tpr_tnr']['mean']:.3f}".rjust(12))
    add("")
    add("Partition stability: SD across seeds is a stability statistic, NOT a confidence")
    add("interval.  SD < 0.02 supports partition independence; SD > 0.05 must be reported")
    add("prominently as instability.")
    add("")
    add("-" * 96)
    add("95% CONFIDENCE INTERVALS -- stratified patient cluster bootstrap")
    add("(CV seed randomised per draw, so the interval carries sampling AND partition variability)")
    add("-" * 96)
    add("Point estimates below are the OBSERVED statistics on the full cohort, not the")
    add("bootstrap means; the bootstrap mean of an AUC is biased toward 0.5 and quoting it")
    add("here would disagree with the mean-over-seeds AUC in the table above.  The")
    add("bootstrap bias (boot_mean - observed) is in metrics.json: if |bias| is an")
    add("appreciable fraction of the interval half-width, the uncorrected percentile")
    add("interval is not trustworthy and a BCa interval is required.")
    add(f"{'arm':<10}{'AUC [95% CI]':<30}{'Balanced accuracy [95% CI]':<30}{'bias':>8}")
    for a in ARMS:
        b = boot[a]
        add(f"{a:<10}"
            + f"{b['auc']['point_estimate']:.3f} [{b['auc']['ci_lo']:.3f}, "
              f"{b['auc']['ci_hi']:.3f}]".ljust(30 - 6)
            + f"{b['balanced_accuracy']['point_estimate']:.3f} "
              f"[{b['balanced_accuracy']['ci_lo']:.3f}, "
              f"{b['balanced_accuracy']['ci_hi']:.3f}]".ljust(30 - 6)
            + f"{b['auc']['boot_bias']:+.4f}".rjust(8))
    add("")
    add("Per-seed AUC CIs (partition held fixed within each interval):")
    for s, d in boot_per_seed.items():
        add("  seed " + s + ":  " + "   ".join(
            f"{a}={d[a]['auc']['point_estimate']:.3f} [{d[a]['auc']['ci_lo']:.3f},"
            f"{d[a]['auc']['ci_hi']:.3f}]" for a in ARMS))
    add("")
    add("-" * 96)
    add("PAIRED COMPARISONS")
    add("-" * 96)
    add("PRIMARY INFERENCE -- paired patient cluster bootstrap on the AUC difference")
    add("(identical patient draws for both arms; CV seed randomised per draw).")
    for key, d in boot_delta.items():
        floor = d.get("p_bootstrap_floor", float("nan"))
        at_floor = np.isfinite(floor) and d["p_bootstrap"] <= floor + 1e-12
        add(f"  {key:<18} dAUC = {d.get('delta_observed', d['delta_mean']):+.4f}  95% CI "
            f"[{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}]   bootstrap p "
            + (f"< {floor:.4f} (resolution limit of B = {d.get('B_used', 0)} draws)"
               if at_floor else f"= {d['p_bootstrap']:.4f}")
            + (f"   Holm p = {holm['bootstrap_p_holm'][key]:.4f}"
               if holm and key in holm.get("bootstrap_p_holm", {}) else ""))
    add("")
    if holm:
        add("MULTIPLICITY: Holm-Bonferroni step-down over the family of "
            f"{holm['family_size_bootstrap']} paired comparisons.  Unadjusted, reading four")
        add("comparisons each at 0.05 gives a family-wise error rate near 0.19.  A")
        add("comparison is claimed only if its HOLM-ADJUSTED p is below 0.05; raw p is")
        add("reported for transparency, not for the claim.")
        for key in sorted(holm["bootstrap_p_holm"]):
            add(f"    {key:<20} raw p = {boot_delta[key]['p_bootstrap']:.4f}   "
                f"Holm p = {holm['bootstrap_p_holm'][key]:.4f}   "
                + ("survives Holm at 0.05"
                   if np.isfinite(holm['bootstrap_p_holm'][key])
                   and holm['bootstrap_p_holm'][key] < 0.05 else "does not survive Holm"))
        add("")
    add("SECONDARY -- DeLong, patient level only (DeLong assumes independent observations,")
    add("which images of one patient are not).  Rule: significant only if p < 0.05 at EVERY")
    add("seed AND every seed yielded a valid variance.")
    add(f"CAVEAT: these predictions are POOLED ACROSS OUTER FOLDS, so {args.folds} different")
    add("models produced them and each patient's data helped train the models that scored the")
    add("others.  The observations are not independent, so the DeLong SE is")
    add("ANTI-CONSERVATIVE.  Quote the bootstrap above, not DeLong, as the inferential test.")
    for key, d in delong.items():
        undec = d.get("n_seeds_without_valid_p", 0)
        add(f"  {key:<18} mean dAUC = {d['mean_delta']:+.4f}   "
            f"p per seed: " + ", ".join(
                ("degenerate" if not np.isfinite(v["p"]) else f"{v['p']:.4f}")
                for v in d['per_seed'].values()) +
            f"   max p = {d['max_p_over_seeds']:.4f}   "
            + ("raw p < 0.05 at every seed" if d['significant_at_every_seed_0.05'] else
               (f"UNDECIDED ({undec} seed(s) had no valid DeLong variance)" if undec
                else "not significant at every seed"))
            + (f"   Holm p = {holm['delong_maxp_holm'][key]:.4f}"
               + ("  [survives Holm]"
                  if np.isfinite(holm['delong_maxp_holm'][key])
                  and holm['delong_maxp_holm'][key] < 0.05 else "  [does NOT survive Holm]")
               if holm and key in holm.get("delong_maxp_holm", {}) else ""))
    add("")
    if pooling:
        add("-" * 96)
        add("POOLING DIAGNOSTIC -- pooled out-of-fold AUC vs the mean of the per-fold AUCs")
        add("-" * 96)
        add("Every headline AUC ranks scores POOLED across the outer folds, so k different")
        add("models' outputs are ranked against each other.  If the folds are calibrated")
        add("differently the pooled AUC is a biased estimate of the per-fold AUC (Airola")
        add("2011; Forman & Scholz 2010).  A large gap here means the pooled headline is")
        add("partly an artefact of between-fold calibration, not of discrimination.")
        add(f"  {'arm':<10}{'pooled':>9}{'mean-of-folds':>16}{'gap':>9}{'per-fold SD':>13}"
            f"{'folds':>7}")
        for a in ARMS:
            q = pooling[a]
            add(f"  {a:<10}{q['pooled_oof_auc_mean_over_seeds']:>9.3f}"
                f"{q['mean_of_per_fold_auc_mean_over_seeds']:>16.3f}"
                f"{q['pooling_gap']:>+9.3f}{q['per_fold_auc_sd']:>13.3f}"
                f"{q['n_folds_used']:>7d}")
        add("")
    add("-" * 96)
    add("IMAGE LEVEL -- DESCRIPTIVE ONLY")
    add("-" * 96)
    add("Labels are patient-level and acute pyelonephritis is frequently unilateral, so")
    add("images of a normal contralateral kidney carry a positive label.  Image-level")
    add("numbers are bounded by that label noise and MUST NOT be read as per-kidney")
    add("accuracy.  Intervals below are patient-cluster bootstraps, never naive image")
    add("bootstraps (a naive image interval here would be roughly half its honest width).")
    for a in IMAGE_ARMS:
        m = across["image_level"][a]
        b = boot_img[a]
        add(f"  {a:<6} AUC {m['auc']['mean']:.3f}+/-{m['auc']['sd']:.3f}  "
            f"cluster-bootstrap 95% CI [{b['auc']['ci_lo']:.3f}, {b['auc']['ci_hi']:.3f}]  "
            f"BalAcc {m['balanced_accuracy']['mean']:.3f}")
    add("")
    add("-" * 96)
    add("OPERATING POINT AT VALIDATION-SELECTED SENSITIVITY >= 0.90 (rule-out framing)")
    add("-" * 96)
    for a in ARMS:
        vals = [per_seed["patient_level_sens90"][s][a] for s in per_seed["patient_level_sens90"]]
        # nanmean, not mean: NPV/PPV are NaN when their denominator is empty, and a single
        # NaN seed would otherwise silently turn the whole reported average into "nan".
        add(f"  {a:<10} sens {np.nanmean([v['tpr_sensitivity'] for v in vals]):.3f}   "
            f"spec {np.nanmean([v['tnr_specificity'] for v in vals]):.3f}   "
            f"BalAcc {np.nanmean([v['balanced_accuracy'] for v in vals]):.3f}   "
            f"NPV {np.nanmean([v['npv'] for v in vals]):.3f}")
    add("")
    add("-" * 96)
    add("DEVIATIONS FROM THE PRE-SPECIFIED PROTOCOL")
    add("-" * 96)
    add("Anything a reader would have to know to reproduce or to trust these numbers, and")
    add("that differs from the approved protocol, is listed here.  A deviation recorded")
    add("only in a chat transcript is a deviation that reaches the manuscript unrecorded.")
    add("")
    add("1. Inner selection set is a SINGLE patient-level holdout"
        f" (1/{args.inner_folds} of the training")
    add("   patients), not the protocol's 4-fold inner CV.  Consequences: there is no")
    add("   pooled inner out-of-fold prediction, no median-E*, and NO REFIT on the full")
    add("   outer-training set -- the reported model is the early-stopped one, trained on")
    add(f"   {100 - round(100 / max(2, args.inner_folds))}% of the outer-training patients."
        "  Thresholds and the fusion")
    add("   layer are set on the remaining fraction (~13 patients per fold at CYCH).")
    add("   This is the largest gap from the protocol.")
    add(f"2. Repetitions: --seeds {args.seeds}"
        + ("  (protocol asks for 5; pass --seeds 5)" if args.seeds < 5 else "  (matches protocol)"))
    add("3. Augmentation is copied verbatim from fedavg_corrected.py (RandomResizedCrop")
    add("   0.8-1.0, +/-15 deg, shear 10, translate 0.1, ColorJitter incl. saturation), NOT")
    add("   the protocol's milder set, so these numbers stay comparable with the federated")
    add("   run.  Saturation jitter on grayscale-origin ultrasound is a known oddity.")
    add("4. Input pixels are the existing *_0_1_inner crops resolved through the manifest.")
    add("   The protocol's re-derivation (clean twin as pixel source, caliper inpainting,")
    add("   grayscale, 3-px erosion) does not exist on disk, so CALIPER / ANNOTATION")
    add("   RESIDUE IS STILL PRESENT IN THE TRAINING PIXELS.  Any claim that the model")
    add("   reads parenchyma rather than annotation is unsupported until that is redone.")
    add("5. Arms implemented: clinical-only, image-only CNN, fused, PCA+RBF-SVM.  NOT")
    add("   implemented: A0 prevalence floor, A4 hand-crafted texture + XGBoost, A5")
    add("   inpainting ablation, A6 YOLO end-to-end ROI, and -- most importantly -- N1,")
    add("   the LABEL-PERMUTATION NEGATIVE CONTROL that is the protocol's leakage proof")
    add("   (D2).  N1 is cheap and should be run before submission.")
    add("6. The clinical arm uses age, sex and renal measurement only; the protocol's")
    add("   age-adjusted renal residual is absent.  Age and renal size correlate at")
    add("   r ~ 0.9 here, so the clinical comparator is probably weaker than intended,")
    add("   which biases every 'imaging adds information' comparison in imaging's favour.")
    add(f"7. Patient aggregation reported = '{args.patient_agg}'.")
    if args.patient_agg == "mean":
        add("   The protocol's rule is mean-over-views-then-max-over-kidneys, which is NOT")
        add("   what the headline numbers above use.")
    else:
        add("   This is the protocol's rule.")
    add("   Both scores are stored in oof_predictions_patient.csv (score_cnn_mean,")
    add("   score_cnn_maxkidney); rerun with --patient-agg to switch the reported one.")
    add(f"8. Early stopping monitors '{args.monitor}'.")
    if args.monitor == "patient_ba":
        add("   Balanced accuracy at the validation Youden point is noisier than AUROC on")
        add("   13-53 patients; --monitor patient_auc switches it.")
    add("9. Holm-Bonferroni IS applied in code to the paired-comparison family; any")
    add("   FURTHER secondary family (protocol section 8.6) must still be adjusted by hand")
    add("   when the manuscript is written.")
    add("10. run_manifest records config, environment, seeds and elapsed time, but NOT a")
    add("    content hash of the input image directories, so a silent change to the pixels")
    add("    would not be detected by re-reading metrics.json.")
    add("")
    add("-" * 96)
    add("READING THESE NUMBERS")
    add("-" * 96)
    add("* Every number above is OUT-OF-FOLD.  Test patients were never used to choose the")
    add("  stopping epoch, any threshold, the fusion coefficients, PCA or the SVM.")
    add("* The headline claim this design can support is: does image information ADD")
    add("  discrimination beyond age, sex and renal measurement?  That is the")
    add("  cnn_vs_clinical and fused_vs_cnn rows above, not the bare AUC.")
    add("* PPV/NPV do not transfer to a general febrile-UTI population: this is an enriched")
    add("  research cohort.")
    add("* The SVM arm reuses embeddings from a CNN trained on the same inner-training")
    add("  images (2022-style pipeline).  Its feature space is therefore optimistic")
    add("  relative to a fully cross-fitted embedding; its evaluation patients are")
    add("  nevertheless strictly held out.")
    add("")
    add("-" * 96)
    add("RESIDUAL SELECTION OPTIMISM  (test folds are clean; the SELECTION set is not)")
    add("-" * 96)
    add("The stopping epoch is chosen to maximise the monitored metric on the inner-")
    add("validation patients.  The fusion layer and EVERY decision threshold are then")
    add("fitted / chosen on the CNN's scores for those SAME patients.  No test patient is")
    add("involved, so the test AUCs above are valid -- but the CNN-score distribution the")
    add("fusion layer is fitted on is optimistically selected and does not match the test")
    add("distribution, so the fused arm's coefficients and operating points carry that")
    add("optimism.  Removing it entirely needs cross-fitted CNN scores (K extra CNNs per")
    add("fold).  The size of the gap, per fold:")
    add(f"  {'seed':>4} {'fold':>4} {'inner-val AUC (selection)':>26} {'test AUC':>10} {'gap':>8}")
    gaps = []
    for rec in fold_records:
        so = rec.get("selection_optimism")
        if not so:
            continue
        v, t = so["cnn_inner_val_patient_auc_at_selected_epoch"], so["cnn_test_patient_auc"]
        if np.isfinite(v) and np.isfinite(t):
            gaps.append(v - t)
        add(f"  {rec['seed']:>4} {rec['fold']:>4} {v:>26.3f} {t:>10.3f} {v - t:>8.3f}")
    if gaps:
        add(f"  mean inner-val minus test AUC = {np.mean(gaps):+.3f}"
            "   (a large positive gap means the thresholds and fusion weights were tuned")
        add("   on a rosier picture of the CNN than the test folds provide).")
    add("=" * 96)
    path.write_text("\n".join(L), encoding="utf-8")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def _json_sanitize(o):
    """Map every non-finite float to None so metrics.json is STRICT, portable JSON.

    `json.dumps(default=...)` is only consulted for types the encoder cannot handle at
    all; a native ``float('nan')`` is handled -- it is emitted as a bare ``NaN`` token,
    which is NOT valid JSON.  Python's own ``json.load`` accepts it, so the breakage is
    invisible from inside this script, but ``jq``, ``JSON.parse``, R's ``jsonlite`` and
    ``pandas.read_json`` all reject the file.  metrics.json is the machine-readable
    deliverable ("every downstream table, figure and statistical test can be recomputed
    without retraining"), so it has to survive a non-Python reader.

    Non-finite values are routine here, not exotic: ``sd`` is NaN for every metric when
    only one seed is run, ``binary_metrics`` returns NaN for npv whenever an operating
    point (the tau_90 rule-out point in particular) predicts every patient positive, and
    ``delong_test`` deliberately returns p = NaN for a degenerate variance.  A CYCH
    ``--smoke`` run emits 100 such tokens.

    ``null`` is the right JSON spelling of "not estimable" and round-trips to NaN in
    pandas/numpy on read-back.
    """
    if isinstance(o, float):            # np.float64 subclasses float, so this catches it
        return o if np.isfinite(o) else None
    if isinstance(o, np.floating):      # np.float32 does not subclass float
        return float(o) if np.isfinite(o) else None
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return [_json_sanitize(v) for v in o.tolist()]
    if isinstance(o, dict):
        return {str(k): _json_sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_sanitize(v) for v in o]
    return o


# ==================================================================================
# 10.  Main
# ==================================================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Single-center patient-level nested CV for ultrasound kidney-inflammation detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--site", required=True, choices=["CYCH", "NCKUH"], help="Hospital to analyse.")
    p.add_argument("--out", required=True, help="Output directory.")
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Image manifest CSV.")
    p.add_argument("--keep-first-exam-only", action=argparse.BooleanOptionalAction, default=True,
                   help="Drop the second exam of the two repeat-exam patients.")
    p.add_argument("--folds", type=int, default=5, help="Outer patient-level folds.")
    p.add_argument("--seeds", type=int, default=3, help="Number of repetitions (seeds 0..n-1).")
    p.add_argument("--inner-folds", type=int, default=4,
                   help="Inner validation split = 1/inner_folds of the training patients.")
    p.add_argument("--arch", default="densenet121", choices=sorted(ARCH_HEADS))
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--monitor", default="patient_ba", choices=["patient_ba", "patient_auc"],
                   help="Inner-validation early-stopping metric (patient level).")
    p.add_argument("--patient-agg", default="mean", choices=["mean", "max_kidney"],
                   help="Image->patient score rule for the reported arms.")
    p.add_argument("--pca-components", type=int, default=50)
    p.add_argument("--svm-C", type=float, default=1.0)
    p.add_argument("--clinical-C", type=float, default=1.0)
    p.add_argument("--fusion-C", type=float, default=1.0)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--bootstrap-seed", type=int, default=20260909)
    p.add_argument("--workers", type=int, default=0,
                   help="DataLoader workers for the TRAINING loader (evaluation passes "
                        "always use 0). On Windows every worker is a spawned interpreter "
                        "that re-imports torch; measured here, 0 gives 26 s/epoch and 4 "
                        "gives 16 s/epoch but costs ~60 s of spawn per fold. 0 is not "
                        "merely 'safe', it is the right default for short folds.")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--strict-determinism", action="store_true",
                   help="Also call torch.use_deterministic_algorithms (slower, may warn).")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--smoke", action="store_true",
                   help="Fast validation run: 1 seed, 2 folds, 2 epochs, 200 bootstrap draws.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.smoke:
        args.seeds, args.folds, args.epochs = 1, 2, 2
        args.patience = max(1, min(args.patience, 2))
        args.bootstrap = min(args.bootstrap, 200)

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    device = torch.device(args.device)
    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    print("=" * 90)
    print(f"SINGLE-CENTER CV  site={args.site}  arch={args.arch}  "
          f"seeds={args.seeds} folds={args.folds}  device={device}")
    print(f"  {env['python']} | torch {env['torch']} | {env['gpu']}")
    print(f"  output -> {out_dir}")
    print("=" * 90)

    _selftest_stats()

    img_df, notes = load_cohort(Path(args.manifest), args.site, args.keep_first_exam_only)
    pat = build_patient_table(img_df)
    print(f"[data] {args.site}: {len(pat)} patients "
          f"({int((pat.label==1).sum())} positive / {int((pat.label==0).sum())} negative), "
          f"{len(img_df)} images "
          f"({notes['images_pos']} positive / {notes['images_neg']} negative)")
    if notes["dropped_second_exam_images"]:
        print(f"[data] repeat-exam rule dropped {notes['dropped_second_exam_images']} images "
              f"from patients {notes['repeat_exam_patients']}")

    seeds = list(range(args.seeds))
    pat_frames, img_frames, fold_records, assignments = [], [], [], []

    for seed in seeds:
        set_all_seeds(seed, strict=args.strict_determinism)
        splits = outer_splits(pat, args.folds, seed)
        for fold, (tr_idx, te_idx) in enumerate(splits):
            set_all_seeds(seed * 100 + fold, strict=args.strict_determinism)
            print(f"\n--- seed {seed} | outer fold {fold+1}/{args.folds} "
                  f"| train patients {len(tr_idx)} | test patients {len(te_idx)} ---")
            pf, imf, rec = run_fold(args, device, seed, fold, pat, img_df, tr_idx, te_idx)
            pat_frames.append(pf)
            img_frames.append(imf)
            fold_records.append(rec)
            # frozen per-patient assignment record (deliverable: partitions never regenerated)
            pids_map = rec.pop("_pids")
            for split_name, pid_list in pids_map.items():
                for pid in pid_list:
                    assignments.append({"site": args.site, "seed": seed, "fold": fold,
                                        "patient_id": pid, "split": split_name})
            m = binary_metrics(pf["label"].to_numpy(), pf["score_cnn"].to_numpy(),
                               pf["pred_cnn"].to_numpy())
            print(f"    best epoch {rec['cnn']['best_epoch']} | "
                  f"test-fold patient AUC: cnn {m['auc']:.3f} | "
                  f"clinical {fast_auc(pf['label'].to_numpy(), pf['score_clinical'].to_numpy()):.3f} | "
                  f"fused {fast_auc(pf['label'].to_numpy(), pf['score_fused'].to_numpy()):.3f} | "
                  f"svm {fast_auc(pf['label'].to_numpy(), pf['score_svm'].to_numpy()):.3f}")

    pat_pred = pd.concat(pat_frames, ignore_index=True)
    img_pred = pd.concat(img_frames, ignore_index=True)
    pat_pred.to_csv(out_dir / "oof_predictions_patient.csv", index=False, encoding="utf-8")
    img_pred.to_csv(out_dir / "oof_predictions_image.csv", index=False, encoding="utf-8")
    pd.DataFrame(assignments).to_csv(out_dir / "fold_assignments.csv", index=False, encoding="utf-8")

    # Every patient exactly ONCE per seed, and EVERY cohort patient present.  The
    # duplicate check alone is not enough: a patient appearing ZERO times also passes it,
    # and the metrics would then be computed on a silently truncated cohort.
    cohort_pids = set(pat["patient_id"])
    cohort_files = set(img_df["filename"])
    for s in seeds:
        got = pat_pred[pat_pred["seed"] == s]["patient_id"]
        c = got.value_counts()
        if not (c == 1).all():
            raise AssertionError(f"seed {s}: some patients appear more than once out-of-fold")
        missing = sorted(cohort_pids - set(got))
        extra = sorted(set(got) - cohort_pids)
        if missing or extra:
            raise AssertionError(
                f"seed {s}: the out-of-fold pool is not the full cohort -- "
                f"{len(missing)} missing (e.g. {missing[:5]}), {len(extra)} unexpected "
                f"(e.g. {extra[:5]}).  Metrics on a partial cohort are not comparable "
                "across seeds and would be reported as if they were.")
        gi = img_pred[img_pred["seed"] == s]["filename"]
        if not gi.is_unique or set(gi) != cohort_files:
            raise AssertionError(
                f"seed {s}: the image-level out-of-fold pool does not match the cohort "
                f"({len(set(gi))} unique of {len(cohort_files)}; duplicates: "
                f"{int(len(gi) - gi.nunique())}).")

    per_seed = per_seed_metrics(pat_pred, img_pred, seeds)
    across = {
        "patient_level": summarise_across_seeds(per_seed, "patient_level", ARMS),
        "patient_level_sens90": summarise_across_seeds(per_seed, "patient_level_sens90", ARMS),
        "image_level": summarise_across_seeds(per_seed, "image_level", IMAGE_ARMS),
    }

    print("\n[stats] cluster bootstrap ...")
    store = build_oof_store(pat_pred, img_pred, seeds)
    boot = cluster_bootstrap(store, seeds, args.bootstrap, args.bootstrap_seed, "patient")
    boot_img = cluster_bootstrap(store, seeds, args.bootstrap, args.bootstrap_seed + 1, "image")
    boot_per_seed = {str(s): cluster_bootstrap(store, seeds, args.bootstrap,
                                               args.bootstrap_seed + 10 + s, "patient",
                                               fixed_seed=s) for s in seeds}
    boot_delta = {
        "cnn_vs_clinical": paired_bootstrap_delta(store, seeds, "cnn", "clinical",
                                                  args.bootstrap, args.bootstrap_seed + 100),
        "fused_vs_cnn": paired_bootstrap_delta(store, seeds, "fused", "cnn",
                                               args.bootstrap, args.bootstrap_seed + 101),
        "cnn_vs_svm": paired_bootstrap_delta(store, seeds, "cnn", "svm",
                                             args.bootstrap, args.bootstrap_seed + 102),
        "fused_vs_clinical": paired_bootstrap_delta(store, seeds, "fused", "clinical",
                                                    args.bootstrap, args.bootstrap_seed + 103),
    }
    delong = run_delong_comparisons(pat_pred, seeds)

    # Family-wise error control over the paired-comparison family (four comparisons).
    holm = {
        "method": "Holm-Bonferroni step-down; family = the paired AUC comparisons reported",
        "family_bootstrap": sorted(boot_delta),
        "family_size_bootstrap": int(len(boot_delta)),
        "bootstrap_p_holm": holm_bonferroni(
            {k: v["p_bootstrap"] for k, v in boot_delta.items()}),
        "family_delong": sorted(delong),
        "delong_maxp_holm": holm_bonferroni(
            {k: v["max_p_over_seeds"] for k, v in delong.items()}),
        "note": "Claim a comparison only on its Holm-adjusted p; raw p is descriptive. "
                "Unadjusted, four comparisons at alpha=0.05 give FWER ~ 0.19.",
    }
    pooling = pooled_vs_per_fold_auc(pat_pred, seeds, ARMS)

    elapsed = time.time() - t0
    metrics = {
        "run_manifest": {
            "site": args.site, "out_dir": str(out_dir), "config": vars(args),
            "environment": env, "elapsed_sec": elapsed,
            "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
            "manifest_file": str(Path(args.manifest).resolve()),
        },
        "cohort": notes,
        "cohort_patients": {
            "n": int(len(pat)), "n_pos": int((pat.label == 1).sum()),
            "n_neg": int((pat.label == 0).sum()),
            "prevalence": float((pat.label == 1).mean()),
        },
        "fold_records": fold_records,
        "per_seed_metrics": per_seed,
        "across_seeds": across,
        "bootstrap_patient_level": boot,
        "bootstrap_image_level_cluster": boot_img,
        "bootstrap_patient_level_per_seed": boot_per_seed,
        "paired_bootstrap_auc_difference": boot_delta,
        "delong": delong,
        "multiplicity_holm": holm,
        "pooled_vs_per_fold_auc": pooling,
        "notes": {
            "unit_of_analysis": "patient (labels are patient-level; APN is often unilateral)",
            "image_level_metrics": "descriptive only; never to be read as per-kidney accuracy",
            "leakage": "train/val/test patient sets asserted disjoint in every fold",
            "selection": "stopping epoch, thresholds, fusion coefficients chosen on the inner "
                         "validation patients only; test folds scored once",
            "primary_inference": "paired patient cluster bootstrap on the AUC difference "
                                 "(paired_bootstrap_auc_difference)",
            "delong_caveat": "DeLong is SECONDARY and anti-conservative here: its variance "
                             "assumes two fixed models on independent observations, but "
                             "these predictions are pooled across outer folds, so several "
                             "models produced them and each patient's data trained the "
                             "models that scored the others.",
            "bootstrap_p_resolution": "p is floored at 2/(B+1); a value equal to the floor "
                                      "means 'below the resolution of B draws', not zero.",
            "residual_selection_optimism": "the fusion layer and every threshold are fitted "
                                           "on CNN scores for the same inner-validation "
                                           "patients that chose the stopping epoch; see "
                                           "fold_records[*].selection_optimism for the "
                                           "inner-val-minus-test AUC gap.",
        },
    }
    # allow_nan=False is a tripwire, not decoration: if _json_sanitize ever misses a
    # non-finite value the run fails loudly here instead of silently shipping a
    # metrics.json that only Python can read.
    (out_dir / "metrics.json").write_text(
        json.dumps(_json_sanitize(metrics), indent=2, ensure_ascii=False,
                   default=_json_default, allow_nan=False), encoding="utf-8")

    make_figures(pat_pred, img_pred, across, out_dir, args.site, seeds[0])
    write_summary(out_dir / "summary.txt", args, notes, fold_records, per_seed, across,
                  boot, boot_img, boot_per_seed, delong, boot_delta, elapsed, env,
                  holm=holm, pooling=pooling)

    print("\n" + "=" * 90)
    print(f"PATIENT-LEVEL out-of-fold results ({args.site}), mean +/- SD over {len(seeds)} seed(s)")
    print("=" * 90)
    print(f"{'arm':<10}{'AUC (mean+/-SD)':>18}{'AUC 95% CI':>20}{'BalAcc':>18}{'Sens':>9}{'Spec':>9}")
    for a in ARMS:
        m = across["patient_level"][a]
        b = boot[a]
        print(f"{a:<10}"
              + f"{m['auc']['mean']:.3f}+/-{m['auc']['sd']:.3f}".rjust(18)
              + f"[{b['auc']['ci_lo']:.3f}, {b['auc']['ci_hi']:.3f}]".rjust(20)  # CI around point_estimate
              + f"{m['balanced_accuracy']['mean']:.3f}+/-{m['balanced_accuracy']['sd']:.3f}".rjust(18)
              + f"{m['tpr_sensitivity']['mean']:.3f}".rjust(9)
              + f"{m['tnr_specificity']['mean']:.3f}".rjust(9))
    print("-" * 90)
    print("PRIMARY paired patient cluster bootstrap on dAUC (DeLong below is secondary and")
    print("anti-conservative: the predictions are pooled across outer folds):")
    for k, d in boot_delta.items():
        floor = d.get("p_bootstrap_floor", float("nan"))
        at_floor = np.isfinite(floor) and d["p_bootstrap"] <= floor + 1e-12
        hp = holm["bootstrap_p_holm"].get(k, float("nan"))
        print(f"  boot   {k:<18} dAUC {d.get('delta_observed', d['delta_mean']):+.4f}  "
              f"95% CI [{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}]  p "
              + (f"< {floor:.4f}" if at_floor else f"= {d['p_bootstrap']:.4f}")
              + f"  Holm p = {hp:.4f}"
              + ("  *" if np.isfinite(hp) and hp < 0.05 else ""))
    for k, d in delong.items():
        undec = d.get("n_seeds_without_valid_p", 0)
        hp = holm["delong_maxp_holm"].get(k, float("nan"))
        # "significant at every seed" is an UNADJUSTED verdict over one comparison.  Printed
        # on its own next to a Holm p above 0.05 it is an over-claim, so print both.
        raw_ok = d['significant_at_every_seed_0.05']
        print(f"  DeLong {k:<18} mean dAUC {d['mean_delta']:+.4f}  max p over seeds "
              f"{d['max_p_over_seeds']:.4f}  Holm p = {hp:.4f}  "
              + (f"UNDECIDED ({undec} seed(s) degenerate)" if undec else
                 ("SIGNIFICANT after Holm" if raw_ok and np.isfinite(hp) and hp < 0.05
                  else ("raw p<0.05 at every seed but does NOT survive Holm" if raw_ok
                        else "ns"))))
    print("-" * 90)
    print("Pooling diagnostic (pooled-OOF AUC minus mean-of-per-fold AUC; a large gap means")
    print("the headline is partly a between-fold calibration artefact):")
    print("  " + "   ".join(
        f"{a} {pooling[a]['pooled_oof_auc_mean_over_seeds']:.3f} vs "
        f"{pooling[a]['mean_of_per_fold_auc_mean_over_seeds']:.3f} "
        f"(gap {pooling[a]['pooling_gap']:+.3f})" for a in ARMS))
    print("-" * 90)
    print(f"Wrote: {out_dir/'metrics.json'}")
    print(f"       {out_dir/'summary.txt'}")
    print(f"       {out_dir/'oof_predictions_patient.csv'}")
    print(f"       {out_dir/'oof_predictions_image.csv'}")
    print(f"       {out_dir/'fold_assignments.csv'}")
    print(f"       roc_patient.png / pr_patient.png / roc_image.png / pr_image.png (600 dpi)")
    print(f"Elapsed: {elapsed/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
