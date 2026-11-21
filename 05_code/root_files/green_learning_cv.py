"""
Green-learning (Saab / PixelHop++ / discriminant feature test) arm for the
single-center kidney-inflammation experiment.

WHAT THIS ARM TESTS
-------------------
The published single-center result is near chance from pixels: DenseNet121 reaches
patient-level AUC 0.544 [0.470, 0.621] at NCKUH and 0.526 [0.386, 0.654] at CYCH,
against a clinical-variable baseline of 0.601 / 0.635.  Two controls already bound
the interpretation of that number.  The permuted-label control returns 0.461
[0.345, 0.572] -- the harness does not leak.  The hospital-of-origin control returns
0.992 [0.984, 0.998] after a single epoch -- these pixels ARE learnable, and the
harness recovers a real signal when one is present.  So the missing inflammation
signal is a property of the data, not of the analysis.

The clinical author asked whether green learning would change that.  Green learning
(C.-C. Jay Kuo's line: the Saab transform, PixelHop / PixelHop++ / IPHop, successive
subspace learning, the discriminant feature test) is a feed-forward,
back-propagation-free representation.  Its convolution filters are channel-wise
PCA eigenvectors computed in closed form from the training patches; between hops it
max-pools; afterwards it selects features by a supervised entropy criterion and
reads them out with a classical classifier.  Its stated strength is exactly this
regime -- a few hundred independent training units, where a 7-million-parameter
DenseNet memorises instead of generalising.  It is a genuinely different inductive
bias, not a re-parameterisation of the CNN, and it is cheap: no GPU, no epochs, no
early stopping.  It is also NOT what the 2022 manuscript used; that pipeline was
YOLOv5 detection + a back-propagated triplet CNN + PCA + SVM, i.e. ordinary transfer
learning.

HOW TO READ A NEAR-CHANCE RESULT FROM IT
----------------------------------------
Be honest about what this arm can and cannot do.  A different feature extractor
cannot manufacture signal that is absent from the data.  Given the two controls
already in hand, the readings are:

  green ~ 0.50-0.55, i.e. indistinguishable from the CNN
      -> STRONG CORROBORATION of the negative result.  Two extractors with opposite
         inductive biases (millions of back-propagated parameters vs. a closed-form
         linear multiscale PCA) fail identically on data where the same harness
         recovers hospital-of-origin at 0.99.  The most parsimonious explanation is
         that the cropped ROI does not carry the inflammation label, i.e. the label,
         the crop, or the reference standard is the limiting factor -- not the model
         family.  This is the expected outcome and it is a result, not a failed run.

  green clearly above the CNN and above the `geom` confound arm
      -> a real, previously missed representation effect.  It would have to survive
         the permuted-label control (--permute-labels must come back at 0.50) and it
         would have to NOT be explained by the `geom` arm before anyone believes it.

  green above the CNN but at or below `geom`
      -> a confound detector, not a tissue-signal detector.  Crop height alone is
         worth patient AUC 0.600 at NCKUH -- the same number as the entire clinical
         model -- and it is uncorrelated there with kidney size (r = 0.007), so it is
         an acquisition/preset/era artifact.  Colour-overlay fraction is worth 0.569
         and mean luma 0.589 at CYCH.  The mandatory `geom` companion arm (native W,
         H, W/H, colour-overlay fraction, mean luma) prices those confounds on the
         identical folds, and the anisotropic stretch + Telea inpaint in the
         preprocessing are there to keep them OUT of the green features.  If green
         does not beat geom, that is reported in the same sentence as the green
         number.

WHAT IS FITTED WHERE (the one defect that would silently inflate this arm)
-------------------------------------------------------------------------
Eight things are fitted here and only one of them looks supervised.  The Saab means,
eigenvectors and biases, the energy ranking that decides which of 1200 hop-3 nodes
survive, the 1st/99th percentile clips, and the StandardScaler all feel like
"harmless preprocessing", and the usual rationalisation -- "it never sees a label,
so it cannot leak" -- is wrong when a fold's test set is 20-53 patients whose pixel
statistics would then shape the basis.  The discriminant feature test is outright
supervised.  So, in this script:

  * EVERYTHING is fitted on the INNER-TRAINING images only: every Saab filter bank at
    every hop, the per-hop bias, the energy ranking and surviving-node lists, the DFT
    clips and ranking, the feature scaler and the classifier.  This is stricter than
    the specification's "fit the Saab tree on the outer-training images"; it is used
    because it is also what the CNN arm does (the CNN never refits on outer-train
    either), so the two arms see identical data and the comparison is like-for-like.
  * The INNER-VALIDATION patients select the regularisation constant C and both
    operating thresholds, and nothing else.
  * The TEST fold is transformed and scored exactly once.
  * The extractor is a single object with fit(imgs, paths) / transform(imgs).  fit takes
    NO labels at all -- the whole Saab tree, including the energy ranking, is
    unsupervised -- and it records the frozenset of paths it saw; transform asserts that
    the fitted state is byte-identical before and after it runs, and every fold asserts
    both that the fitting paths are disjoint from the validation and test paths and that
    cache[img_idx] really addresses those paths.
  * --permute-labels runs this identical code path on patient labels permuted before
    any split is drawn.  It must return pooled patient AUC ~ 0.50 with a CI covering
    0.50.  That single run catches every variant of the leak above.

TWO DELIBERATE DEPARTURES FROM THE WRITTEN SPECIFICATION (both recorded in summary.txt)
--------------------------------------------------------------------------------------
1. The extractor and the classifier are fitted on the INNER-TRAINING images and are
   never refitted on the full outer-training set.  Reason above: the CNN arm does not
   refit either, so this keeps the comparison like-for-like, and it is the stricter of
   the two options.  Consequence: the DFT cut K = min(300, n_train_images/4) is
   computed on the inner-training image count, so K is about 154 at NCKUH and 52 at
   CYCH rather than the specification's 206 / 70.
2. The 1e-6 cumulative-energy floor is replaced by a true numerical-null guard
   (1e-12); the rank caps are the operative selector.  The specification's own stated
   node counts (4233 features) and its 1e-6 floor are inconsistent on this data,
   because the adjusted bias rides on every AC channel and pushes deep AC nodes to
   cumulative energy ~1e-8.  At 1e-6 the arm would carry 2244 features and would have
   amputated exactly the high-frequency speckle channels the rank rule exists to
   protect.  --energy-floor 1e-6 reproduces the literal text.

Usage
    python green_learning_cv.py --site CYCH  --out results_green/CYCH
    python green_learning_cv.py --site NCKUH --out results_green/NCKUH
    python green_learning_cv.py --site CYCH  --permute-labels --out results_green/permute_CYCH
    python green_learning_cv.py --site CYCH  --smoke --out results_green/_smoke

torch is never used for computation here; it is pulled in transitively because the
harness module imports it.  Everything below is numpy / scipy / sklearn / OpenCV.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import single_center_cv as S  # noqa: E402

# ----------------------------------------------------------------------------------
# Arms.  `green`  = Saab/PixelHop++ features -> DFT -> L2 logistic regression.
#        `fused`  = logit(green patient score) + age/sex/renal measurement, the fusion
#                   layer fitted on the inner-validation patients exactly as the CNN
#                   arm's fused model is, so the two fused arms are comparable.
#        `geom`   = the pre-declared confound arm: native W, H, W/H, colour-overlay
#                   fraction, mean luma.  This is what the acquisition artifacts alone
#                   are worth on the identical folds.
#        `clinical` = age, sex, renal measurement.  Recomputed here (rather than read
#                   from the CNN run) so this script is self-contained; when the folds
#                   are replayed it must reproduce the CNN run's clinical scores, and
#                   that agreement is asserted as a tripwire.
# ----------------------------------------------------------------------------------
GL_PATIENT_ARMS = ["green", "fused", "geom", "clinical"]
GL_IMAGE_ARMS = ["green", "geom"]
GL_ARM_LABELS = {
    "green": "Green learning (Saab/PixelHop++ + DFT + L2-LR)",
    "fused": "Fused (green probability + clinical)",
    "geom": "Geometry/overlay confound control",
    "clinical": "Clinical-only (LR: age, sex, renal measurement)",
    "cnn": "Image-only CNN (DenseNet121, from the published run)",
}
GL_ARM_COLORS = {
    "green": "#2E8B57", "fused": "#55A868", "geom": "#B8860B",
    "clinical": "#4C72B0", "cnn": "#C44E52",
}
GEOM_FEATURES = ["native_w", "native_h", "aspect_w_over_h", "overlay_frac", "mean_luma"]
CLINICAL_FEATURES = list(S.CLINICAL_FEATURES)   # age_years, gender_code, renal_measure_cm


# ==================================================================================
# 0.  Configuration of the green-learning architecture
# ==================================================================================
@dataclass
class GLConfig:
    """Every architectural constant.  Fixed a priori -- none of these is tuned on data.

    in_h / in_w = 96 x 160.  The measured median native crop is 153 x 89 (aspect 1.69),
    so the median image is resampled by under 8 % in either axis: speckle -- the
    physiological texture of interest -- is neither blurred away by a heavy downsample
    nor invented by an upsample, and 160/96 = 1.667 is the measured median aspect so the
    median image is barely deformed.  The resize is a STRETCH, not a letterbox, and that
    is forced: letterboxing writes native W and H into the pad geometry, and crop height
    alone is worth patient AUC 0.600 at NCKUH.
    """
    in_h: int = 96
    in_w: int = 160
    kernel: int = 5
    pool: int = 2
    # (n_feature_nodes, n_propagated_nodes) per hop; hop 3 is terminal.
    caps: Tuple[Tuple[int, int], ...] = ((25, 16), (96, 48), (128, 0))
    # NUMERICAL-NULL GUARD, not a selection rule.  The specification writes this floor as
    # 1e-6 on the CUMULATIVE energy AND states the resulting node counts (400 hop-2
    # candidates -> 96 features / 48 propagated, 1200 hop-3 candidates -> 128 features,
    # 4233 features in total).  Measured on the real cohort those two statements are
    # inconsistent: because the adjusted bias d1 = 4.96 rides on every hop-1 AC channel,
    # a hop-1 AC channel's patch energy is ~25 * 4.96^2 = 615 and its local DC child takes
    # essentially all of it, so its AC grandchildren sit at cumulative energy ~1e-8.  A
    # 1e-6 floor therefore culls hop 2 from 400 candidates to 41 and hop 3 to 66 features
    # (2244 features instead of 4233) -- i.e. it amputates exactly the high-frequency
    # speckle channels that section 2.4 introduces the rank rule to protect, and it
    # destroys the "exactly known dimensionality" the rule exists to give.  The rank is
    # therefore the operative selector and the floor is set to a true numerical null.
    # --energy-floor 1e-6 reproduces the literal text; the diagnostics record how many
    # nodes each setting would have dropped.
    energy_floor: float = 1e-12
    patch_samples: Tuple[int, ...] = (500_000, 200_000, 200_000)
    grid_rows: int = 2
    grid_cols: int = 4
    dft_bins: int = 32
    dft_min_weight_frac: float = 0.02
    dft_min_raw: int = 8
    dft_k_cap: int = 300
    dft_k_divisor: int = 4
    chunk: int = 64
    max_neg_frac: float = 1e-4

    @property
    def n_hops(self) -> int:
        return len(self.caps)

    def hop_shapes(self) -> List[Dict[str, int]]:
        """Conv/pool geometry, computed rather than asserted, so --smoke sizes stay honest."""
        out, h, w = [], self.in_h, self.in_w
        for _ in range(self.n_hops):
            ch, cw = h - self.kernel + 1, w - self.kernel + 1
            ph, pw = ch // self.pool, cw // self.pool
            out.append({"conv_h": ch, "conv_w": cw, "pool_h": ph, "pool_w": pw})
            h, w = ph, pw
        return out

    def receptive_fields(self) -> List[int]:
        rf, stride, out = 1, 1, []
        for _ in range(self.n_hops):
            rf = rf + (self.kernel - 1) * stride
            out.append(rf)
            rf = rf + (self.pool - 1) * stride
            stride *= self.pool
        return out


def smoke_config() -> GLConfig:
    """Small but structurally identical: same code path, ~2 minutes."""
    return GLConfig(in_h=48, in_w=80, caps=((25, 8), (24, 12), (32, 0)),
                    patch_samples=(100_000, 50_000, 50_000), chunk=32)


# ==================================================================================
# 1.  Preprocessing (per image, data-independent, fits nothing, cannot leak)
# ==================================================================================
def preprocess_one(abs_path: str, in_h: int, in_w: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (canvas float32 in [0,1] of shape (in_h, in_w), geom features).

    93 % of these crops carry burned-in colour overlay (calipers / annotation), five
    times more of it at CYCH than at NCKUH.  Plain BT.601 luma maps a saturated red
    caliper to ~76 and green to ~150, so the overlay would survive the grayscale
    conversion as a bright or dark blob and the first Saab hop -- a PCA on raw pixels --
    would lock onto it.  Telea inpainting removes it first.  Everything here is a
    per-image operation: nothing is estimated across images, so nothing can leak.
    """
    bgr = cv2.imread(abs_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"cv2.imread returned None for {abs_path}")
    native_h, native_w = int(bgr.shape[0]), int(bgr.shape[1])
    mx = bgr.max(2).astype(np.int16)
    mn = bgr.min(2).astype(np.int16)
    mask = ((mx - mn) > 8).astype(np.uint8)
    overlay_frac = float(mask.mean())
    if mask.any():
        bgr = cv2.inpaint(bgr, mask, 3, cv2.INPAINT_TELEA)
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mean_luma = float(g.mean()) / 255.0
    interp = cv2.INTER_AREA if (native_w >= in_w and native_h >= in_h) else cv2.INTER_LINEAR
    x = cv2.resize(g, (in_w, in_h), interpolation=interp).astype(np.float32) / 255.0
    geom = np.array([native_w, native_h, native_w / max(1.0, float(native_h)),
                     overlay_frac, mean_luma], dtype=np.float64)
    return x, geom


def build_image_cache(img_df: pd.DataFrame, cfg: GLConfig, quiet: bool = False
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Preprocess the whole site once.  Returns (canvases, geom, paths).

    The cache is addressed downstream by `img_idx`, NOT by DataFrame row position, and
    every split is selected as `cache[rows["img_idx"]]`.  So fill it by img_idx here
    rather than by enumeration order: if img_df were ever re-sorted or filtered between
    load_cohort and this call, enumeration order would silently stop matching img_idx and
    the Saab bank would be fitted on a mixture that includes test images -- while every
    path-level guard in run_one_fold still passed, because the PATHS handed to fit() would
    still be the inner-training ones.  The returned `paths` array is in cache-row order and
    is what run_one_fold checks the split indices against.
    """
    n = len(img_df)
    arr = np.empty((n, cfg.in_h, cfg.in_w), dtype=np.float32)
    geo = np.empty((n, len(GEOM_FEATURES)), dtype=np.float64)
    paths = np.empty(n, dtype=object)
    idx = np.asarray(img_df["img_idx"].to_numpy(), dtype=np.int64)
    S._require(np.array_equal(np.sort(idx), np.arange(n)),
               "img_idx is not a permutation of 0..n-1, but the image cache is addressed "
               "by img_idx; refusing to build a cache whose rows cannot be trusted.")
    it = zip(idx.tolist(), img_df["abs_path"].tolist())
    if not quiet:
        it = tqdm(it, total=n, desc="preprocess", unit="img", leave=False)
    for k, p in it:
        arr[k], geo[k] = preprocess_one(p, cfg.in_h, cfg.in_w)
        paths[k] = p
    return arr, geo, paths


# ==================================================================================
# 2.  The Saab transform
# ==================================================================================
@dataclass
class SaabUnit:
    """One channel-wise Saab unit: N=k^2 inputs -> 1 DC + (N-1) AC responses."""
    mu: np.ndarray        # (N,)   mean of the AC residuals over training patches
    W: np.ndarray         # (N, N) columns = [a0 (DC, fixed), a1..a_{N-1} (AC, fitted)]
    lam: np.ndarray       # (N-1,) AC eigenvalues, descending
    energy: np.ndarray    # (N,)   local fractional energy, sums to 1
    n_patches: int


def _sign_canonical(V: np.ndarray) -> np.ndarray:
    """Pin the sign of every eigenvector so the channel order and sign are reproducible
    across BLAS/LAPACK builds: the entry of largest magnitude is made positive."""
    V = np.array(V, dtype=np.float64, copy=True)
    for j in range(V.shape[1]):
        i = int(np.argmax(np.abs(V[:, j])))
        if V[i, j] < 0:
            V[:, j] *= -1.0
    return V


def fit_saab_unit(P: np.ndarray) -> SaabUnit:
    """Fit one Saab unit from training patches P of shape (M, N).

    DC anchor a0 = 1/sqrt(N) is FIXED a priori, never estimated.  Residuals
    r = x - (a0'x) a0 all lie in 1-perp, so range(C) subset 1-perp and every AC
    eigenvector is automatically orthogonal to a0; that is what makes {a0, a1..a_{N-1}}
    a complete orthonormal basis and gives the Parseval identity used for the energy
    definition.  Numerically the DC direction is an exact null vector of C, so it is
    deflated to a uniquely extreme eigenvalue before eigh rather than being trusted to
    sort itself to the bottom of a cluster of near-zero eigenvalues.

    eigh, never eig: eig on a symmetric matrix returns unsorted and possibly complex
    eigenvectors, which would scramble the channel order between fit and transform.
    """
    P = np.asarray(P, dtype=np.float64)
    M, N = P.shape
    S._require(M > N, f"Saab unit needs more patches ({M}) than dimensions ({N})")
    a0 = np.full(N, 1.0 / np.sqrt(N), dtype=np.float64)
    y0 = P @ a0
    R = P - np.outer(y0, a0)
    mu = R.mean(axis=0)
    Rc = R - mu
    C = (Rc.T @ Rc) / (M - 1)
    C = 0.5 * (C + C.T)

    P0 = np.eye(N) - np.outer(a0, a0)
    big = 1e3 * (float(np.trace(C)) + 1.0)
    Cd = P0 @ C @ P0 - big * np.outer(a0, a0)
    Cd = 0.5 * (Cd + Cd.T)
    evals, evecs = np.linalg.eigh(Cd)          # ascending
    order = np.argsort(evals)[::-1]            # descending
    evals, evecs = evals[order], evecs[:, order]
    A = _sign_canonical(evecs[:, :N - 1])
    lam = evals[:N - 1]

    W = np.column_stack([a0, A])
    S._require(float(np.max(np.abs(A.T @ a0))) < 1e-6,
               "Saab AC filters are not orthogonal to the DC anchor")
    S._require(float(np.max(np.abs(W.T @ W - np.eye(N)))) < 1e-6,
               "Saab basis is not orthonormal; Parseval-based energy would be meaningless")

    Y = P @ W
    E = (Y ** 2).mean(axis=0)
    parseval = float((P ** 2).sum(axis=1).mean())
    S._require(abs(float(E.sum()) - parseval) <= 1e-6 * max(1.0, parseval),
               f"Parseval violated: sum(E)={E.sum():.6g} vs mean||x||^2={parseval:.6g}")
    e = E / max(float(E.sum()), 1e-300)
    return SaabUnit(mu=mu, W=W.astype(np.float64), lam=lam, energy=e, n_patches=int(M))


# ==================================================================================
# 3.  Hop mechanics (im2col + batched matmul; no ReLU -- see below)
# ==================================================================================
def _conv_hop(x: np.ndarray, Wm: np.ndarray, bv: np.ndarray) -> np.ndarray:
    """x (n, C, H, W) -> (n, C*Nk, Ho, Wo).  Channel c of x is convolved with unit c's
    own filters, so output channel c*Nk + j is child j of parent c.

    No ReLU, deliberately.  The adjusted bias d is chosen so every training response is
    non-negative, which makes ReLU provably the identity on training data; applying it
    at TEST time on a patch whose norm exceeds d would then apply a nonlinearity
    inconsistently between fit and transform.  The honest consequence: the hop cascade
    is exactly a linear multiscale PCA composed with max-pooling -- max-pool is the only
    nonlinearity in the representation.  frac_negative_ac_response is logged to prove
    the ReLU really would have been inactive.
    """
    n, C, H, W = x.shape
    k = int(round(np.sqrt(Wm.shape[1])))
    Ho, Wo = H - k + 1, W - k + 1
    sw = sliding_window_view(x, (k, k), axis=(2, 3))              # (n,C,Ho,Wo,k,k)
    Pm = np.ascontiguousarray(sw.transpose(1, 0, 2, 3, 4, 5)).reshape(C, n * Ho * Wo, k * k)
    Y = np.matmul(Pm, Wm) + bv[:, None, :]                        # (C, n*Ho*Wo, Nk)
    del Pm
    Nk = Y.shape[2]
    Y = Y.reshape(C, n, Ho, Wo, Nk).transpose(1, 0, 4, 2, 3)
    return np.ascontiguousarray(Y).reshape(n, C * Nk, Ho, Wo)


def _pool2(y: np.ndarray, p: int) -> np.ndarray:
    n, C, H, W = y.shape
    H2, W2 = H // p, W // p
    S._require(H2 > 0 and W2 > 0, f"pooling {H}x{W} by {p} leaves an empty map")
    return y[:, :, :H2 * p, :W2 * p].reshape(n, C, H2, p, W2, p).max(axis=(3, 5))


def _max_patch_norm(x: np.ndarray, k: int, chunk: int = 32) -> float:
    """max over ALL training patches of ||x||_2, exactly (integral-image box sum).

    d_h = max_m ||x^(m)|| is the Cauchy-Schwarz bound |a'x| <= ||a|| ||x|| = ||x||, so
    this d guarantees every training AC response is non-negative.  Estimating it from a
    patch SUBSAMPLE would understate it and let a handful of training responses go
    negative, which is exactly the symptom the frac_negative tripwire is meant to catch.
    """
    best = 0.0
    n = x.shape[0]
    for i in range(0, n, chunk):
        xs = x[i:i + chunk].astype(np.float64) ** 2
        c = np.cumsum(np.cumsum(xs, axis=2), axis=3)
        c = np.pad(c, ((0, 0), (0, 0), (1, 0), (1, 0)))
        box = (c[:, :, k:, k:] - c[:, :, :-k, k:] - c[:, :, k:, :-k] + c[:, :, :-k, :-k])
        m = float(box.max()) if box.size else 0.0
        best = max(best, m)
    return float(np.sqrt(max(best, 0.0)))


def _sample_patches(x: np.ndarray, c: int, n_samples: int, k: int,
                    rng: np.random.Generator) -> np.ndarray:
    """Random patches of channel c, WITH replacement (a 25x25 covariance from 200k draws
    is 8000 samples per dimension; sampling without replacement would need a 15M-element
    permutation for no statistical gain)."""
    sw = sliding_window_view(x[:, c], (k, k), axis=(1, 2))        # (n,Ho,Wo,k,k)
    n, Ho, Wo = sw.shape[0], sw.shape[1], sw.shape[2]
    total = n * Ho * Wo
    if total <= n_samples:
        sel = np.arange(total)
    else:
        sel = rng.integers(0, total, size=n_samples)
    ii, rem = np.divmod(sel, Ho * Wo)
    yy, xx = np.divmod(rem, Wo)
    return sw[ii, yy, xx].reshape(-1, k * k).astype(np.float64)


def _cell_slices(n: int, parts: int) -> List[slice]:
    idx = np.array_split(np.arange(n), parts)
    S._require(all(len(p) for p in idx),
               f"spatial pooling grid {parts} is finer than the map dimension {n}")
    return [slice(int(p[0]), int(p[-1]) + 1) for p in idx]


def _cell_stats(y: np.ndarray, rs: List[slice], cs: List[slice]) -> np.ndarray:
    """(n, C, H, W) -> (n, C*17): per 2x4 cell mean and std, plus one global max per
    channel.  Accumulated in float64: hop-3 responses sit near 125 with small spatial
    deviations, and a float32 one-pass std there loses most of the signal."""
    n, C = y.shape[0], y.shape[1]
    out = np.empty((n, C, len(rs) * len(cs) * 2 + 1), dtype=np.float32)
    j = 0
    for r in rs:
        for c in cs:
            blk = y[:, :, r, c]
            out[:, :, j] = blk.mean(axis=(2, 3), dtype=np.float64); j += 1
            out[:, :, j] = blk.std(axis=(2, 3), dtype=np.float64); j += 1
    out[:, :, j] = y.max(axis=(2, 3))
    return out.reshape(n, C * out.shape[2])


# ==================================================================================
# 4.  The PixelHop++ extractor:  fit(train) / transform(any)
# ==================================================================================
@dataclass
class HopFit:
    units: List[SaabUnit]                      # one per input channel of this hop
    d: float                                   # single scalar bias shared by the hop
    feat_nodes: List[Tuple[int, int]]          # (parent, child) kept as FEATURES, ranked
    feat_energy: List[float]                   # cumulative energy of each kept node
    prop_nodes: List[Tuple[int, int]]          # (parent, child) PROPAGATED, ranked
    prop_energy: List[float]
    n_candidates: int
    pool_h: int
    pool_w: int


class GreenExtractor:
    """Saab / PixelHop++ feature extractor.

    fit(imgs, paths) -- every fitted object (filters, biases, energy ranking, surviving
    node lists) comes from these images and nothing else.  transform(imgs) indexes into
    the stored node lists and NEVER recomputes a ranking; a fingerprint of the whole
    fitted state is compared before and after every transform, so a refit hidden inside
    transform raises instead of silently inflating the AUC.
    """

    def __init__(self, cfg: GLConfig, seed: int):
        self.cfg = cfg
        self.seed = int(seed)
        self.hops: List[HopFit] = []
        self.feature_names: List[str] = []
        self.fit_paths: frozenset = frozenset()
        self._fitted = False
        self._fingerprint: str = ""
        self.diag: Dict[str, object] = {}

    # ---------------------------------------------------------------- internals --
    def _fingerprint_state(self) -> str:
        h = hashlib.sha256()
        for hop in self.hops:
            h.update(np.asarray([hop.d], dtype=np.float64).tobytes())
            h.update(np.asarray(hop.feat_nodes, dtype=np.int64).tobytes())
            h.update(np.asarray(hop.prop_nodes, dtype=np.int64).tobytes())
            for u in hop.units:
                h.update(np.ascontiguousarray(u.W, dtype=np.float64).tobytes())
                h.update(np.ascontiguousarray(u.energy, dtype=np.float64).tobytes())
        h.update("|".join(self.feature_names).encode("utf-8"))
        return h.hexdigest()

    def _hop_matrices(self, hop: HopFit) -> Tuple[np.ndarray, np.ndarray]:
        Wm = np.stack([np.ascontiguousarray(u.W, dtype=np.float32) for u in hop.units], 0)
        Nk = Wm.shape[2]
        bv = np.full((Wm.shape[0], Nk), np.float32(hop.d))
        bv[:, 0] = 0.0                    # the DC channel carries no bias
        return Wm, bv

    @staticmethod
    def _rank_nodes(cum: np.ndarray, parents: np.ndarray, children: np.ndarray,
                    n_feat: int, n_prop: int, floor: float):
        """Rank candidates by cumulative energy descending, ties by (parent, child).
        The surviving index list is a FITTED OBJECT; transform only indexes into it."""
        keep = cum >= floor
        cum, parents, children = cum[keep], parents[keep], children[keep]
        order = np.lexsort((children, parents, -cum))
        nf = int(min(n_feat, len(order)))
        npz = int(min(n_prop, nf))
        feat = [(int(parents[i]), int(children[i])) for i in order[:nf]]
        feat_e = [float(cum[i]) for i in order[:nf]]
        prop = [(int(parents[i]), int(children[i])) for i in order[:npz]]
        prop_e = [float(cum[i]) for i in order[:npz]]
        return feat, feat_e, prop, prop_e, int(keep.sum())

    # --------------------------------------------------------------------- fit --
    def fit(self, imgs: np.ndarray, paths: Sequence[str]) -> "GreenExtractor":
        cfg = self.cfg
        rng = np.random.default_rng(self.seed)
        t0 = time.time()
        self.hops, self.feature_names = [], []
        cur = np.ascontiguousarray(imgs[:, None, :, :], dtype=np.float32)
        cum_parent = np.array([1.0])                      # cumulative energy of the input
        diag_hops = []

        for h in range(cfg.n_hops):
            n_feat, n_prop = cfg.caps[h]
            C = cur.shape[1]
            d = _max_patch_norm(cur, cfg.kernel, chunk=max(8, cfg.chunk // 2))
            units: List[SaabUnit] = []
            for c in range(C):
                Pm = _sample_patches(cur, c, cfg.patch_samples[h], cfg.kernel, rng)
                units.append(fit_saab_unit(Pm))
            Nk = cfg.kernel ** 2
            cum = np.concatenate([cum_parent[c] * units[c].energy for c in range(C)])
            parents = np.repeat(np.arange(C), Nk)
            children = np.tile(np.arange(Nk), C)
            feat, feat_e, prop, prop_e, n_cand = self._rank_nodes(
                cum, parents, children, n_feat, n_prop if h < cfg.n_hops - 1 else 0,
                cfg.energy_floor)

            shape = cfg.hop_shapes()[h]
            hop = HopFit(units=units, d=d, feat_nodes=feat, feat_energy=feat_e,
                         prop_nodes=prop, prop_energy=prop_e, n_candidates=n_cand,
                         pool_h=shape["pool_h"], pool_w=shape["pool_w"])
            self.hops.append(hop)

            diag_hops.append({
                "hop": h + 1,
                "input_channels": int(C),
                "candidate_nodes": int(C * Nk),
                "candidates_above_energy_floor": int(n_cand),
                "energy_floor": float(cfg.energy_floor),
                "candidates_below_1e-6_cumulative_energy": int((cum < 1e-6).sum()),
                "feature_nodes": int(len(feat)),
                "propagated_nodes": int(len(prop)),
                "conv_map": f"{shape['conv_h']}x{shape['conv_w']}",
                "pooled_map": f"{shape['pool_h']}x{shape['pool_w']}",
                "receptive_field_px": int(cfg.receptive_fields()[h]),
                "bias_d": float(d),
                "features_from_this_hop": int(len(feat) * (cfg.grid_rows * cfg.grid_cols * 2 + 1)),
                "top_cumulative_energy": [float(v) for v in feat_e[:10]],
                "kept_energy_sum": float(np.sum(feat_e)),
            })
            if h == 0:
                diag_hops[0]["hop1_local_fractional_energy"] = [
                    float(v) for v in units[0].energy]

            if h < cfg.n_hops - 1:
                # Stage the propagated maps for the next hop's fit.  Only the propagated
                # channels are retained, so peak memory is n_train x n_prop x Hp x Wp.
                Wm, bv = self._hop_matrices(hop)
                idx = np.array([p * Nk + c for p, c in prop], dtype=np.int64)
                nxt = np.empty((cur.shape[0], len(idx), hop.pool_h, hop.pool_w),
                               dtype=np.float32)
                for i in range(0, cur.shape[0], cfg.chunk):
                    y = _pool2(_conv_hop(cur[i:i + cfg.chunk], Wm, bv), cfg.pool)
                    nxt[i:i + y.shape[0]] = y[:, idx]
                    del y
                cur = nxt
                cum_parent = np.asarray(prop_e, dtype=np.float64)
            else:
                del cur

        # Feature names, in the exact column order produced by _cell_stats.
        stats = ([f"r{r}c{c}_{s}" for r in range(cfg.grid_rows)
                  for c in range(cfg.grid_cols) for s in ("mean", "std")] + ["gmax"])
        for h, hop in enumerate(self.hops):
            for rank, (p, c) in enumerate(hop.feat_nodes):
                for st in stats:
                    self.feature_names.append(f"h{h+1}_n{rank}_p{p}c{c}_{st}")

        self.fit_paths = frozenset(str(p) for p in paths)
        self._fitted = True
        self._fingerprint = self._fingerprint_state()
        self.diag = {
            "hops": diag_hops,
            "n_fit_images": int(imgs.shape[0]),
            "feature_dim": int(len(self.feature_names)),
            "fit_seconds": round(time.time() - t0, 2),
        }
        return self

    # --------------------------------------------------------------- transform --
    def transform(self, imgs: np.ndarray, desc: str = "transform",
                  quiet: bool = True) -> Tuple[np.ndarray, Dict[str, float]]:
        S._require(self._fitted, "GreenExtractor.transform called before fit")
        before = self._fingerprint
        cfg = self.cfg
        n = imgs.shape[0]
        F = len(self.feature_names)
        X = np.empty((n, F), dtype=np.float32)
        Nk = cfg.kernel ** 2
        neg = np.zeros(cfg.n_hops, dtype=np.int64)
        tot = np.zeros(cfg.n_hops, dtype=np.int64)
        t0 = time.time()

        mats = [self._hop_matrices(hop) for hop in self.hops]
        fidxs = [np.array([p * Nk + c for p, c in hop.feat_nodes], dtype=np.int64)
                 for hop in self.hops]
        pidxs = [np.array([p * Nk + c for p, c in hop.prop_nodes], dtype=np.int64)
                 for hop in self.hops]
        min_resp = np.full(cfg.n_hops, np.inf)     # min AC response, per hop
        min_dc = np.full(cfg.n_hops, np.inf)       # min DC response, per hop

        rng_iter = range(0, n, cfg.chunk)
        if not quiet:
            rng_iter = tqdm(list(rng_iter), desc=desc, unit="chunk", leave=False)
        for i in rng_iter:
            cur = np.ascontiguousarray(imgs[i:i + cfg.chunk, None, :, :], dtype=np.float32)
            m = cur.shape[0]
            cols = []
            for h, hop in enumerate(self.hops):
                Wm, bv = mats[h]
                y = _conv_hop(cur, Wm, bv)
                # The quantity being measured -- and the quantity the fold guard asserts
                # against max_neg_frac -- is the fraction of *AC* responses that are
                # negative, i.e. the fraction of places where the adjusted bias failed to
                # dominate and a ReLU would not have been the identity.  Count over the AC
                # channels ONLY.  Counting over all channels while dividing by the AC total
                # is only equivalent while every DC response is non-negative, which holds
                # at hop 1 (pixels in [0,1]) but at hops 2-3 only CONDITIONALLY on the
                # earlier hops having stayed non-negative: once hop 1 emits a negative, the
                # pooled map handed to hop 2 has negative entries and a hop-2 DC response
                # (a positive-weighted sum) can be negative too.  Those DC negatives would
                # then be charged to an AC-only denominator and the reported
                # frac_negative_ac_response would not be the AC fraction it is named.
                # The DC minimum is reported separately: a negative there is a different
                # fault (a negative input), not a failure of the bias.
                yv = y.reshape(m, -1, Nk, y.shape[2], y.shape[3])
                y_ac, y_dc = yv[:, :, 1:], yv[:, :, 0]
                mn = float(y_ac.min())
                min_resp[h] = min(min_resp[h], mn)
                min_dc[h] = min(min_dc[h], float(y_dc.min()))
                if mn < 0.0:
                    neg[h] += int(np.count_nonzero(y_ac < 0))
                tot[h] += int(y_ac.size)
                del yv, y_ac, y_dc
                y = _pool2(y, cfg.pool)
                rs = _cell_slices(y.shape[2], cfg.grid_rows)
                cs = _cell_slices(y.shape[3], cfg.grid_cols)
                cols.append(_cell_stats(y[:, fidxs[h]], rs, cs))
                if h < cfg.n_hops - 1:
                    cur = np.ascontiguousarray(y[:, pidxs[h]])
                del y
            X[i:i + m] = np.concatenate(cols, axis=1)
            del cols, cur

        S._require(self._fingerprint_state() == before,
                   "GreenExtractor.transform mutated the fitted state -- a refit is "
                   "happening inside transform, which would leak the transformed set "
                   "into the filter bank.")
        frac = {f"hop{h+1}": float(neg[h] / max(1, tot[h])) for h in range(cfg.n_hops)}
        frac["overall"] = float(neg.sum() / max(1, tot.sum()))
        frac["min_ac_response_per_hop"] = [float(v) for v in min_resp]
        frac["min_dc_response_per_hop"] = [float(v) for v in min_dc]
        frac["n_ac_responses"] = int(tot.sum())
        frac["transform_seconds"] = round(time.time() - t0, 2)
        return X, frac


# ==================================================================================
# 5.  Discriminant feature test (supervised -- inner-training fold only)
# ==================================================================================
class DFTSelector:
    """Weighted 32-bin entropy split search, one pass over a (n, F) feature bank.

    The clip to the training 1st/99th percentile before the uniform binning is not
    cosmetic: Saab responses are heavy-tailed, and without it >99 % of samples land in
    bin 0, every split degenerates against the occupancy guard, and the ranking becomes
    noise.  The two percentiles are fitted objects.

    The weights are the harness's own S.sample_weights (per-patient equalisation x
    patient-level class balance), so the DFT optimises the same quantity the classifier
    and the patient-level evaluation do.

    K = min(300, n_train_images // 4).  No elbow detection: an elbow on 4233 noisy
    scores is not reproducible across folds and would turn K into a hidden
    hyper-parameter fitted on the fold.
    """

    def __init__(self, cfg: GLConfig):
        self.cfg = cfg
        self.lo: Optional[np.ndarray] = None
        self.hi: Optional[np.ndarray] = None
        self.scores: Optional[np.ndarray] = None
        self.order: Optional[np.ndarray] = None
        self.keep: Optional[np.ndarray] = None
        self.K: int = 0
        self.h_all: float = float("nan")

    @staticmethod
    def _ent2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        t = a + b
        with np.errstate(divide="ignore", invalid="ignore"):
            p = np.where(t > 0, a / np.maximum(t, 1e-300), 0.0)
            q = 1.0 - p
            H = -(np.where(p > 0, p * np.log2(np.maximum(p, 1e-300)), 0.0) +
                  np.where(q > 0, q * np.log2(np.maximum(q, 1e-300)), 0.0))
        return np.where(t > 0, H, 0.0)

    def fit(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> "DFTSelector":
        cfg = self.cfg
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).astype(int)
        w = np.asarray(w, dtype=np.float64)
        n, F = X.shape
        B = cfg.dft_bins

        self.lo = np.percentile(X, 1.0, axis=0).astype(np.float64)
        self.hi = np.percentile(X, 99.0, axis=0).astype(np.float64)
        Z = self._normalise(X)
        b = np.minimum((Z * B).astype(np.int32), B - 1)
        b = np.maximum(b, 0)

        cols = np.arange(F, dtype=np.int64)
        flat = b.astype(np.int64) * F + cols[None, :]
        m1 = (y == 1)
        h1 = np.bincount(flat[m1].ravel(), weights=np.repeat(w[m1], F),
                         minlength=B * F).reshape(B, F)
        h0 = np.bincount(flat[~m1].ravel(), weights=np.repeat(w[~m1], F),
                         minlength=B * F).reshape(B, F)
        r1 = np.bincount(flat[m1].ravel(), minlength=B * F).reshape(B, F).astype(np.float64)
        r0 = np.bincount(flat[~m1].ravel(), minlength=B * F).reshape(B, F).astype(np.float64)
        del flat

        c1, c0 = np.cumsum(h1, axis=0), np.cumsum(h0, axis=0)
        cr1, cr0 = np.cumsum(r1, axis=0), np.cumsum(r0, axis=0)
        t1, t0 = c1[-1], c0[-1]
        W = t1 + t0
        L1, L0 = c1[:-1], c0[:-1]
        R1, R0 = t1 - L1, t0 - L0
        WL, WR = L1 + L0, R1 + R0
        nL = cr1[:-1] + cr0[:-1]
        nR = (cr1[-1] + cr0[-1]) - nL

        loss = (WL * self._ent2(L1, L0) + WR * self._ent2(R1, R0)) / np.maximum(W, 1e-300)
        ok = ((WL >= cfg.dft_min_weight_frac * W) & (WR >= cfg.dft_min_weight_frac * W) &
              (nL >= cfg.dft_min_raw) & (nR >= cfg.dft_min_raw))
        self.h_all = float(self._ent2(np.atleast_1d(t1.sum()), np.atleast_1d(t0.sum()))[0])
        loss = np.where(ok, loss, np.inf)
        s = loss.min(axis=0)
        s = np.where(np.isfinite(s), s, self.h_all)

        self.scores = s
        self.order = np.lexsort((cols, s))              # ascending score, ties by index
        self.K = int(min(cfg.dft_k_cap, max(1, n // cfg.dft_k_divisor), F))
        self.keep = np.sort(self.order[:self.K])
        return self

    def _normalise(self, X: np.ndarray) -> np.ndarray:
        rng = self.hi - self.lo
        safe = np.where(rng > 0, rng, 1.0)
        Z = (np.clip(X, self.lo, self.hi) - self.lo) / safe
        return np.where(rng > 0, Z, 0.0)

    def transform(self, X: np.ndarray) -> np.ndarray:
        S._require(self.keep is not None, "DFTSelector.transform called before fit")
        return np.asarray(X, dtype=np.float32)[:, self.keep]

    def top_names(self, names: Sequence[str], k: int = 20) -> List[Dict[str, float]]:
        return [{"rank": int(i), "feature": str(names[int(j)]),
                 "dft_entropy": float(self.scores[int(j)])}
                for i, j in enumerate(self.order[:k])]


# ==================================================================================
# 6.  Classifier: L2 logistic regression, C chosen on inner-validation PATIENT AUC
# ==================================================================================
C_GRID = (1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0)


def _fit_lr(Xs: np.ndarray, y: np.ndarray, w: np.ndarray, C: float) -> LogisticRegression:
    m = LogisticRegression(penalty="l2", solver="lbfgs", C=C, max_iter=5000, tol=1e-6,
                           fit_intercept=True)
    m.fit(Xs, y, sample_weight=w)
    return m


def select_C_by_inner_val(Xtr: np.ndarray, ytr: np.ndarray, wtr: np.ndarray,
                          Xva: np.ndarray, va_rows: pd.DataFrame,
                          grid: Sequence[float] = C_GRID) -> Tuple[float, dict, StandardScaler, dict]:
    """Fit one LR per C on the INNER-TRAINING images; score the inner-validation
    PATIENTS; keep the C with the best patient-level AUC, ties to the smaller C.

    Why L2-LR and not XGBoost: the effective sample size is the number of PATIENTS
    (211 training patients at NCKUH, 78 at CYCH), not the images -- 1-4 views of one
    child on one day are near-duplicates.  A depth-3, 300-tree booster on 206 features
    has thousands of effective parameters against ~78 independent units and would
    interpolate the fold; worse, its early stopping would consume the same ~53-patient
    inner-validation set already used to pick C.  The green-learning claim under test is
    that the REPRESENTATION carries the signal, so the readout must be the least
    expressive thing that can read it.

    Why not a linear SVM: decision_function is unbounded and uncalibrated, and the
    harness aggregates image -> patient by the MEAN of a probability and then feeds that
    mean through logit() into the fused arm and through youden_threshold() into the
    operating point.  An SVM would give a valid AUC and an incomparable everything else.
    """
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xva_s = sc.transform(Xtr), sc.transform(Xva)
    yv = None
    best = (None, -np.inf, None)
    trace = {}
    for C in grid:
        m = _fit_lr(Xtr_s, ytr, wtr, C)
        pv = m.predict_proba(Xva_s)[:, 1]
        merged = S.aggregate_patient_scores(va_rows, pv)
        auc = S.fast_auc(merged["label"].to_numpy(), merged["score_mean"].to_numpy())
        trace[f"{C:g}"] = float(auc)
        if np.isfinite(auc) and auc > best[1] + 1e-12:
            best = (C, float(auc), m)
            yv = pv
    if best[0] is None:                                # every C degenerate
        C = grid[0]
        m = _fit_lr(Xtr_s, ytr, wtr, C)
        yv = m.predict_proba(Xva_s)[:, 1]
        best = (C, float("nan"), m)
    return best[0], {"model": best[2], "val_probs": yv, "inner_val_patient_auc": best[1]}, sc, trace


# ==================================================================================
# 7.  Folds
# ==================================================================================
def read_fold_assignments(site: str) -> Optional[pd.DataFrame]:
    p = S.PROJECT_ROOT / "results_single_center" / site / "fold_assignments.csv"
    if not p.exists():
        return None
    return pd.read_csv(p, dtype={"patient_id": str})


def read_cnn_patient_oof(site: str) -> Optional[pd.DataFrame]:
    p = S.PROJECT_ROOT / "results_single_center" / site / "oof_predictions_patient.csv"
    if not p.exists():
        return None
    return pd.read_csv(p, dtype={"patient_id": str})


def build_fold_plan(pat: pd.DataFrame, site: str, seeds: Sequence[int], folds: int,
                    inner_folds: int, allow_replay: bool,
                    limit_folds: Optional[int] = None) -> Tuple[List[dict], dict]:
    """Return the list of folds to run, plus a provenance record.

    The frozen fold_assignments.csv is the AUTHORITY when it covers the requested
    (seed, fold) grid, and the harness's own generator is re-run as a tripwire that must
    agree with it.  Regeneration alone is exact only under an unstated conjunction --
    identical manifest, keep_first_exam_only, the patient_id->str cast, the sort by
    patient_id, folds=5, inner_folds=4, AND the installed scikit-learn version, since
    StratifiedGroupKFold's assignment heuristic is not a versioned public guarantee.
    The CSV cannot drift; replaying it lets the paired comparison be defended as "scored
    on the identical partitions recorded in fold_assignments.csv".

    `folds` is the CV DESIGN (5, the value the CNN run used); `limit_folds` runs only the
    first N of those folds per seed.  --smoke uses the second knob, never the first, so
    that the smoke run exercises the real replay path instead of silently generating a
    different 2-fold partition that could never be compared to anything.
    """
    n_run = folds if limit_folds is None else int(min(limit_folds, folds))
    fa = read_fold_assignments(site) if allow_replay else None
    prov: dict = {"fold_source": "regenerated", "fold_assignments_csv": None,
                  "regeneration_matches_csv": None, "notes": ""}
    plan: List[dict] = []

    stored: Dict[Tuple[int, int], Dict[str, List[str]]] = {}
    if fa is not None:
        cohort = set(pat["patient_id"])
        for (s, f), g in fa.groupby(["seed", "fold"]):
            stored[(int(s), int(f))] = {sp: gg["patient_id"].tolist()
                                        for sp, gg in g.groupby("split")}
        prov["fold_assignments_csv"] = str(
            S.PROJECT_ROOT / "results_single_center" / site / "fold_assignments.csv")
        covered = all((s, f) in stored for s in seeds for f in range(n_run))
        same_cohort = (set(fa["patient_id"]) == cohort)
        if not covered:
            prov["notes"] = ("frozen fold_assignments.csv does not cover the requested "
                             "seed x fold grid; regenerating from the harness generator")
            fa = None
        elif not same_cohort:
            prov["notes"] = ("frozen fold_assignments.csv covers a different patient set "
                             "than the loaded cohort; regenerating")
            fa = None

    matches = True
    for seed in seeds:
        splits = S.outer_splits(pat, folds, seed)
        for fold in range(n_run):
            tr_idx, te_idx = splits[fold]
            p_tr = pat.iloc[tr_idx].reset_index(drop=True)
            itr, iva = S.inner_split(p_tr, inner_folds, seed)
            gen = {
                "inner_train": p_tr.iloc[itr]["patient_id"].tolist(),
                "inner_val": p_tr.iloc[iva]["patient_id"].tolist(),
                "test": pat.iloc[te_idx]["patient_id"].tolist(),
            }
            if fa is not None:
                rec = stored[(seed, fold)]
                agree = all(set(rec[k]) == set(gen[k])
                            for k in ("inner_train", "inner_val", "test"))
                matches = matches and agree
                use = {k: sorted(rec[k]) for k in ("inner_train", "inner_val", "test")}
            else:
                use = {k: sorted(v) for k, v in gen.items()}
            plan.append({"seed": int(seed), "fold": int(fold), **use})

    if fa is not None:
        prov["fold_source"] = "replayed from results_single_center/%s/fold_assignments.csv" % site
        prov["regeneration_matches_csv"] = bool(matches)
        S._require(matches,
                   "fold_assignments.csv disagrees with the regenerated split -- the "
                   "green-learning arm would NOT be on the CNN run's partitions.")
        prov["notes"] = ("frozen partitions replayed; the harness generator was re-run and "
                         "reproduced every (seed, fold, patient, split) tuple exactly")
    elif not prov["notes"]:
        prov["notes"] = ("no frozen fold_assignments.csv available (or --permute-labels "
                         "changes the stratification); partitions generated from the seed")
    return plan, prov


def run_one_fold(cfg: GLConfig, site: str, seed: int, fold: int,
                 pat: pd.DataFrame, img_df: pd.DataFrame, cache: np.ndarray,
                 geom: np.ndarray, cache_paths: np.ndarray, pids: Dict[str, List[str]],
                 fusion_C: float, clinical_C: float) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    t_fold = time.time()
    S.set_all_seeds(seed * 100 + fold)
    pids_tr, pids_va, pids_te = pids["inner_train"], pids["inner_val"], pids["test"]
    S.assert_patient_disjoint(pids_tr, pids_va, pids_te, f"green {site} s{seed}f{fold}")

    def rows_of(ids) -> pd.DataFrame:
        return img_df[img_df["patient_id"].isin(set(ids))].sort_values(
            "img_idx").reset_index(drop=True)

    def pat_of(ids) -> pd.DataFrame:
        return pat[pat["patient_id"].isin(set(ids))].sort_values(
            "patient_id").reset_index(drop=True)

    img_tr, img_va, img_te = rows_of(pids_tr), rows_of(pids_va), rows_of(pids_te)
    pat_tr, pat_va, pat_te = pat_of(pids_tr), pat_of(pids_va), pat_of(pids_te)
    S._require(len(img_tr) and len(img_va) and len(img_te),
               f"empty split at seed {seed} fold {fold}")
    for nm, tbl in (("inner_train", pat_tr), ("inner_val", pat_va), ("test", pat_te)):
        S._require(tbl["label"].nunique() >= 2,
                   f"DEGENERATE FOLD s{seed}f{fold}: {nm} is single-class; thresholds and "
                   "AUC are undefined there.")

    i_tr = img_tr["img_idx"].to_numpy()
    i_va = img_va["img_idx"].to_numpy()
    i_te = img_te["img_idx"].to_numpy()

    # ---- the leakage guard, stated at the level that matters: FILE PATHS ---------
    p_tr = set(img_tr["abs_path"]); p_va = set(img_va["abs_path"]); p_te = set(img_te["abs_path"])
    S._require(not (p_tr & p_va) and not (p_tr & p_te) and not (p_va & p_te),
               "image files are shared between splits")

    # ---- and the guard that the path guard alone CANNOT give ---------------------
    # Everything below selects pixels as cache[img_idx].  Checking that fit() was handed
    # the inner-training PATH LIST proves nothing about which pixels it got: that
    # correspondence is created in build_image_cache and consumed 500 lines later.  Verify
    # it per split against the cache's own path vector, so a cache/img_idx desynchronisation
    # aborts here instead of quietly fitting the Saab bank on test images.
    for _nm, _idx, _rows in (("inner_train", i_tr, img_tr), ("inner_val", i_va, img_va),
                             ("test", i_te, img_te)):
        S._require(list(cache_paths[_idx]) == list(_rows["abs_path"]),
                   f"CACHE INDEX DESYNCHRONISED for the {_nm} split: cache[img_idx] does not "
                   "address the images of these rows, so the Saab bank, the DFT and the "
                   "classifier would be fitted on the wrong pixels while every path-level "
                   "guard still passed.")

    # ---------------------------------------------------------- Saab filter bank --
    ext = GreenExtractor(cfg, seed=seed * 100 + fold)
    ext.fit(cache[i_tr], img_tr["abs_path"].tolist())
    S._require(ext.fit_paths == frozenset(img_tr["abs_path"].tolist()),
               "the Saab filter bank was fitted on something other than the inner-training images")
    S._require(ext.fit_paths.isdisjoint(p_va) and ext.fit_paths.isdisjoint(p_te),
               "SAAB FILTER BANK LEAK: the filters saw validation or test images.")

    Xtr, neg_tr = ext.transform(cache[i_tr], "transform train")
    Xva, neg_va = ext.transform(cache[i_va], "transform val")
    Xte, neg_te = ext.transform(cache[i_te], "transform test")
    S._require(neg_tr["overall"] <= cfg.max_neg_frac,
               f"frac_negative_ac_response on the FITTING set is {neg_tr['overall']:.3g} "
               f"> {cfg.max_neg_frac:g}: the adjusted bias does not dominate its own "
               "training patches, which can only mean the bias or the basis is wrong.")
    S._require(max(neg_va["overall"], neg_te["overall"]) <= cfg.max_neg_frac,
               f"frac_negative_ac_response is {max(neg_va['overall'], neg_te['overall']):.3g} "
               f"> {cfg.max_neg_frac:g} on held-out images: the resize or the intensity "
               "normalisation differs between fit and transform.")

    # -------------------------------------------------- DFT + scaler + LR (green) --
    w_tr = S.sample_weights(img_tr)
    y_tr = img_tr["label"].to_numpy()
    t_dft = time.time()
    dft = DFTSelector(cfg).fit(Xtr, y_tr, w_tr)
    dft_secs = round(time.time() - t_dft, 2)
    Ztr, Zva, Zte = dft.transform(Xtr), dft.transform(Xva), dft.transform(Xte)
    C_green, sel, sc_g, C_trace = select_C_by_inner_val(Ztr, y_tr, w_tr, Zva, img_va)
    p_va_g = sel["val_probs"]
    p_te_g = sel["model"].predict_proba(sc_g.transform(Zte))[:, 1]

    pat_va_g = S.aggregate_patient_scores(img_va, p_va_g)
    pat_te_g = S.aggregate_patient_scores(img_te, p_te_g)
    thr_g = S.youden_threshold(pat_va_g["label"].to_numpy(), pat_va_g["score_mean"].to_numpy(),
                               f"green s{seed}f{fold}")
    thr_g90 = S.sensitivity_threshold(pat_va_g["label"].to_numpy(),
                                      pat_va_g["score_mean"].to_numpy(), 0.90,
                                      f"green s{seed}f{fold}")
    thr_g_img = S.youden_threshold(img_va["label"].to_numpy(), p_va_g, f"green-img s{seed}f{fold}")

    # ------------------------------------------------------------- geom (control) --
    Gtr, Gva, Gte = geom[i_tr], geom[i_va], geom[i_te]
    C_geom, sel_m, sc_m, C_trace_geom = select_C_by_inner_val(Gtr, y_tr, w_tr, Gva, img_va)
    p_va_m = sel_m["val_probs"]
    p_te_m = sel_m["model"].predict_proba(sc_m.transform(Gte))[:, 1]
    pat_va_m = S.aggregate_patient_scores(img_va, p_va_m)
    pat_te_m = S.aggregate_patient_scores(img_te, p_te_m)
    thr_m = S.youden_threshold(pat_va_m["label"].to_numpy(), pat_va_m["score_mean"].to_numpy())
    thr_m90 = S.sensitivity_threshold(pat_va_m["label"].to_numpy(),
                                      pat_va_m["score_mean"].to_numpy(), 0.90)
    thr_m_img = S.youden_threshold(img_va["label"].to_numpy(), p_va_m)

    # --------------------------------------------------------------- clinical arm --
    Xc_tr = pat_tr[CLINICAL_FEATURES].to_numpy(float)
    Xc_va = pat_va[CLINICAL_FEATURES].to_numpy(float)
    Xc_te = pat_te[CLINICAL_FEATURES].to_numpy(float)
    sc_c = StandardScaler().fit(Xc_tr)
    clf_c = LogisticRegression(max_iter=2000, C=clinical_C)
    clf_c.fit(sc_c.transform(Xc_tr), pat_tr["label"].to_numpy())
    s_c_va = clf_c.predict_proba(sc_c.transform(Xc_va))[:, 1]
    s_c_te = clf_c.predict_proba(sc_c.transform(Xc_te))[:, 1]
    thr_c = S.youden_threshold(pat_va["label"].to_numpy(), s_c_va)
    thr_c90 = S.sensitivity_threshold(pat_va["label"].to_numpy(), s_c_va, 0.90)

    # ------------------------------------------------------------------ fused arm --
    # Fitted on the INNER-VALIDATION patients -- patients the extractor and the
    # classifier never saw -- exactly as the CNN run's fused arm is.  Fitting it on the
    # classifier's own training patients would feed the fuser near-memorised
    # probabilities and inflate this arm.
    va_order = pat_va.set_index("patient_id").loc[pat_va_g["patient_id"]].reset_index()
    te_order = pat_te.set_index("patient_id").loc[pat_te_g["patient_id"]].reset_index()
    S._require(list(va_order["patient_id"]) == list(pat_va_g["patient_id"]),
               "val row misalignment between the patient table and the green patient scores")
    S._require(list(te_order["patient_id"]) == list(pat_te_g["patient_id"]),
               "test row misalignment between the patient table and the green patient scores")
    S._require(bool((te_order["label"].to_numpy() == pat_te_g["label"].to_numpy()).all()),
               "test label mismatch after reordering")
    Xf_va = np.column_stack([S.logit(pat_va_g["score_mean"].to_numpy()),
                             va_order[CLINICAL_FEATURES].to_numpy(float)])
    Xf_te = np.column_stack([S.logit(pat_te_g["score_mean"].to_numpy()),
                             te_order[CLINICAL_FEATURES].to_numpy(float)])
    yf_va = pat_va_g["label"].to_numpy()
    if len(np.unique(yf_va)) < 2:
        s_f_te, thr_f, thr_f90 = pat_te_g["score_mean"].to_numpy(), thr_g, thr_g90
    else:
        sc_f = StandardScaler().fit(Xf_va)
        clf_f = LogisticRegression(max_iter=2000, C=fusion_C)
        clf_f.fit(sc_f.transform(Xf_va), yf_va)
        s_f_te = clf_f.predict_proba(sc_f.transform(Xf_te))[:, 1]
        thr_f, thr_f90 = S.cv_oof_threshold(Xf_va, yf_va, seed, C=fusion_C)

    # -------------------------------------------------------------------- assemble --
    ids = pat_te_g["patient_id"].to_numpy()
    S._require(list(pat_te_m["patient_id"]) == list(ids),
               "geom row misalignment against the green patient score table")
    # s_c_te comes out in pat_te's row order; pat_out is written in pat_te_g's order.  Both
    # happen to be patient_id-sorted today, but `te_order` is CONSTRUCTED from pat_te_g's
    # ids, so comparing it to `ids` cannot detect a pat_te <-> pat_te_g ordering difference.
    # Align explicitly by patient_id -- the same thing single_center_cv.run_fold does with
    # its own s_c_te_aligned -- so a silently permuted clinical arm is impossible rather
    # than merely improbable.  (A permuted clinical arm would corrupt score_clinical, the
    # green-vs-clinical comparison, and would be caught by the CNN-reproduction tripwire
    # only on a replayed, non-permuted run.)
    S._require(len(pat_te) == len(ids) and set(pat_te["patient_id"]) == set(ids),
               "the patient table and the green patient score table cover different test "
               "patients; the clinical arm could not be aligned onto the output rows")
    s_c_te = pd.Series(s_c_te, index=pat_te["patient_id"].to_numpy()).loc[ids].to_numpy()
    pat_out = pd.DataFrame({
        "site": site, "seed": seed, "fold": fold,
        "patient_id": ids,
        "label": pat_te_g["label"].to_numpy(),
        "n_images": pat_te_g["n_images"].to_numpy(),
        "age_years": te_order["age_years"].to_numpy(),
        "gender_code": te_order["gender_code"].to_numpy(),
        "renal_measure_cm": te_order["renal_measure_cm"].to_numpy(),
        "score_green": pat_te_g["score_mean"].to_numpy(),
        "score_green_mean": pat_te_g["score_mean"].to_numpy(),
        "score_green_maxkidney": pat_te_g["score_maxkidney"].to_numpy(),
        "score_fused": s_f_te,
        "score_geom": pat_te_m["score_mean"].to_numpy(),
        "score_geom_mean": pat_te_m["score_mean"].to_numpy(),
        "score_geom_maxkidney": pat_te_m["score_maxkidney"].to_numpy(),
        "score_clinical": s_c_te,
        "thr_green": thr_g, "thr_fused": thr_f, "thr_geom": thr_m, "thr_clinical": thr_c,
        "thr_green_sens90": thr_g90, "thr_fused_sens90": thr_f90,
        "thr_geom_sens90": thr_m90, "thr_clinical_sens90": thr_c90,
    })
    for a in GL_PATIENT_ARMS:
        pat_out[f"pred_{a}"] = (pat_out[f"score_{a}"] >= pat_out[f"thr_{a}"]).astype(int)
        pat_out[f"pred_{a}_sens90"] = (
            pat_out[f"score_{a}"] >= pat_out[f"thr_{a}_sens90"]).astype(int)

    img_out = img_te[["patient_id", "filename", "relative_path", "view", "side", "label"]].copy()
    img_out.insert(0, "fold", fold); img_out.insert(0, "seed", seed); img_out.insert(0, "site", site)
    img_out["prob_green"] = p_te_g
    img_out["prob_geom"] = p_te_m
    img_out["thr_green_img"] = thr_g_img
    img_out["thr_geom_img"] = thr_m_img
    img_out["pred_green"] = (p_te_g >= thr_g_img).astype(int)
    img_out["pred_geom"] = (p_te_m >= thr_m_img).astype(int)

    rec = {
        "seed": seed, "fold": fold,
        "splits": [S.split_counts(pat, img_df, pids_tr, "inner_train"),
                   S.split_counts(pat, img_df, pids_va, "inner_val"),
                   S.split_counts(pat, img_df, sorted(set(pids_tr) | set(pids_va)),
                                  "outer_train_total"),
                   S.split_counts(pat, img_df, pids_te, "test")],
        "leakage_check": "passed: train/val/test patient AND file sets are pairwise disjoint",
        # The one preprocessing step that is still keyed on native crop size after the
        # anisotropic stretch has destroyed absolute size: the specification's
        # INTER_AREA-if-downsampling / INTER_LINEAR-if-upsampling rule.  The two kernels
        # leave different high-frequency content, so the branch writes a (weak) function
        # of crop W and H back into the speckle the Saab hops read.  Measured on the whole
        # cohort, the per-patient INTER_AREA fraction alone is worth patient AUC 0.556 at
        # NCKUH and 0.520 at CYCH, against 0.600 for crop height itself.  It is the
        # specification's own rule so it is not overridden here, but it is recorded per
        # split so the residual confound is auditable rather than invisible.
        "interp_branch_area_fraction": {
            nm: float(((geom[ix, 0] >= cfg.in_w) & (geom[ix, 1] >= cfg.in_h)).mean())
            for nm, ix in (("inner_train", i_tr), ("inner_val", i_va), ("test", i_te))},
        "extractor": ext.diag,
        "frac_negative_ac_response": {"fit_set": neg_tr, "inner_val": neg_va, "test": neg_te},
        "dft": {
            "n_candidate_features": int(Xtr.shape[1]),
            "n_selected": int(dft.K),
            "rule": f"K = min({cfg.dft_k_cap}, floor(n_train_images/{cfg.dft_k_divisor}))",
            "n_train_images": int(len(img_tr)),
            "entropy_of_full_fitting_set": float(dft.h_all),
            "seconds": dft_secs,
            "top20": dft.top_names(ext.feature_names, 20),
        },
        "selected_C": {"green": float(C_green), "geom": float(C_geom)},
        "C_grid_inner_val_patient_auc": {"green": C_trace, "geom": C_trace_geom},
        "inner_val_patient_auc": {
            "green": float(sel["inner_val_patient_auc"]),
            "geom": float(sel_m["inner_val_patient_auc"]),
        },
        "selection_optimism": {
            "green_inner_val_patient_auc": float(sel["inner_val_patient_auc"]),
            "green_test_patient_auc": float(S.fast_auc(pat_te_g["label"].to_numpy(),
                                                       pat_te_g["score_mean"].to_numpy())),
            "note": "the inner-val AUC picked C and both thresholds; it is a SELECTION "
                    "statistic, not a performance estimate.",
        },
        "thresholds": {"green_patient_youden": float(thr_g), "green_image_youden": float(thr_g_img),
                       "geom_patient_youden": float(thr_m), "clinical_patient_youden": float(thr_c),
                       "fused_patient_youden": float(thr_f)},
        "fold_seconds": round(time.time() - t_fold, 2),
        "_pids": {"inner_train": pids_tr, "inner_val": pids_va, "test": pids_te},
    }
    return pat_out, img_out, rec


# ==================================================================================
# 8.  Reporting
# ==================================================================================
def gl_figures(pat_pred: pd.DataFrame, img_pred: pd.DataFrame, across: dict,
               out_dir: Path, site: str, ref_seed: int) -> None:
    def _curves(df, arms, col_fmt, level, roc_name, pr_name, key):
        d = df[df["seed"] == ref_seed]
        if not len(d):
            return
        y = d["label"].to_numpy()
        n_seeds = int(df["seed"].nunique())
        for kind in ("roc", "pr"):
            fig, ax = plt.subplots(figsize=(5.4, 5.0))
            means = []
            for a in arms:
                s = d[col_fmt.format(a)].to_numpy()
                if kind == "roc":
                    fpr, tpr, _ = roc_curve(y, s)
                    here = float(roc_auc_score(y, s))
                    m = across[key][a]["auc"]
                    ax.plot(fpr, tpr, lw=1.6, color=GL_ARM_COLORS[a],
                            label=f"{GL_ARM_LABELS[a]}\nAUC {here:.3f} (this curve, seed {ref_seed})")
                    means.append(f"{a} {m['mean']:.3f}+/-{m['sd']:.3f}")
                else:
                    pr, rc, _ = precision_recall_curve(y, s)
                    here = float(average_precision_score(y, s))
                    m = across[key][a]["average_precision"]
                    ax.plot(rc, pr, lw=1.6, color=GL_ARM_COLORS[a],
                            label=f"{GL_ARM_LABELS[a]}\nAP {here:.3f} (this curve, seed {ref_seed})")
                    means.append(f"{a} {m['mean']:.3f}+/-{m['sd']:.3f}")
            if kind == "roc":
                ax.plot([0, 1], [0, 1], ls="--", lw=0.9, color="0.5")
                ax.set_xlabel("1 - specificity"); ax.set_ylabel("Sensitivity")
                ax.set_title(f"{site}: green learning, {level}-level ROC, out-of-fold, "
                             f"seed {ref_seed} of {n_seeds}\n{n_seeds}-seed mean AUC +/- SD: "
                             + "; ".join(means), fontsize=7.2)
                ax.legend(fontsize=6.2, loc="lower right", frameon=False)
                fname = roc_name
            else:
                ax.axhline(float(np.mean(y)), ls="--", lw=0.9, color="0.5", label="Prevalence")
                ax.set_xlabel("Recall (sensitivity)"); ax.set_ylabel("Precision (PPV)")
                ax.set_title(f"{site}: green learning, {level}-level precision-recall, "
                             f"out-of-fold, seed {ref_seed} of {n_seeds}\n"
                             f"{n_seeds}-seed mean AP +/- SD: " + "; ".join(means), fontsize=7.2)
                ax.legend(fontsize=6.2, loc="lower left", frameon=False)
                fname = pr_name
            fig.tight_layout(); fig.savefig(out_dir / fname, dpi=600); plt.close(fig)

    _curves(pat_pred, GL_PATIENT_ARMS, "score_{}", "Patient",
            "roc_patient.png", "pr_patient.png", "patient_level")
    _curves(img_pred, GL_IMAGE_ARMS, "prob_{}", "Image",
            "roc_image.png", "pr_image.png", "image_level")


def _fmt_ci(d: dict) -> str:
    return f"{d['point_estimate']:.3f} [{d['ci_lo']:.3f}, {d['ci_hi']:.3f}]"


def write_summary(path: Path, args, cfg: GLConfig, notes: dict, prov: dict,
                  fold_records: List[dict], across_pat: dict, across_img: dict,
                  boot_pat: dict, boot_img: dict, comparisons: dict, holm: dict,
                  permute: Optional[dict], elapsed: float, env: dict) -> None:
    L: List[str] = []
    add = L.append
    bar = "=" * 86
    add(bar)
    add(f"GREEN LEARNING (Saab / PixelHop++ / DFT) -- {args.site}"
        + ("   [NEGATIVE CONTROL: PERMUTED LABELS]" if args.permute_labels else ""))
    add(bar)
    add("")
    add("WHAT THIS ARM TESTS")
    add("  A back-propagation-free representation with the opposite inductive bias to the")
    add("  DenseNet already run: closed-form channel-wise PCA convolution filters (Saab),")
    add("  max-pooling between hops, a supervised entropy feature test, and an L2 logistic")
    add("  readout.  Its claimed strength is the few-hundred-sample regime.  It cannot")
    add("  manufacture signal that is absent; if it also lands at chance, that is strong")
    add("  corroboration of the negative result rather than a wasted run -- the same")
    add("  harness recovers hospital-of-origin at AUC 0.992 from these same pixels.")
    add("")
    add("COHORT")
    add(f"  site {args.site}   patients {notes.get('patients_final')}   "
        f"images {notes.get('images_after_exam_filter')}   "
        f"patient prevalence {notes.get('patient_prevalence'):.3f}")
    add(f"  repeat-exam rule: keep_first_exam_only={args.keep_first_exam_only}; "
        f"dropped {notes.get('dropped_second_exam_images')} second-exam images")
    add("")
    add("PARTITIONS")
    add(f"  source           : {prov['fold_source']}")
    add(f"  regeneration     : {prov['regeneration_matches_csv']}")
    add(f"  note             : {prov['notes']}")
    if "clinical_arm_max_abs_diff_vs_cnn_run" in prov:
        add(f"  CNN join         : on {prov.get('cnn_join_keys')}; clinical-arm reproduction "
            f"max|diff| = {prov['clinical_arm_max_abs_diff_vs_cnn_run']:.3g}")
        add("  This is the end-to-end tripwire that the green arm really is on the published")
        add("  run's partitions: the clinical arm here uses the same features, the same C and")
        add("  the same inner-training patients, so on identical folds it must reproduce that")
        add("  run's score_clinical to machine precision.")
        if prov.get("cnn_join_is_cross_partition"):
            add("  *** CROSS-PARTITION JOIN: the clinical arm does NOT reproduce the published")
            add("  run, so the `cnn` row and every *_vs_cnn comparison below pair scores from")
            add("  DIFFERENT patient splits.  They are descriptive only, not like-for-like. ***")
    add(f"  seeds {list(range(args.seeds))}   outer folds in design {prov['folds_in_design']}   "
        f"folds actually run per seed {prov['folds_run_per_seed']}   "
        f"inner split 1/{args.inner_folds} of the training patients")
    if not prov["full_cohort_coverage"]:
        add("  PARTIAL SWEEP: not every cohort patient is scored out-of-fold, so the pooled")
        add("  numbers below describe only the folds that were run.  Not a headline result.")
    add("")
    add("ARCHITECTURE (all constants fixed a priori; none tuned on data)")
    shapes = cfg.hop_shapes(); rfs = cfg.receptive_fields()
    add(f"  input canvas     : {cfg.in_h} x {cfg.in_w}  (anisotropic stretch, NOT letterbox)")
    add(f"  kernel/stride/pad: {cfg.kernel} x {cfg.kernel}, stride 1, valid")
    add(f"  pooling          : {cfg.pool} x {cfg.pool} max, stride {cfg.pool}, after every hop")
    for h in range(cfg.n_hops):
        f_, p_ = cfg.caps[h]
        add(f"  hop {h+1}: conv {shapes[h]['conv_h']:>3} x {shapes[h]['conv_w']:<3} -> pool "
            f"{shapes[h]['pool_h']:>3} x {shapes[h]['pool_w']:<3}   receptive field "
            f"{rfs[h]:>3} px   feature nodes {f_:>4}   propagated {p_:>3}")
    add("  Aggregation per kept node: 2 x 4 grid of (mean, std) + 1 global max = 17 numbers.")
    if fold_records:
        e = fold_records[0]["extractor"]
        # The rank caps make this constant across folds, but the ENERGY FLOOR does not:
        # when the floor binds, the surviving-node count is a function of the fitting set
        # and changes fold to fold.  Quoting fold 0 alone would then understate or
        # overstate the bank every other fold actually used.
        all_dims = sorted({int(r["extractor"]["feature_dim"]) for r in fold_records})
        add(f"  MEASURED feature dimensionality: {e['feature_dim']}"
            + "  =  " + " + ".join(f"{d['feature_nodes']}x17={d['features_from_this_hop']}"
                                   for d in e["hops"])
            + ("" if len(all_dims) == 1 else
               f"   [fold 0 only -- IT VARIES ACROSS FOLDS: {all_dims}]"))
        for d in e["hops"]:
            add(f"    hop {d['hop']}: {d['candidate_nodes']} candidate nodes -> "
                f"{d['feature_nodes']} features / {d['propagated_nodes']} propagated; "
                f"bias d = {d['bias_d']:.4f}; kept energy {d['kept_energy_sum']:.6f}")
        h1e = e["hops"][0].get("hop1_local_fractional_energy", [])
        if h1e:
            add("    hop-1 fractional energy (first 8): "
                + ", ".join(f"{v:.5f}" for v in h1e[:8]))
        add(f"  Energy null guard: {cfg.energy_floor:g} on cumulative energy.  Candidates "
            "below the")
        add("  specification's literal 1e-6 floor, per hop: "
            + ", ".join(f"hop{d['hop']} {d['candidates_below_1e-6_cumulative_energy']}"
                        f"/{d['candidate_nodes']}" for d in e["hops"]))
        add("  The adjusted bias rides on every AC channel, so a hop-1 AC channel's patch")
        add("  energy is ~25 d1^2 and its local DC child takes essentially all of it, leaving")
        add("  its own AC grandchildren at cumulative energy ~1e-8.  A literal 1e-6 floor")
        add("  would therefore amputate the deep high-frequency speckle channels the rank")
        add("  rule exists to protect (measured with the full architecture at NCKUH: 41")
        add("  instead of 96 hop-2 nodes, 66 instead of 128 hop-3 nodes, 2244 features")
        add("  instead of the 4233 the specification itself states).")
        # Which of the two rules actually bound THIS run is a property of the run, not a
        # constant: --energy-floor 1e-6 makes the floor operative and then the sentence
        # "the rank caps are the operative selector" would be false in the deliverable.
        floor_bound = [f"hop{d['hop']} ({d['candidates_above_energy_floor']} candidates "
                       f"above the floor < cap {cfg.caps[i][0]})"
                       for i, d in enumerate(e["hops"])
                       if d["candidates_above_energy_floor"] < cfg.caps[i][0]]
        if floor_bound:
            add(f"  *** IN THIS RUN THE FLOOR ({cfg.energy_floor:g}), NOT THE RANK CAP, IS THE")
            add("  OPERATIVE SELECTOR at: " + "; ".join(floor_bound) + ".")
            add("  The node counts and the feature dimensionality above are therefore set by")
            add("  the floor, they are fitting-set dependent, and the specification's stated")
            add("  4233-feature bank does not describe this run. ***")
        else:
            add(f"  In this run every hop had more candidates above {cfg.energy_floor:g} than "
                "its rank cap")
            add("  keeps, so the caps are the operative selector and the floor did nothing but")
            add("  guard numerical nulls -- which is what it is there for.")
    ks = sorted({r["dft"]["n_selected"] for r in fold_records})
    add(f"  DFT survivors per fold (K = min({cfg.dft_k_cap}, n_train_images/"
        f"{cfg.dft_k_divisor})): {ks}")
    fs = [r["fold_seconds"] for r in fold_records]
    if fs:
        add(f"  Fit+transform+fit-classifier time per fold: median {np.median(fs):.1f} s, "
            f"range {min(fs):.1f}-{max(fs):.1f} s over {len(fs)} folds")
    # Both the inner-validation and the test transforms are held out and both are guarded
    # in run_one_fold; quote the max over both, not over the test folds alone.
    negs = [r["frac_negative_ac_response"][k]["overall"]
            for r in fold_records for k in ("inner_val", "test")]
    negs_fit = [r["frac_negative_ac_response"]["fit_set"]["overall"] for r in fold_records]
    dcmin = min(min(r["frac_negative_ac_response"][k]["min_dc_response_per_hop"])
                for r in fold_records for k in ("fit_set", "inner_val", "test"))
    add(f"  frac_negative_ac_response: max {max(negs):.3g} over held-out images "
        f"(inner-validation and test), {max(negs_fit):.3g} on the fitting images "
        f"(hard limit {cfg.max_neg_frac:g}).")
    add(f"  Counted over AC channels only; the smallest DC response anywhere was "
        f"{dcmin:.4g} (a negative DC would mean a negative input, a different fault).")
    add("  So the ReLU a deep cascade would apply is provably inactive, and this")
    add("  representation is EXACTLY a linear multiscale PCA composed with max-pooling;")
    add("  max-pool is its only nonlinearity.  That is stated rather than hidden, because")
    add("  it bounds what the arm could ever have found.")
    add("")
    add("WHAT WAS FITTED WHERE")
    add("  inner-training images : Saab filters/biases at every hop, energy ranking and")
    add("                          surviving-node lists, DFT clips and ranking, feature")
    add("                          scaler, logistic regression.")
    add("  inner-validation pats : the regularisation constant C and both operating")
    add("                          thresholds, and the fused arm's fusion layer.  Nothing else.")
    add("  test fold             : transformed and scored once.")
    add("  This is STRICTER than the specification (which refits the extractor on the full")
    add("  outer-training set); it is used because the CNN arm also never refits on")
    add("  outer-train, so the two arms see identical data.")
    add("")
    add("SELECTED HYPER-PARAMETER")
    cs = [r["selected_C"]["green"] for r in fold_records]
    add(f"  green LR C, per fold: {sorted(set(cs))}   (grid {list(C_GRID)}, chosen by "
        f"inner-validation PATIENT AUC, ties to the smaller C)")
    add("")
    add(bar)
    add("RESULTS -- patient level, pooled out-of-fold, patient-cluster bootstrap "
        f"(B={args.bootstrap}, seed {args.bootstrap_seed})")
    add(bar)
    add(f"  {'arm':<10}{'AUC [95% CI]':<26}{'mean+/-SD':<18}{'bal.acc [95% CI]'}")
    for a in [x for x in GL_PATIENT_ARMS + ["cnn"] if x in boot_pat]:
        m = across_pat[a]["auc"]
        add(f"  {a:<10}{_fmt_ci(boot_pat[a]['auc']):<26}"
            f"{m['mean']:.3f} +/- {m['sd']:.3f}{'':<3}"
            f"{_fmt_ci(boot_pat[a]['balanced_accuracy'])}")
    if "cnn" in boot_pat:
        add("  `cnn` is the PUBLISHED DenseNet121 run, re-bootstrapped here from its own")
        add("  out-of-fold table on exactly the patients and seeds this script scored, so it")
        add("  is a like-for-like reference rather than a number copied from another report.")
    add("")
    add("RESULTS -- image level (patients are still the bootstrap resampling unit)")
    add(f"  {'arm':<10}{'AUC [95% CI]':<26}{'mean+/-SD over seeds'}")
    for a in [x for x in GL_IMAGE_ARMS if x in boot_img]:
        m = across_img[a]["auc"]
        add(f"  {a:<10}{_fmt_ci(boot_img[a]['auc']):<26}{m['mean']:.3f} +/- {m['sd']:.3f}")
    add("  Image-level intervals are reported for descriptive completeness ONLY.  The")
    add("  headline is the patient-level number: 1-4 views of one child on one day are")
    add("  near-duplicates, so an image-level CI treated as n independent units would be")
    add("  roughly 40 % too narrow and a chance result could acquire an interval")
    add("  excluding 0.50.")
    add("")
    if "green" in boot_pat and "geom" in boot_pat:
        g, m = boot_pat["green"]["auc"], boot_pat["geom"]["auc"]
        verdict = ("BEATS" if g["point_estimate"] > m["point_estimate"] else
                   "DOES NOT BEAT")
        add("THE CONFOUND SENTENCE (required, in one place)")
        add(f"  Green learning reaches patient AUC {_fmt_ci(g)} and {verdict} the pre-declared")
        add(f"  geometry/overlay confound arm at {_fmt_ci(m)}.  The confound arm uses only")
        add("  native crop width and height, aspect, colour-overlay fraction and mean luma --")
        add("  no tissue texture at all.  Crop height alone is worth patient AUC 0.600 at")
        add("  NCKUH, uncorrelated there with kidney size, i.e. an acquisition artifact.")
        if fold_records and "interp_branch_area_fraction" in fold_records[0]:
            fr = [r["interp_branch_area_fraction"]["test"] for r in fold_records]
            add(f"  RESIDUAL ROUTE, stated because it is not fully closed: the resize kernel is")
            add(f"  still chosen per image by native size (INTER_AREA when the crop is at least")
            add(f"  {cfg.in_w}x{cfg.in_h}, INTER_LINEAR otherwise -- the specification's own rule), and the")
            add(f"  two kernels leave different high-frequency content.  INTER_AREA fraction in")
            add(f"  the test folds: {min(fr):.3f}-{max(fr):.3f}.  On the whole cohort that branch")
            add("  indicator alone carries patient AUC 0.556 at NCKUH and 0.520 at CYCH, so a")
            add("  green result within ~0.06 of chance cannot be separated from it.")
        add("")
    add("TOP-20 DFT-SELECTED FEATURES (fold 0 of the first seed; names carry hop / node / statistic)")
    if fold_records:
        for t in fold_records[0]["dft"]["top20"]:
            add(f"  {t['rank']:>3}  {t['feature']:<34} entropy {t['dft_entropy']:.4f}")
        names = [t["feature"] for t in fold_records[0]["dft"]["top20"]]
        n_h1n0 = sum(1 for x in names if x.startswith("h1_n0_"))
        add(f"  {n_h1n0} of 20 come from h1_n0 (the hop-1 DC channel = local mean brightness).")
        if n_h1n0 >= 10:
            add("  MORE THAN HALF THE SELECTED FEATURES ARE LOCAL BRIGHTNESS.  This arm must be")
            add("  described as a brightness detector, not a texture model.")
    add("")
    if comparisons:
        add(bar)
        add("PAIRED COMPARISONS")
        add(bar)
        fam = sorted(k for k, v in comparisons.items() if v.get("in_holm_family", True))
        out_fam = sorted(set(comparisons) - set(fam))
        add(f"  Family adjusted with Holm (m = {len(fam)}): {fam}")
        add("  This is the GREEN-LEARNING family only.  The published CNN run adjusted its")
        add("  own four comparisons within its own family; those adjusted p-values are")
        add("  unchanged by this script.  A reader who wants one family over all nine")
        add("  comparisons should multiply accordingly -- that would be more conservative,")
        add("  never less.")
        if out_fam:
            add(f"  EXCLUDED from the family: {out_fam}.  These pair scores from different")
            add("  partitions (see PARTITIONS above), so they are descriptive only and must")
            add("  not be interpreted; keeping them in would have enlarged m and made the")
            add("  interpretable comparisons artificially conservative.  Their adjusted p is")
            add("  reported as nan rather than as a number nobody may read.")
        add("")
        for k, v in comparisons.items():
            b = v["paired_bootstrap"]
            add(f"  {k}" + ("" if v.get("in_holm_family", True)
                            else "   [CROSS-PARTITION -- descriptive only, not in the family]"))
            add(f"    delta AUC (paired patient-cluster bootstrap) "
                f"{b['delta_observed']:+.3f} [{b['ci_lo']:+.3f}, {b['ci_hi']:+.3f}]  "
                f"p = {b['p_bootstrap']:.4f} (floor {b['p_bootstrap_floor']:.4f})")
            add(f"    Holm-adjusted p (bootstrap, primary) = "
                f"{holm.get(k, float('nan')):.4f}")
            d = v["delong"]
            add(f"    DeLong (secondary, anti-conservative on CV-pooled scores): "
                f"mean delta {d['mean_delta']:+.3f}, max p over seeds "
                f"{d['max_p_over_seeds']:.4f}, undecided seeds {d['n_seeds_without_valid_p']}")
        add("")
        add("  DeLong's variance assumes two FIXED models scored on independent")
        add("  observations.  Here the scores are pooled across outer folds, so k models")
        add("  produced them and every patient helped train the models that scored the")
        add("  others; the DeLong SE is therefore anti-conservative.  The paired patient-")
        add("  cluster bootstrap is the primary inference.")
    else:
        add("PAIRED COMPARISONS AGAINST THE PUBLISHED CNN RUN: SKIPPED.")
        add("  results_single_center/%s/oof_predictions_patient.csv was not found, or the "
            "partitions" % args.site)
        add("  are not the CNN run's (permuted labels re-stratify the folds), so no honest")
        add("  paired join exists.  The green-learning numbers above stand on their own.")
    add("")
    if permute:
        add(bar)
        add("NEGATIVE CONTROL VERDICT")
        add(bar)
        add(f"  pooled patient AUC on permuted labels : {permute['pooled_patient_auc']:.3f}  "
            f"95% CI [{permute['ci95'][0]:.3f}, {permute['ci95'][1]:.3f}]")
        add(f"  per-fold: {[round(a, 3) for a in permute['per_fold_patient_auc']]}")
        add(f"  VERDICT: {permute['verdict']}")
        add("")
    add("HOW TO READ THIS")
    add("  Read the green number against three things at once: 0.50 (chance), the CNN's")
    add("  0.544 (NCKUH) / 0.526 (CYCH), and the geom confound arm above.  A green arm at")
    add("  chance means two extractors with opposite inductive biases fail identically on")
    add("  data where the same harness recovers the hospital at 0.99 -- the most")
    add("  parsimonious reading is that the cropped ROI does not carry the inflammation")
    add("  label.  A green arm above the CNN but at or below geom is a confound detector.")
    add("  A different feature extractor cannot create signal that is not in the data.")
    add("")
    add(f"  elapsed {elapsed/60:.1f} min   python {env['python']}   numpy {env['numpy']}   "
        f"sklearn {env['sklearn']}   opencv {env['opencv']}")
    add(f"  command: {env['argv']}")
    add(bar)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


# ==================================================================================
# 9.  Driver
# ==================================================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Green-learning (Saab / PixelHop++ / DFT) arm on the harness's own "
                    "patient-level folds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--site", required=True, choices=["CYCH", "NCKUH"])
    p.add_argument("--out", required=True)
    p.add_argument("--manifest", default=str(S.DEFAULT_MANIFEST))
    p.add_argument("--keep-first-exam-only", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--folds", type=int, default=5,
                   help="Outer patient-level folds in the CV DESIGN (must stay 5 to replay "
                        "the CNN run's frozen partitions).")
    p.add_argument("--limit-folds", type=int, default=0,
                   help="Run only the first N of the --folds folds per seed (0 = all). Used "
                        "by --smoke so the smoke run still replays the real partitions.")
    p.add_argument("--seeds", type=int, default=3, help="Repetitions; seeds 0..n-1.")
    p.add_argument("--inner-folds", type=int, default=4)
    p.add_argument("--permute-labels", action="store_true",
                   help="Negative control: permute PATIENT labels before any split is drawn.")
    p.add_argument("--permute-seed", type=int, default=12345)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--bootstrap-seed", type=int, default=20260909)
    p.add_argument("--clinical-C", type=float, default=1.0)
    p.add_argument("--fusion-C", type=float, default=1.0)
    p.add_argument("--chunk", type=int, default=0,
                   help="Images per transform chunk (0 = architecture default).")
    p.add_argument("--energy-floor", type=float, default=0.0,
                   help="Cumulative-energy null guard (0 = architecture default 1e-12). "
                        "Pass 1e-6 for the specification's literal text; see GLConfig.")
    p.add_argument("--replay-folds", action=argparse.BooleanOptionalAction, default=True,
                   help="Replay results_single_center/<SITE>/fold_assignments.csv when it "
                        "covers the requested grid.")
    p.add_argument("--smoke", action="store_true",
                   help="1 seed, 2 folds, 48x80 canvas, small node caps, 200 bootstrap draws.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = smoke_config() if args.smoke else GLConfig()
    if args.smoke:
        args.seeds, args.limit_folds = 1, 2
        args.bootstrap = min(args.bootstrap, 200)
    if args.chunk:
        cfg.chunk = int(args.chunk)
    if args.energy_floor:
        cfg.energy_floor = float(args.energy_floor)
    limit = int(args.limit_folds) or None
    n_run = args.folds if limit is None else min(limit, args.folds)

    t0 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seeds))
    # --seeds 0, --folds 0 or a negative --limit-folds all produce an empty run grid.
    # Left unguarded the run does the cohort load and the whole-site preprocessing first
    # and then dies ~10 s later inside pd.concat with "No objects to concatenate", which
    # names neither the flag nor the reason.  Fail immediately and say which flag is wrong.
    S._require(len(seeds) >= 1 and n_run >= 1,
               f"empty run grid: --seeds {args.seeds} x {n_run} folds per seed leaves "
               "nothing to run.  --seeds must be >= 1 and --folds / --limit-folds must "
               "leave at least one fold.")

    S._selftest_stats(verbose=True)
    S.set_all_seeds(0)

    # ------------------------------------------------------------------- cohort --
    img_df, notes = S.load_cohort(Path(args.manifest), args.site, args.keep_first_exam_only)
    pat = S.build_patient_table(img_df)
    if args.permute_labels:
        rng = np.random.default_rng(args.permute_seed)
        shuffled = pat["label"].to_numpy().copy()
        rng.shuffle(shuffled)
        truth = pat["label"].to_numpy().copy()
        pat = pat.assign(label=shuffled)
        lut = dict(zip(pat["patient_id"], pat["label"]))
        img_df = img_df.assign(label=img_df["patient_id"].map(lut).astype(int))
        notes["control"] = "N1 negative: patient labels permuted before any split"
        notes["permute_seed"] = int(args.permute_seed)
        notes["label_agreement_with_truth"] = float((shuffled == truth).mean())
    notes["patients_final"] = int(len(pat))
    notes["patient_prevalence"] = float(pat["label"].mean())

    print(f"[green] site={args.site} patients={len(pat)} images={len(img_df)} "
          f"prevalence={pat['label'].mean():.3f} "
          f"canvas={cfg.in_h}x{cfg.in_w} caps={cfg.caps}"
          + ("  [PERMUTED LABELS]" if args.permute_labels else ""))

    # -------------------------------------------------------------- fold plan --
    allow_replay = bool(args.replay_folds) and not args.permute_labels
    # The frozen fold_assignments.csv validates only against the PATIENT set, and the
    # repeat-exam rule drops images, never patients (4 images at each site).  So a run with
    # --no-keep-first-exam-only replays those partitions cleanly, the regeneration tripwire
    # agrees (the patient table is identical), and even the clinical-reproduction tripwire
    # passes (clinical covariates are taken from the earliest exam either way) -- while the
    # green arm is fitted and scored on images the published CNN run never saw.  Refuse.
    S._require(not (allow_replay and not args.keep_first_exam_only),
               "results_single_center/<SITE>/fold_assignments.csv was produced under "
               "--keep-first-exam-only.  Replaying it with --no-keep-first-exam-only would "
               "score the green arm on the repeat-exam images the published CNN run "
               "excluded, and no tripwire in this script can see that (the patient set, the "
               "regenerated partitions and the clinical covariates are all unchanged).  Use "
               "--keep-first-exam-only, or --no-replay-folds if you really want that cohort.")
    plan, prov = build_fold_plan(pat, args.site, seeds, args.folds, args.inner_folds,
                                 allow_replay=allow_replay, limit_folds=limit)
    prov["folds_in_design"] = int(args.folds)
    prov["folds_run_per_seed"] = int(n_run)
    prov["full_cohort_coverage"] = bool(n_run == args.folds)
    prov["keep_first_exam_only"] = bool(args.keep_first_exam_only)
    print(f"[green] partitions: {prov['fold_source']}  "
          f"(regeneration agrees: {prov['regeneration_matches_csv']})")
    print(f"[green] {prov['notes']}")
    if not prov["full_cohort_coverage"]:
        print(f"[green] PARTIAL SWEEP: only {n_run} of {args.folds} outer folds per seed are "
              "run, so not every cohort patient is scored out-of-fold.  The pooled AUCs, the "
              "CIs and any negative-control verdict below describe ONLY the folds that ran "
              "and are NOT a headline result.")

    # --------------------------------------------------------------- image cache --
    cache, geom, cache_paths = build_image_cache(img_df, cfg)
    print(f"[green] preprocessed {cache.shape[0]} images to {cache.shape[1]}x{cache.shape[2]} "
          f"float32 ({cache.nbytes/1e6:.0f} MB cache); mean overlay fraction "
          f"{geom[:, 3].mean():.4f}")

    # ------------------------------------------------------------------- folds --
    pat_rows, img_rows, fold_records = [], [], []
    for spec in tqdm(plan, desc=f"{args.site} folds", unit="fold"):
        po, io, rec = run_one_fold(cfg, args.site, spec["seed"], spec["fold"], pat, img_df,
                                   cache, geom, cache_paths,
                                   {k: spec[k] for k in ("inner_train", "inner_val", "test")},
                                   args.fusion_C, args.clinical_C)
        pat_rows.append(po); img_rows.append(io); fold_records.append(rec)
        auc = S.fast_auc(po["label"].to_numpy(), po["score_green"].to_numpy())
        aucm = S.fast_auc(po["label"].to_numpy(), po["score_geom"].to_numpy())
        tqdm.write(f"  s{spec['seed']}f{spec['fold']}  test patients {len(po)}  "
                   f"green {auc:.3f}  geom {aucm:.3f}  K={rec['dft']['n_selected']}  "
                   f"C={rec['selected_C']['green']:g}  {rec['fold_seconds']:.0f}s")

    pat_pred = pd.concat(pat_rows, ignore_index=True)
    img_pred = pd.concat(img_rows, ignore_index=True)

    # ---- integrity: one row per patient per seed, and no patient scored twice ----
    for s in seeds:
        d = pat_pred[pat_pred["seed"] == s]
        dup = d["patient_id"][d["patient_id"].duplicated()].tolist()
        S._require(not dup, f"seed {s}: patient scored in more than one fold: {dup[:5]}")
        di = img_pred[img_pred["seed"] == s]
        S._require(not di["filename"].duplicated().any(),
                   f"seed {s}: an image was scored in more than one fold")
        if n_run == args.folds:
            # A duplicate-only check passes when a patient appears ZERO times, so both are
            # tested.  With a partial fold sweep (--limit-folds / --smoke) full coverage is
            # not expected and the pooled numbers cover only the folds actually run.
            S._require(set(d["patient_id"]) == set(pat["patient_id"]),
                       f"seed {s}: {len(set(pat['patient_id']) - set(d['patient_id']))} "
                       "cohort patients have no out-of-fold prediction")
            S._require(set(di["filename"]) == set(img_df["filename"]),
                       f"seed {s}: some cohort images have no out-of-fold prediction")
    truth = dict(zip(pat["patient_id"], pat["label"]))
    bad = [p for p, l in zip(pat_pred["patient_id"], pat_pred["label"]) if truth.get(p) != l]
    S._require(not bad, f"patient label disagreement between image rows and patient table: {bad[:3]}")

    fa_rows = []
    for r in fold_records:
        for sp, ids in r.pop("_pids").items():
            for pid in ids:
                fa_rows.append({"site": args.site, "seed": r["seed"], "fold": r["fold"],
                                "patient_id": pid, "split": sp})
    pd.DataFrame(fa_rows).to_csv(out / "fold_assignments.csv", index=False, encoding="utf-8")
    pat_pred.to_csv(out / "oof_predictions_patient.csv", index=False, encoding="utf-8")
    img_pred.to_csv(out / "oof_predictions_image.csv", index=False, encoding="utf-8")

    # -------------------------------------------------- pooled inference (harness) --
    # ARMS / IMAGE_ARMS are module globals resolved at call time, so rebinding them here
    # makes build_oof_store / cluster_bootstrap / per_seed_metrics work on this arm's
    # columns without editing the harness.
    cnn_df = None if args.permute_labels else read_cnn_patient_oof(args.site)
    merged_pat = pat_pred
    cmp_arms = []
    if cnn_df is not None:
        keys = ["site", "seed", "patient_id"] + (["fold"] if "replayed" in prov["fold_source"] else [])
        m = pat_pred.merge(cnn_df[keys + ["label", "score_cnn", "score_clinical",
                                          "pred_cnn", "pred_cnn_sens90",
                                          "thr_cnn", "thr_cnn_sens90"]],
                           on=keys, how="inner", suffixes=("", "_cnn"))
        if len(m) == len(pat_pred):
            S._require(bool((m["label"] == m["label_cnn"]).all()),
                       "label disagreement between the green and CNN out-of-fold tables")
            merged_pat = m
            cmp_arms = ["cnn"]
            clin_gap = float(np.max(np.abs(m["score_clinical"] - m["score_clinical_cnn"])))
            prov["clinical_arm_max_abs_diff_vs_cnn_run"] = clin_gap
            # End-to-end tripwire.  The clinical arm here is fitted with the same features,
            # the same C and the same inner-training patients as the published run's, so on
            # replayed partitions it must reproduce that run's scores to machine precision.
            # If it does not, this arm is NOT on the CNN run's folds and no paired
            # comparison below would be meaningful.
            prov["cnn_join_keys"] = list(keys)
            prov["cnn_join_is_cross_partition"] = bool(clin_gap >= 1e-6)
            if "replayed" in prov["fold_source"]:
                S._require(clin_gap < 1e-6,
                           f"the clinical arm does not reproduce the published run's "
                           f"clinical scores (max|diff| = {clin_gap:.3g}); the partitions "
                           "or the fitting convention differ, so the paired comparison "
                           "against the CNN would not be like-for-like.")
            elif clin_gap >= 1e-6:
                # Not replayed, so `fold` is NOT a merge key and nothing above forces the two
                # runs onto the same partitions.  The join still pairs one green and one CNN
                # score per (patient, seed), which is why it silently succeeds -- but those
                # two scores were produced by models trained on DIFFERENT patient splits, and
                # the `cnn` row plus green_vs_cnn / fused_vs_cnn would then be published as
                # like-for-like when they are not.  Say so, loudly and in summary.txt.
                print("[green] WARNING: partitions were NOT replayed and the clinical arm does "
                      f"not reproduce the published run (max|diff| = {clin_gap:.3g}).  The CNN "
                      "comparisons below pair scores from DIFFERENT partitions and are NOT "
                      "like-for-like; treat them as descriptive only.")
            print(f"[green] joined {len(m)} rows to the published CNN run on {keys}; "
                  f"clinical-arm reproduction max|diff| = {clin_gap:.2e}")
        else:
            cnn_df = None
            print(f"[green] CNN out-of-fold table found but only {len(m)}/{len(pat_pred)} rows "
                  "join; skipping the paired comparison.")

    pat_arms = GL_PATIENT_ARMS + cmp_arms
    old_arms, old_img = S.ARMS, S.IMAGE_ARMS
    S.ARMS, S.IMAGE_ARMS = pat_arms, GL_IMAGE_ARMS
    for a in cmp_arms:
        S.ARM_LABELS[a] = GL_ARM_LABELS[a]
    try:
        store = S.build_oof_store(merged_pat, img_pred, seeds)
        boot_pat = S.cluster_bootstrap(store, seeds, args.bootstrap, args.bootstrap_seed,
                                       "patient")
        boot_img = S.cluster_bootstrap(store, seeds, args.bootstrap, args.bootstrap_seed,
                                       "image")
        per_seed = S.per_seed_metrics(merged_pat, img_pred, seeds)
        across_pat = S.summarise_across_seeds(per_seed, "patient_level", pat_arms)
        across_img = S.summarise_across_seeds(per_seed, "image_level", GL_IMAGE_ARMS)
        pooling = S.pooled_vs_per_fold_auc(merged_pat, seeds, pat_arms)

        pairs = [("green", "geom"), ("green", "clinical"), ("fused", "green")]
        if cmp_arms:
            pairs += [("green", "cnn"), ("fused", "cnn")]
        cross_partition = bool(prov.get("cnn_join_is_cross_partition"))
        comparisons, pvals = {}, {}
        for i, (a, b) in enumerate(pairs):
            bd = S.paired_bootstrap_delta(store, seeds, a, b, args.bootstrap,
                                          args.bootstrap_seed + 104 + i)
            per = {}
            for s in seeds:
                ps = merged_pat[merged_pat["seed"] == s]
                per[str(s)] = S.delong_test(ps["label"].to_numpy(),
                                            ps[f"score_{a}"].to_numpy(),
                                            ps[f"score_{b}"].to_numpy())
            pv = [v["p"] for v in per.values() if np.isfinite(v["p"])]
            dl = [v["delta"] for v in per.values() if np.isfinite(v["delta"])]
            comparisons[f"{a}_vs_{b}"] = {
                "comparison": f"AUC({GL_ARM_LABELS[a]}) - AUC({GL_ARM_LABELS[b]})",
                "paired_bootstrap": bd,
                "delong": {
                    "per_seed": per, "n_seeds": len(seeds),
                    "n_seeds_without_valid_p": int(len(seeds) - len(pv)),
                    "max_p_over_seeds": float(np.max(pv)) if pv else float("nan"),
                    "min_p_over_seeds": float(np.min(pv)) if pv else float("nan"),
                    "mean_delta": float(np.mean(dl)) if dl else float("nan"),
                    "significant_at_every_seed_0.05": bool(
                        len(pv) == len(seeds) and pv and max(pv) < 0.05),
                },
            }
            comparisons[f"{a}_vs_{b}"]["in_holm_family"] = not (
                cross_partition and "cnn" in (a, b))
            pvals[f"{a}_vs_{b}"] = bd["p_bootstrap"]
        # Holm must be applied to the family of comparisons actually being INTERPRETED.
        # On a cross-partition join the script has already declared green_vs_cnn and
        # fused_vs_cnn not like-for-like; leaving them in the family would still let two
        # comparisons nobody may interpret enlarge m from 3 to 5 and make the three valid
        # green comparisons more conservative than the design calls for (a raw p of 0.02
        # would be adjusted to 0.10 instead of 0.06).  Drop them from the family, keep
        # them in metrics.json as descriptive, and report their adjusted p as NaN.
        family = {k: v for k, v in pvals.items()
                  if comparisons[k]["in_holm_family"]}
        holm = S.holm_bonferroni(family)
        for k in pvals:
            holm.setdefault(k, float("nan"))
        holm_delong = S.holm_bonferroni(
            {k: v["delong"]["max_p_over_seeds"] for k, v in comparisons.items()
             if v["in_holm_family"]})
        for k in pvals:
            holm_delong.setdefault(k, float("nan"))
    finally:
        S.ARMS, S.IMAGE_ARMS = old_arms, old_img

    # ------------------------------------------------------- negative-control verdict --
    permute_block = None
    if args.permute_labels:
        # The headline is the bootstrap's own point estimate (mean over seeds of the
        # per-seed pooled AUC), NOT an AUC over the stacked seed rows -- stacking would
        # score every patient once per seed and treat the repetitions as extra patients.
        pooled = boot_pat["green"]["auc"]["point_estimate"]
        lo, hi = boot_pat["green"]["auc"]["ci_lo"], boot_pat["green"]["auc"]["ci_hi"]
        per_fold = []
        for (_s, _f), g in pat_pred.groupby(["seed", "fold"]):
            if g["label"].nunique() > 1:
                per_fold.append(S.fast_auc(g["label"].to_numpy(),
                                           g["score_green"].to_numpy()))
        permute_block = {
            "pooled_patient_auc": float(pooled),
            "ci95": [float(lo), float(hi)],
            "per_fold_patient_auc": [float(v) for v in per_fold],
            "verdict": ("PASS - consistent with a leak-free green-learning arm"
                        if lo <= 0.5 <= hi else
                        "FAIL - the 95% CI excludes 0.50 on permuted labels; something in "
                        "the Saab/DFT/classifier chain is fitted outside the training fold"),
        }

    gl_figures(pat_pred, img_pred, {"patient_level": across_pat, "image_level": across_img},
               out, args.site, seeds[0])

    env = {"python": platform.python_version(), "numpy": np.__version__,
           "sklearn": __import__("sklearn").__version__, "opencv": cv2.__version__,
           "pandas": pd.__version__, "platform": platform.platform(),
           "argv": " ".join([Path(sys.argv[0]).name] + sys.argv[1:])}
    elapsed = time.time() - t0

    metrics = {
        "arm": "green learning (Saab / PixelHop++ + DFT + L2 logistic regression)",
        "site": args.site,
        "permuted_labels": bool(args.permute_labels),
        "config": {**vars(args), "architecture": {
            "in_h": cfg.in_h, "in_w": cfg.in_w, "kernel": cfg.kernel, "pool": cfg.pool,
            "caps": [list(c) for c in cfg.caps], "hop_shapes": cfg.hop_shapes(),
            "receptive_fields_px": cfg.receptive_fields(),
            "grid": [cfg.grid_rows, cfg.grid_cols], "dft_bins": cfg.dft_bins,
            "C_grid": list(C_GRID), "chunk": cfg.chunk,
            "patch_samples": list(cfg.patch_samples)}},
        "cohort_notes": notes,
        "partitions": prov,
        "fold_records": fold_records,
        "per_seed": per_seed,
        "across_seeds_patient": across_pat,
        "across_seeds_image": across_img,
        "bootstrap_patient": boot_pat,
        "bootstrap_image": boot_img,
        "pooled_vs_per_fold": pooling,
        "comparisons": comparisons,
        "holm_family": sorted(k for k, v in comparisons.items()
                              if v.get("in_holm_family", True)),
        "holm_family_excluded_cross_partition": sorted(
            k for k, v in comparisons.items() if not v.get("in_holm_family", True)),
        "holm_adjusted_p_bootstrap_primary": holm,
        "holm_adjusted_p_delong_secondary": holm_delong,
        "negative_control": permute_block,
        "elapsed_min": round(elapsed / 60, 2),
        "env": env,
    }
    (out / "metrics.json").write_text(
        json.dumps(S._json_sanitize(metrics), indent=2, default=S._json_default),
        encoding="utf-8")
    write_summary(out / "summary.txt", args, cfg, notes, prov, fold_records, across_pat,
                  across_img, boot_pat, boot_img, comparisons, holm, permute_block,
                  elapsed, env)

    # ------------------------------------------------------------------- console --
    print("\n" + "=" * 86)
    print(f"GREEN LEARNING -- {args.site}"
          + ("  [PERMUTED-LABEL NEGATIVE CONTROL]" if args.permute_labels else ""))
    print(f"  partitions: {prov['fold_source']}")
    fdim = fold_records[0]["extractor"]["feature_dim"]
    all_fdim = sorted({int(r["extractor"]["feature_dim"]) for r in fold_records})
    dims = " + ".join(f"hop{d['hop']} {d['feature_nodes']}x17={d['features_from_this_hop']}"
                      for d in fold_records[0]["extractor"]["hops"])
    print(f"  feature bank: {fdim} = {dims}"
          + ("" if len(all_fdim) == 1 else f"  [fold 0; varies across folds: {all_fdim}]")
          + f";  DFT keeps {sorted({r['dft']['n_selected'] for r in fold_records})}")
    print(f"  fold time: median {np.median([r['fold_seconds'] for r in fold_records]):.0f} s")
    print("  patient level, pooled out-of-fold:")
    for a in [x for x in GL_PATIENT_ARMS + cmp_arms if x in boot_pat]:
        print(f"    {a:<10}AUC {_fmt_ci(boot_pat[a]['auc'])}   "
              f"mean+/-SD {across_pat[a]['auc']['mean']:.3f} +/- {across_pat[a]['auc']['sd']:.3f}")
    if comparisons:
        print("  paired comparisons (Holm-adjusted bootstrap p):")
        for k, v in comparisons.items():
            tag = "" if v.get("in_holm_family", True) else "  [cross-partition, not adjusted]"
            print(f"    {k:<20}delta {v['paired_bootstrap']['delta_observed']:+.3f} "
                  f"[{v['paired_bootstrap']['ci_lo']:+.3f}, {v['paired_bootstrap']['ci_hi']:+.3f}]"
                  f"  p_holm {holm.get(k, float('nan')):.4f}{tag}")
    if permute_block:
        print(f"  NEGATIVE CONTROL: pooled AUC {permute_block['pooled_patient_auc']:.3f} "
              f"[{permute_block['ci95'][0]:.3f}, {permute_block['ci95'][1]:.3f}]")
        print(f"  VERDICT: {permute_block['verdict']}")
    print(f"  elapsed {elapsed/60:.1f} min -> {out}")
    print("=" * 86)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
