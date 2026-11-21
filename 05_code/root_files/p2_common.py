"""Shared paths and helpers for the second NCKUH paper (probe-routed, group-specific models).

Nothing here writes into the first paper's folders; earlier experiment outputs are read-only inputs.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "02_experiments"
REGISTRY = EXP / "NCKUH_revised_383_audit_20261111"
CACHE = EXP / "NCKUH_revised_383_contour_cache_20261111"
OLD_ROUTER = EXP / "NCKUH_revised_383_probe_router_20261112"
P2 = EXP / "NCKUH_P2_probe_routed_20261112"
WORKBOOK_V2 = ROOT / "2004-2019-NCKUH-亞大(2).xlsx"
import os

# Paper-2 repeated draws: same three fold seeds as paper 1's draws. P2_DRAW selects one (0, 1, 2).
# Unset P2_DRAW keeps the original single-partition run (seed 20261112, outputs in P2 itself).
DRAW_SEEDS = (20261102, 20261103, 20261104)
DRAW = os.environ.get("P2_DRAW")
if DRAW is None:
    SEED = 20261112
    P2_OUT = P2
else:
    SEED = DRAW_SEEDS[int(DRAW)]
    P2_OUT = P2 / {"paper1": "draws_paper1_split", "ruler1": "draws_ruler1", "conv255": "draws_conv255"}.get(os.environ.get("P2_SPLIT"), "draws") / f"draw{int(DRAW)}_seed{SEED}"
    P2_OUT.mkdir(parents=True, exist_ok=True)
# Split mode: "fold5" (superseded five-fold design) or "paper1" (paper 1: development 42 and held out 42 patients,
# train / development (scored) / held out (never trained on nor scored). Held-out patients get fold -1.
SPLIT = os.environ.get("P2_SPLIT", "fold5")
FOLD_IDS = (0,) if SPLIT in ("paper1", "ruler1", "conv255") else (0, 1, 2, 3, 4)
PAPER1_DEVELOPMENT, PAPER1_HELD_OUT = 42, 42  # paper 1: 126 train / 42 development / 42 held out


def assign_folds(patients, seed=None):
    """Return {patient_alias: fold} for a pool DataFrame (columns DMSA_APN, workbook_probe, patient_alias)."""
    from sklearn.model_selection import StratifiedKFold, train_test_split
    seed = SEED if seed is None else seed
    strata = (patients.DMSA_APN.astype(str) + "|" + patients.workbook_probe.replace({"both": "convex"}).astype(str)).to_numpy()
    idx = np.arange(len(patients))
    fold = np.full(len(patients), -1)
    if SPLIT == "conv255":
        return _conv255_folds(patients, seed)
    if SPLIT in ("paper1", "ruler1"):
        rest, held = train_test_split(idx, test_size=PAPER1_HELD_OUT, random_state=seed, stratify=strata)
        train, dev = train_test_split(rest, test_size=PAPER1_DEVELOPMENT, random_state=seed + 1, stratify=strata[rest])
        fold[dev] = 0
        fold[train] = 1
        if SPLIT == "ruler1":
            fold = _apply_first_study_ruler(patients, fold, seed)
    else:
        for k, (_, hold) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(idx, strata)):
            fold[hold] = k
    return dict(zip(patients.patient_alias, fold))


def first_study_development_aliases(seed):
    """Paper-2 aliases of the first study's development patients for one of its three splits."""
    import nckuh_dmsa_fullcohort_fullframe_20261103 as base
    import nckuh_dmsa_multiquery_mil_20261103 as old_mil
    link = pd.read_csv(REGISTRY / "patient_linkage_private.csv", dtype=str).set_index("patient_id").patient_alias
    reference = old_mil.load_reference()[0]
    to_new = dict(zip(reference.patient_alias, reference.patient_id.astype(str).map(link)))
    assign = pd.read_csv(base.OUT / "patient_assignments.csv")
    development = assign[(assign.seed == seed) & assign.role.eq("development")].patient_alias.map(to_new)
    return {a for a in development if isinstance(a, str)}


def _apply_first_study_ruler(patients, fold, seed):
    """Convex group: development = the first study's development patients, training = all other convex patients.
    Linear-only patients keep the 188/42/42 split; a patient with both probes is never a second development patient."""
    ruler = first_study_development_aliases(seed)
    probe = patients.workbook_probe.to_numpy()
    out = fold.copy()
    for i, alias in enumerate(patients.patient_alias):
        if alias in ruler:
            out[i] = 0
        elif probe[i] == "convex":
            out[i] = 1
        elif probe[i] == "both" and fold[i] == 0:
            out[i] = 1
    return out


STATE_SIDES = {"N": (0, 0), "L": (1, 0), "R": (0, 1), "B": (1, 1)}  # (left, right)


def pool_mask(images):
    """Images that may be fitted or scored. conv255: every cohort patient (the 388 patients of the (2) workbook that have images,
    including the earlier exploratory-test role and the patients added on 2026-11-17, role added_388) with convex- or linear-routed images."""
    base = images.role.isin(("train", "development")) & images.route.isin(("convex", "linear"))
    if SPLIT != "conv255":
        return base
    return images.route.isin(("convex", "linear"))  # route = workbook Probe for single-probe patients, detector for patients with both probes


def _conv255_folds(patients, seed):
    """Convex group (convex-only and both): stratified split into 42 development, 42 held-out (-1) and 171 training patients.
    Linear-only patients (133): stratified split into 22 development, 22 held-out and 89 training patients (the convex proportions)."""
    from sklearn.model_selection import train_test_split
    strata_all = (patients.DMSA_APN.astype(str)).to_numpy()
    probe = patients.workbook_probe.to_numpy()
    idx = np.arange(len(patients))
    fold = np.full(len(patients), -1)

    def split(members, n_held, n_dev):
        rest, held = train_test_split(members, test_size=n_held, random_state=seed, stratify=strata_all[members])
        train, dev = train_test_split(rest, test_size=n_dev, random_state=seed + 1, stratify=strata_all[rest])
        fold[dev], fold[train] = 0, 1
        return held

    split(idx[np.isin(probe, ("convex", "both"))], PAPER1_HELD_OUT, PAPER1_DEVELOPMENT)
    split(idx[probe == "linear"], 22, 22)
    return dict(zip(patients.patient_alias, fold))


def metric(labels, scores):
    labels, scores = np.asarray(labels, int), np.asarray(scores, float)
    return {"auroc": float(roc_auc_score(labels, scores)), "ap": float(average_precision_score(labels, scores))}


def cluster_interval(patients, labels, scores, draws=2000, seed=SEED):
    """Patient-cluster bootstrap 95% interval for AUROC and AP."""
    patients, labels, scores = np.asarray(patients), np.asarray(labels, int), np.asarray(scores, float)
    members = [np.flatnonzero(patients == alias) for alias in np.unique(patients)]
    rng = np.random.default_rng(seed)
    kept = []
    for _ in range(draws):
        idx = np.concatenate([members[k] for k in rng.integers(0, len(members), len(members))])
        if np.unique(labels[idx]).size == 2:
            kept.append((roc_auc_score(labels[idx], scores[idx]), average_precision_score(labels[idx], scores[idx])))
    low, high = np.quantile(kept, [.025, .975], axis=0)
    return {"auroc_95pct": [float(low[0]), float(high[0])], "ap_95pct": [float(low[1]), float(high[1])],
            "valid_bootstraps": len(kept)}


def load_cohort():
    """Return (patients, kidneys, images) of the registered paper-2 cohort."""
    patients = pd.read_csv(P2 / "cohort/patients.csv")
    kidneys = pd.read_csv(P2 / "cohort/kidneys.csv")
    images = pd.read_csv(P2 / "cohort/images.csv")
    return patients, kidneys, images
