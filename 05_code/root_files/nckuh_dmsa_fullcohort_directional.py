"""Compare native-axis and contour-aligned Coiflet/GLRLM features."""
import sys

import cv2
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import nckuh_dmsa_fullcohort_fullframe_20261103 as base
from nckuh_dmsa_cluster_ci import cluster_intervals
from nckuh_dmsa_fullcohort_clinical_features_20261103 import fit_anova, load_reference
from nckuh_dmsa_inductive_graph import OUT as GRAPH

OUT = base.ROOT / "02_experiments/NCKUH_DMSA_fullcohort_directional_20261103"
sys.path.insert(0, str(base.ROOT / "02_experiments/CYCH_honest_reevaluation_20261003"))
ARMS = ("native", "aligned")
EXTRACTOR = None


def matched_canvases(gray, mask, prescribed_angle=None):
    size = int(np.ceil(np.hypot(*gray.shape))) + 8
    oy, ox = (size - gray.shape[0]) // 2, (size - gray.shape[1]) // 2
    borders = ((oy, size - gray.shape[0] - oy), (ox, size - gray.shape[1] - ox))
    y, x = np.nonzero(mask)
    if len(x) < 2:
        raise ValueError("Missing usable kidney contour")
    eigenvalues, vectors = np.linalg.eigh(np.cov(np.stack((x, y))))
    vx, vy = vectors[:, -1]
    angle = (np.degrees(np.arctan2(vy, vx)) + 90) % 180 - 90
    if eigenvalues[1] <= 1.05 * eigenvalues[0]:
        angle = 0.
    if prescribed_angle is not None:
        angle = float(prescribed_angle)
    native = np.pad(gray, borders)
    roi = np.pad(mask, borders)
    transform = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1.)
    aligned = cv2.warpAffine(native, transform, (size, size), flags=cv2.INTER_LINEAR)
    rotated_roi = cv2.warpAffine(roi, transform, (size, size), flags=cv2.INTER_NEAREST)
    aligned[rotated_roi == 0] = 0
    return (native, roi), (aligned, rotated_roi), angle


def extract_texture(image, roi):
    global EXTRACTOR
    import logging
    import SimpleITK as sitk
    import radiomics
    from radiomics import featureextractor
    from extract_site import I3_FEATURES
    if EXTRACTOR is None:
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
        radiomics.setVerbosity(logging.ERROR)
        EXTRACTOR = featureextractor.RadiomicsFeatureExtractor(
            binWidth=16, normalize=False, distances=[1, 2], force2D=False, minimumROIDimensions=2)
        EXTRACTOR.disableAllImageTypes()
        EXTRACTOR.enableImageTypeByName("Original")
        EXTRACTOR.enableImageTypeByName("Wavelet", customArgs={"wavelet": "coif1", "level": 1})
        EXTRACTOR.disableAllFeatures()
        EXTRACTOR.enableFeaturesByName(**I3_FEATURES)
    result = EXTRACTOR.execute(sitk.GetImageFromArray(image.astype(np.int16)), sitk.GetImageFromArray(roi))
    features = {k: float(v) for k, v in result.items() if not k.startswith("diagnostics")}
    assert len(features) == 25 and np.isfinite(list(features.values())).all()
    return features


def worker(job):
    ident, gray_path, mask_path, *override = job
    gray = cv2.imread(str(gray_path), 0)
    mask = (cv2.imread(str(mask_path), 0) > 0).astype(np.uint8)
    native, aligned, angle = matched_canvases(gray, mask, override[0] if override else None)
    row = {"image_id": ident, "alignment_degrees": angle}
    pairs = [("aligned", aligned)] if override else list(zip(ARMS, (native, aligned)))
    for arm, (image, roi) in pairs:
        features = extract_texture(image, roi)
        row.update({f"{arm}_{k}": v for k, v in features.items()})
    return row


def run():
    from extract_site import run_pool

    if OUT.exists():
        raise FileExistsError(OUT)
    patients, kidneys, images = load_reference()
    jobs = [(i, base.OUT / "percentile" / f"{i}.png", base.OUT / "masks" / f"{i}.png")
            for i in images.image_id]
    frame = run_pool(worker, jobs, processes=8)
    assert frame.shape == (820, 51) and frame.index.tolist() == images.image_id.tolist()
    arrays = {a: base.orient_kidneys(base.patient_pair_features(
        frame.filter(regex=f"^{a}_").to_numpy(), images, patients), patients, kidneys) for a in ARMS}
    assignments = pd.read_csv(base.OUT / "patient_assignments.csv")
    reference = pd.read_csv(GRAPH / "development_predictions.csv")
    reference = reference[reference.group.eq("knn_fusion")].set_index(["seed", "kidney_alias"])
    y = kidneys.kidney_APN.to_numpy(int)
    records = []
    for seed in base.SEEDS:
        roles = kidneys.patient_alias.map(assignments[assignments.seed.eq(seed)].set_index("patient_alias").role)
        train = np.flatnonzero(roles.eq("train"))
        dev = np.flatnonzero(roles.eq("development"))
        target = kidneys.iloc[dev]
        assert target.patient_alias.nunique() == 42 and kidneys.iloc[train].patient_alias.nunique() == 126
        ref = reference.loc[[(seed, a) for a in target.kidney_alias]]
        np.testing.assert_array_equal(ref.label, y[dev])
        np.testing.assert_array_equal(ref.patient_alias, target.patient_alias)
        scores = {"fixed_reference": ref.score.to_numpy()}
        for arm, x in arrays.items():
            scores[arm], _, _ = fit_anova(x, train, dev, y, seed, 64)
            scores[arm + "_fusion"] = .75 * scores["fixed_reference"] + .25 * scores[arm]
        for group, score in scores.items():
            records.append(target[["patient_alias", "kidney_alias", "side", "DMSA_APN"]].assign(
                seed=seed, group=group, label=y[dev], score=score))
    pred = pd.concat(records, ignore_index=True)
    ci, delta, _ = cluster_intervals(pred, list(pred.group.unique()), base.SEEDS, "fixed_reference")
    _, alignment_delta, _ = cluster_intervals(pred, list(ARMS), base.SEEDS, "native")
    OUT.mkdir()
    frame.to_csv(OUT / "directional_features.csv")
    pred.to_csv(OUT / "development_predictions.csv", index=False)
    ci.to_csv(OUT / "development_ci.csv", index=False)
    pd.concat([delta, alignment_delta]).to_csv(OUT / "paired_delta_ci.csv", index=False)
    print(ci.pivot(index="group", columns="metric", values="mean").to_string())


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        run()
