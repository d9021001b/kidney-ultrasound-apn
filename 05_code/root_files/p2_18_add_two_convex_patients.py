"""Add the two convex patients of the (2) workbook that were not in the 383-patient registry (workbook ids redacted).

Their 5 images are in ki-data/test_predict as panel-with-mask images. Stage 'build' extracts contour caches and every
frozen feature for the 5 images with the original encoders (same code, same checkpoints) and appends the rows to the
existing arrays (rows 0-1361 are unchanged; the originals are copied to backup_before_add2 first). Stage 'cohort'
appends the patients, kidneys and images to the paper-2 cohort files. Label-blind: no APN label is read in 'build'.
Usage: python p2_18_add_two_convex_patients.py build|cohort
"""
import json, shutil, sys
from pathlib import Path
import cv2, numpy as np, pandas as pd
import nckuh_dmsa_fullcohort_fullframe_20261103 as base
from nckuh_dmsa_fullcohort_fullframe_20261103 import split_panel, crop_masked, percentile_gray
from p2_common import CACHE, EXP, P2, ROOT, WORKBOOK_V2

EXT = EXP / "NCKUH_388_added_convex_20261117"
LEG = EXP / "NCKUH_revised_383_legacy_features_20261112"
DINO = EXP / "NCKUH_revised_383_dinov3_20261111"
FOCAL = EXP / "NCKUH_revised_383_focal_echo_20261111"
GEOM = EXP / "NCKUH_revised_383_relative_geometry_20261112"
TEST = ROOT / "03_data_models/ki-data/test_predict"
# Hospital chart numbers are redacted in the public release; supply them locally.
PATIENTS = {"P384": "<chart-id-P384>", "P385": "<chart-id-P385>"}
BACKUP = EXT / "backup_before_add2"
N_OLD = 1362


def new_images():
    rows = []
    for alias, pid in PATIENTS.items():
        for path in sorted(TEST.glob(pid + "*.png")):
            view = path.stem.split("_")[-1].split("-")[0]
            if view[0] not in "LR" or view[1] not in "LT":
                raise ValueError(path.name)
            side = "left" if view.startswith("L") else "right"
            rows.append({"image_id": f"I{N_OLD + len(rows) + 1:04d}", "patient_alias": alias, "kidney_alias": f"{alias}_{side[0].upper()}",
                         "side": side, "view": view, "source_type": "panel_with_mask", "path": str(path)})
    return pd.DataFrame(rows)


def load_pair(r):
    """(frame, mask) of one added image: panel with mask, or a raw frame with a predicted component mask."""
    if isinstance(getattr(r, "mask_path", None), str):
        frame = cv2.imread(r.path)
        return frame, (cv2.imread(r.mask_path, 0) > 0).astype(np.uint8)
    return split_panel(Path(r.path))


def build(imgs=None, tag="add2"):
    EXT.mkdir(exist_ok=True)
    backup = EXT / f"backup_before_{tag}"
    if backup.exists():
        raise FileExistsError("already added")
    if imgs is None:
        imgs = new_images()
        assert len(imgs) == 5
    n_cur = int(imgs.image_id.str[1:].astype(int).min()) - 1
    ids = imgs.image_id.tolist()
    suffix = "" if tag == "add2" else "_" + tag
    # contour caches (same functions as extract_nckuh_383_contour)
    for name in ("clean_rgb", "masks", "percentile"):
        (EXT / name).mkdir(exist_ok=True)
    qc = []
    for r in imgs.itertuples():
        image, mask = load_pair(r)
        roi, roi_mask = crop_masked(image, mask)
        gray = percentile_gray(roi, roi_mask)
        for name, arr in (("clean_rgb", roi), ("masks", roi_mask * 255), ("percentile", gray)):
            assert cv2.imwrite(str(EXT / name / f"{r.image_id}.png"), arr)
        qc.append({"image_id": r.image_id, "patient_alias": r.patient_alias, "source_type": r.source_type, "height": image.shape[0], "width": image.shape[1],
                   "roi_height": roi.shape[0], "roi_width": roi.shape[1], "mask_pixels": int(mask.sum()),
                   "mask_fraction_in_source_frame": float(mask.mean()), "outside_mask_nonzero": int(np.count_nonzero(gray[roi_mask == 0]))})
    qc = pd.DataFrame(qc)
    assert qc.outside_mask_nonzero.sum() == 0
    imgs.drop(columns=[c for c in ("path", "mask_path") if c in imgs]).to_csv(EXT / f"added_image_manifest{suffix}.csv", index=False)
    qc.to_csv(EXT / f"extraction_qc_added{suffix}.csv", index=False)

    import torch
    sys.path.insert(0, str(base.RAD_SOURCE))
    import encode_nckuh_383_legacy_features_20261112 as leg
    leg.OUT, leg.CACHE = EXT / f"legacy{suffix}", EXT
    leg.OUT.mkdir(exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for fn in (leg.encode_tiles, leg.encode_usfm, leg.encode_reflection, leg.encode_patches, leg.encode_texture):
        print(fn.__name__, fn(ids), flush=True)

    # dense + radiomics (same pool code)
    sys.path.insert(0, str(ROOT / "02_experiments/CYCH_honest_reevaluation_20261003"))
    from extract_site import _radiomics_worker, dense_tight, run_pool
    jobs = [(i, EXT / "percentile" / f"{i}.png", EXT / "masks" / f"{i}.png") for i in ids]
    rad = run_pool(_radiomics_worker, jobs, processes=2)
    assert rad.index.tolist() == ids
    dense = dense_tight(EXT / "percentile", EXT / "masks", ids)
    # URI-CADS
    import nckuh_dmsa_uricads_official_frozen_20261105 as uri
    mean, std = uri.public_normalization()
    canv = []
    for i in ids:
        bgr = cv2.imread(str(EXT / "clean_rgb" / f"{i}.png"))
        m = cv2.imread(str(EXT / "masks" / f"{i}.png"), 0) > 0
        canv.append(base.tight_canvas(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), m, size=800)[0])
    canv = np.stack(canv)
    emb, sc = [], []
    for ck in uri.MODEL_PATHS:
        model = uri.OfficialGlobalEncoder(ck)
        e, s = uri.encode_batches(model, canv, mean, std, batch_size=8)
        emb.append(e.astype(np.float32)); sc.append(s.astype(np.float32))
        del model
    uri_emb, uri_sc = np.stack(emb, 1), np.stack(sc, 1)
    # DINOv3
    from transformers import AutoImageProcessor, DINOv3ViTConfig, DINOv3ViTModel
    W = ROOT / "03_data_models/dinov3_vith16plus_weights_20260913"
    cfg = DINOv3ViTConfig.from_pretrained(W, local_files_only=True)
    dm = DINOv3ViTModel.from_pretrained(W, config=cfg, local_files_only=True, dtype=torch.float32, attn_implementation="sdpa").cuda().eval().requires_grad_(False)
    proc = AutoImageProcessor.from_pretrained(W, local_files_only=True)
    cls, pat = [], []
    with torch.inference_mode():
        for i in ids:
            g = cv2.imread(str(EXT / "percentile" / f"{i}.png"), 0); m = cv2.imread(str(EXT / "masks" / f"{i}.png"), 0)
            canvas, sel = base.tight_canvas(g, m > 0)
            tok = dm(pixel_values=proc(images=[canvas], return_tensors="pt", do_resize=False)["pixel_values"].cuda()).last_hidden_state.cpu().numpy()
            cls.append(tok[0, 0])
            cov = sel.reshape(14, 16, 14, 16).mean((1, 3)).reshape(-1)
            keep = cov >= .8
            if not keep.any():
                keep[np.argmax(cov)] = True
            f = tok[0, 5:][keep]
            pat.append(np.r_[f.mean(0), f.std(0)])
    cls, pat = np.asarray(cls, np.float32), np.asarray(pat, np.float32)
    # focal echo + relative geometry
    from nckuh_dmsa_fullcohort_clinical_features_20261103 import focal_features
    from nckuh_dmsa_fullframe_relative_geometry_probe_20261102 import relative_geometry
    focal, geom = [], []
    for r in imgs.itertuples():
        frame, mask = load_pair(r)
        focal.append(focal_features(frame, mask))
        geom.append(relative_geometry(frame, mask)[0])
    focal, geom = np.asarray(focal, np.float32), np.asarray(geom, np.float32)

    # back up, then append
    targets = [CACHE / "radimagenet_dense_tight.npy", CACHE / "uricads_fold_embeddings.npy", CACHE / "uricads_fold_pathology_scores.npy",
               CACHE / "radiomics.csv", LEG / "tile_mean.npy", LEG / "usfm_gap.npy", LEG / "rad_color.npy",
               LEG / "rad_color_horizontal_reflection.npy", LEG / "radimagenet_local_float16.npy", LEG / "patch_offsets.npy",
               LEG / "directional_features.csv", DINO / "dinov3_class.npy", DINO / "dinov3_patch.npy", FOCAL / "focal_echo.npy",
               GEOM / "relative_geometry_image.npy", CACHE / "extraction_qc.csv"]
    backup.mkdir()
    for t in targets:
        shutil.copy2(t, backup / f"{t.parent.name}__{t.name}")
    cat = lambda path, new: np.save(path, np.concatenate([np.load(path), new]))
    cat(CACHE / "radimagenet_dense_tight.npy", dense)
    cat(CACHE / "uricads_fold_embeddings.npy", uri_emb)
    cat(CACHE / "uricads_fold_pathology_scores.npy", uri_sc)
    cat(LEG / "tile_mean.npy", np.load(leg.OUT / "tile_mean.npy"))
    cat(LEG / "usfm_gap.npy", np.load(leg.OUT / "usfm_gap.npy"))
    cat(LEG / "rad_color.npy", np.load(leg.OUT / "rad_color.npy"))
    cat(LEG / "rad_color_horizontal_reflection.npy", np.load(leg.OUT / "rad_color_horizontal_reflection.npy"))
    cat(DINO / "dinov3_class.npy", cls)
    cat(DINO / "dinov3_patch.npy", pat)
    cat(FOCAL / "focal_echo.npy", focal)
    cat(GEOM / "relative_geometry_image.npy", geom)
    old_feat, new_feat = np.load(LEG / "radimagenet_local_float16.npy"), np.load(leg.OUT / "radimagenet_local_float16.npy")
    old_off, new_off = np.load(LEG / "patch_offsets.npy"), np.load(leg.OUT / "patch_offsets.npy")
    assert old_off[-1] == len(old_feat) and len(old_off) == n_cur + 1
    np.save(LEG / "radimagenet_local_float16.npy", np.concatenate([old_feat, new_feat]))
    np.save(LEG / "patch_offsets.npy", np.concatenate([old_off, new_off[1:] + old_off[-1]]))
    old_t = pd.read_csv(LEG / "directional_features.csv", index_col=0)
    pd.concat([old_t, pd.read_csv(leg.OUT / "directional_features.csv", index_col=0)]).to_csv(LEG / "directional_features.csv")
    old_r = pd.read_csv(CACHE / "radiomics.csv", index_col=0)
    pd.concat([old_r, rad.reindex(columns=old_r.columns)]).to_csv(CACHE / "radiomics.csv")
    for name in ("clean_rgb", "masks", "percentile"):
        for i in ids:
            shutil.copy2(EXT / name / f"{i}.png", CACHE / name / f"{i}.png")
    pd.concat([pd.read_csv(CACHE / "extraction_qc.csv"), qc]).to_csv(CACHE / "extraction_qc.csv", index=False)
    print("appended", flush=True)


if __name__ == "__main__":
    {"build": build}[sys.argv[1]]()
