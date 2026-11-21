"""Add the 41 linear-only patients of the (2) workbook that were not in the paper-2 cohort.
36 patients had no panel or polygon contour; their contours are the predicted component masks of the linear-fine-tuned segmenter
(finetune_nckuh_383_linear_segmentation_20261112 / infer_nckuh_383_missing_linear_masks_20261112; component_review). 5 patients were
added to the workbook later and have panel-with-mask images in ki-data/test_predict. Image/DMSA-date concordance is not checked.
Usage: python p2_20_add_linear_patients.py build | register"""
import json, sys
from pathlib import Path
import cv2, numpy as np, pandas as pd
from nckuh_dmsa_fullcohort_fullframe_20261103 import split_panel, parse_source_key, normalize_patient_id
from p2_common import EXP, P2, REGISTRY, ROOT, STATE_SIDES, WORKBOOK_V2, SEED
from p2_18_add_two_convex_patients import EXT, TEST, build, PATIENTS as CONVEX_ADDED

SEG = EXP / "NCKUH_revised_383_probe_router_20261112/linear_segmentation_finetune/missing_linear_candidates"
TAG = "add41lin"
FIRST = 1368


def workbook():
    d = pd.read_excel(WORKBOOK_V2, sheet_name=6, dtype={"id": str})
    d["key"] = d.id.str.lstrip("0")
    return d.set_index("key")


def assemble():
    wb = workbook()
    link = pd.read_csv(REGISTRY / "patient_linkage_private.csv", dtype=str)
    link["key"] = link.patient_id.str.lstrip("0")
    reg = link.set_index("key").patient_alias
    known = set(reg.index)
    added_convex = {pid.lstrip("0") for pid in CONVEX_ADDED.values()}
    new_linear = [k for k in wb.index if k not in known and wb.Probe[k] == "linear"]
    assert len(new_linear) == 5, new_linear
    alias_new = {k: f"P{386 + i}" for i, k in enumerate(sorted(new_linear))}
    rows = []
    # 36 patients without contour: predicted components
    comp = pd.read_csv(SEG / "component_review/component_manifest_deidentified.csv")
    src = pd.read_csv(SEG / "source_linkage_private.csv").set_index("candidate_id").source_path
    for r in comp.itertuples():
        path = src[r.candidate_id]
        view = parse_source_key(Path(path))[2]
        rows.append({"patient_alias": r.patient_alias, "side": "left" if r.side == "L" else "right", "view": view, "path": path,
                     "mask_path": str(SEG / "component_review/masks" / f"{r.instance_id}.png"), "source_type": "predicted_component_mask"})
    # 5 patients added to the workbook: panel with mask
    for k, alias in alias_new.items():
        for path in sorted(TEST.glob(k.zfill(8) + "*.png")):
            try:
                split_panel(path)
            except Exception as e:
                print("skip", path.name, e); continue
            view = path.stem.split("_")[-1].split("-")[0]
            rows.append({"patient_alias": alias, "side": "left" if view.startswith("L") else "right", "view": view, "path": str(path),
                         "mask_path": np.nan, "source_type": "panel_with_mask"})
    imgs = pd.DataFrame(rows).sort_values(["patient_alias", "side", "view", "path"]).reset_index(drop=True)
    imgs["kidney_alias"] = imgs.patient_alias + "_" + imgs.side.str[0].str.upper()
    imgs.insert(0, "image_id", [f"I{FIRST + i:04d}" for i in range(len(imgs))])
    inv = {**reg.to_dict(), **{k: a for k, a in alias_new.items()}}
    alias_key = {a: k for k, a in inv.items()}
    imgs["workbook_key"] = imgs.patient_alias.map(alias_key)
    return imgs, wb, alias_new


def main(stage):
    imgs, wb, alias_new = assemble()
    print(len(imgs), imgs.patient_alias.nunique(), imgs.source_type.value_counts().to_dict())
    if stage == "build":
        build(imgs, TAG)
        imgs.drop(columns=["path", "mask_path"]).to_csv(EXT / "added_linear_manifest_private.csv", index=False)
    else:
        raise SystemExit("register is a separate step")


if __name__ == "__main__":
    main(sys.argv[1])
